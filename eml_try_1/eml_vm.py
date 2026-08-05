"""
A ctypes wrapper for the C EML stack machine.

Provides a Python interface for evaluating the bytecode compiled by
eml_compiler.compile_eml().
"""

import ctypes
import os
import time


# The instruction struct — must match EmlInstruction in eml_c.c
class EmlInstruction(ctypes.Structure):
    _fields_ = [
        ("opcode",     ctypes.c_int),
        ("const_real", ctypes.c_double),
        ("const_imag", ctypes.c_double),
    ]


class EmlVM:
    """A wrapper for the C implementation of the EML stack machine."""
    
    def __init__(self, dll_path=None):
        if dll_path is None:
            dll_path = os.path.join(os.path.dirname(__file__), 'eml_c.dll')
        
        # On Windows a MinGW-built DLL may depend on libgcc and so on.
        mingw_bin = os.path.join(
            os.environ.get('MSYSTEM_PREFIX', r'C:\dev\mingw64'), 'bin'
        )
        if os.path.isdir(mingw_bin) and hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(mingw_bin)
        
        self._lib = ctypes.CDLL(dll_path)
        
        # int eml_vm_eval(const EmlInstruction*, int, double, double,
        #                 double*, double*)
        self._lib.eml_vm_eval.argtypes = [
            ctypes.POINTER(EmlInstruction),
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self._lib.eml_vm_eval.restype = ctypes.c_int
    
    def _to_c_program(self, bytecode):
        """Converts a list of tuples [(op, real, imag), ...] into a C array."""
        n = len(bytecode)
        arr = (EmlInstruction * n)()
        for i, (op, r, im) in enumerate(bytecode):
            arr[i].opcode = op
            arr[i].const_real = r
            arr[i].const_imag = im
        return arr, n
    
    def eval(self, bytecode, x_val):
        """
        Evaluate the bytecode at the given x.

        Parameters:
            bytecode — the list of tuples from compile_eml()
            x_val    — the value of x (float or complex)

        Returns:
            complex — the evaluation result

        Raises:
            RuntimeError on a VM error (stack overflow and so on)
        """
        arr, n = self._to_c_program(bytecode)
        cx = complex(x_val)
        out_r = ctypes.c_double()
        out_i = ctypes.c_double()
        
        rc = self._lib.eml_vm_eval(
            arr, n,
            cx.real, cx.imag,
            ctypes.byref(out_r), ctypes.byref(out_i)
        )
        
        if rc != 0:
            raise RuntimeError(
                f"VM error (code {rc}): stack overflow/underflow "
                f"or an unknown opcode. Program length: {n}"
            )
        
        return complex(out_r.value, out_i.value)
    
    def eval_timed(self, bytecode, x_val, iterations=1):
        """
        Evaluate the bytecode while measuring the time.

        Returns:
            (result: complex, time_sec: float)
        """
        arr, n = self._to_c_program(bytecode)
        cx = complex(x_val)
        out_r = ctypes.c_double()
        out_i = ctypes.c_double()
        
        # Warm-up
        self._lib.eml_vm_eval(
            arr, n, cx.real, cx.imag,
            ctypes.byref(out_r), ctypes.byref(out_i)
        )
        
        # Measurement
        t0 = time.perf_counter()
        for _ in range(iterations):
            self._lib.eml_vm_eval(
                arr, n, cx.real, cx.imag,
                ctypes.byref(out_r), ctypes.byref(out_i)
            )
        t1 = time.perf_counter()
        
        result = complex(out_r.value, out_i.value)
        return result, (t1 - t0) / iterations
