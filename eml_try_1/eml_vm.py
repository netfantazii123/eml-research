"""
Обёртка (ctypes) для C стек-машины EML.

Предоставляет Python-интерфейс для вычисления байткода,
скомпилированного eml_compiler.compile_eml().
"""

import ctypes
import os
import time


# Структура инструкции — должна совпадать с EmlInstruction в eml_c.c
class EmlInstruction(ctypes.Structure):
    _fields_ = [
        ("opcode",     ctypes.c_int),
        ("const_real", ctypes.c_double),
        ("const_imag", ctypes.c_double),
    ]


class EmlVM:
    """Обёртка для C-реализации стек-машины EML."""
    
    def __init__(self, dll_path=None):
        if dll_path is None:
            dll_path = os.path.join(os.path.dirname(__file__), 'eml_c.dll')
        
        # На Windows с MinGW-сборкой DLL может зависеть от libgcc и т.д.
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
        """Конвертирует список кортежей [(op, real, imag), ...] в C-массив."""
        n = len(bytecode)
        arr = (EmlInstruction * n)()
        for i, (op, r, im) in enumerate(bytecode):
            arr[i].opcode = op
            arr[i].const_real = r
            arr[i].const_imag = im
        return arr, n
    
    def eval(self, bytecode, x_val):
        """
        Вычислить байткод при заданном x.
        
        Параметры:
            bytecode — список кортежей из compile_eml()
            x_val    — значение x (float или complex)
        
        Возвращает:
            complex — результат вычисления
        
        Raises:
            RuntimeError при ошибке VM (переполнение стека и т.д.)
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
                f"VM error (code {rc}): переполнение/недобор стека "
                f"или неизвестный opcode. Длина программы: {n}"
            )
        
        return complex(out_r.value, out_i.value)
    
    def eval_timed(self, bytecode, x_val, iterations=1):
        """
        Вычислить байткод с замером времени.
        
        Возвращает:
            (result: complex, time_sec: float)
        """
        arr, n = self._to_c_program(bytecode)
        cx = complex(x_val)
        out_r = ctypes.c_double()
        out_i = ctypes.c_double()
        
        # Прогрев
        self._lib.eml_vm_eval(
            arr, n, cx.real, cx.imag,
            ctypes.byref(out_r), ctypes.byref(out_i)
        )
        
        # Замер
        t0 = time.perf_counter()
        for _ in range(iterations):
            self._lib.eml_vm_eval(
                arr, n, cx.real, cx.imag,
                ctypes.byref(out_r), ctypes.byref(out_i)
            )
        t1 = time.perf_counter()
        
        result = complex(out_r.value, out_i.value)
        return result, (t1 - t0) / iterations
