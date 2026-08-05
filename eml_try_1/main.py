import sympy as sp
import time
import cmath
import ctypes
import os
from numba import njit, types
from eml_core import to_eml_form, eval_eml

# ============================================================
# Loading the C library (DLL)
# ============================================================
dll_path = os.path.join(os.path.dirname(__file__), 'eml_c.dll')
try:
    # On Windows a MinGW-built DLL may depend on libgcc, libwinpthread etc.
    # Add the MinGW/bin directory so that ctypes finds those dependencies.
    mingw_bin = os.path.join(os.environ.get('MSYSTEM_PREFIX', r'C:\dev\mingw64'), 'bin')
    if os.path.isdir(mingw_bin) and hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(mingw_bin)
    eml_lib = ctypes.CDLL(dll_path)

    # --- Primitive eml(x,y) operator for element-wise calls ---
    eml_lib.eml_c.argtypes = [
        ctypes.c_double, ctypes.c_double, 
        ctypes.c_double, ctypes.c_double, 
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)
    ]
    eml_lib.eml_c.restype = None

    def eml_ctypes(x, y):
        """Call a single eml(x,y) node through C/ctypes."""
        cx, cy = complex(x), complex(y)
        out_real = ctypes.c_double()
        out_imag = ctypes.c_double()
        eml_lib.eml_c(cx.real, cx.imag, cy.real, cy.imag,
                       ctypes.byref(out_real), ctypes.byref(out_imag))
        return complex(out_real.value, out_imag.value)

    # --- Complete C graphs (one FFI call for the whole expression) ---
    _graph_names = [
        'eml_graph_exp', 'eml_graph_log', 'eml_graph_sin', 'eml_graph_cos',
        'eml_graph_complex1', 'eml_graph_trig_identity',
        'eml_graph_damped', 'eml_graph_crazy'
    ]
    _c_graphs = {}
    for _name in _graph_names:
        _fn = getattr(eml_lib, _name, None)
        if _fn is not None:
            _fn.argtypes = [
                ctypes.c_double, ctypes.c_double,
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)
            ]
            _fn.restype = None
            _c_graphs[_name] = _fn

    def call_c_graph(name, x_val):
        """Call a complete C graph by name. Returns complex or None."""
        fn = _c_graphs.get(name)
        if fn is None:
            return None
        cx = complex(x_val)
        out_r = ctypes.c_double()
        out_i = ctypes.c_double()
        fn(cx.real, cx.imag, ctypes.byref(out_r), ctypes.byref(out_i))
        return complex(out_r.value, out_i.value)

    C_LIB_OK = True
except Exception as e:
    print(f"Failed to load the C library: {e}")
    eml_ctypes = None
    call_c_graph = None
    C_LIB_OK = False


# ============================================================
# Numba JIT-compiled EML operator
# ============================================================
# NOTE: cmath inside Numba is supported through Numba's own
# implementation rather than the standard CPython cmath.
# This works for complex128, but:
#   - the behavior may change in future Numba versions
#   - for float32/float64 (not complex) use math, not cmath
#   - always check compatibility when upgrading Numba
# If Numba stops supporting cmath, replace it with:
#   return (x.real + 1j*x.imag).__exp__() - ...  (numpy fallback)
@njit(types.complex128(types.complex128, types.complex128), fastmath=True)
def eml_fast(x, y):
    """The EML operator optimized through JIT."""
    return cmath.exp(x) - cmath.log(y)


# ============================================================
# Mapping of test cases to C graph names
# ============================================================
_TEST_TO_C_GRAPH = {
    "Exponent":                                       "eml_graph_exp",
    "Logarithm":                                      "eml_graph_log",
    "Sine":                                           "eml_graph_sin",
    "Cosine":                                         "eml_graph_cos",
    "Complex equation 1":                             "eml_graph_complex1",
    "Trigonometric identity (sin^2 + cos^2)":         "eml_graph_trig_identity",
    "Damped oscillator":                              "eml_graph_damped",
    "Crazy trigonometry":                             "eml_graph_crazy",
}


def test_expression(name, expr, x_val=2.5):
    """
    Tests the conversion of an expression to EML and verifies it numerically
    with timing. Compares: sympy evalf -> Numba JIT -> C element-wise -> C full graph.
    """
    x = sp.Symbol('x')
    print("=" * 60)
    print(f"[{name}]")
    print(f"Source equation: {expr}")
    
    # Convert to EML
    t0 = time.perf_counter()
    eml_expr = to_eml_form(expr)
    t_conv = time.perf_counter() - t0
    
    print(f"EML form: {eml_expr}")
    print(f"Conversion time: {t_conv:.6f} s")
    
    # Equivalence check (numeric substitution)
    try:
        # --- The source value through sympy ---
        t1 = time.perf_counter()
        orig_val = expr.subs(x, x_val).evalf()
        t_orig = time.perf_counter() - t1
        
        # --- EML through sympy (expanded back into exp/log) ---
        t2 = time.perf_counter()
        unwrapped_eml = eval_eml(eml_expr)
        eml_val = unwrapped_eml.subs(x, x_val).evalf()
        t_eml = time.perf_counter() - t2
        
        # Clean the imaginary part of rounding errors
        diff = sp.simplify(orig_val - eml_val)
        is_close = abs(diff) < 1e-9
        
        print(f"Check at x={x_val}:")
        print(f"  Source (sympy evalf):      {orig_val} ({t_orig:.6f} s)")
        print(f"  EML (sympy evalf):         {eml_val} ({t_eml:.6f} s)")
        
        # --- JIT Numba ---
        t3 = time.perf_counter()
        fast_func = sp.lambdify(x, eml_expr, modules=[{'eml': eml_fast}, 'numpy'])
        _ = fast_func(complex(x_val))  # JIT warm-up
        t_jit_compile = time.perf_counter() - t3
        
        t4 = time.perf_counter()
        jit_val = fast_func(complex(x_val))
        t_jit = time.perf_counter() - t4
        print(f"  EML Numba JIT:             {jit_val} ({t_jit:.9f} s, "
              f"compilation: {t_jit_compile:.6f} s)")
        
        # --- C element-wise (each node is a separate ctypes FFI call) ---
        if eml_ctypes:
            t5 = time.perf_counter()
            c_func = sp.lambdify(x, eml_expr, modules=[{'eml': eml_ctypes}, 'numpy'])
            t_c_compile = time.perf_counter() - t5
            
            t6 = time.perf_counter()
            c_val = c_func(complex(x_val))
            t_c = time.perf_counter() - t6
            print(f"  EML C element-wise:        {c_val} ({t_c:.9f} s, "
                  f"lambdify: {t_c_compile:.6f} s)")
        
        # --- C full graph (one FFI call for the whole expression) ---
        c_graph_name = _TEST_TO_C_GRAPH.get(name)
        if C_LIB_OK and c_graph_name and call_c_graph:
            t7 = time.perf_counter()
            c_graph_val = call_c_graph(c_graph_name, x_val)
            t_c_graph = time.perf_counter() - t7
            if c_graph_val is not None:
                print(f"  EML C full graph:          {c_graph_val} ({t_c_graph:.9f} s)")
        
        # Summary
        if is_close:
            print("  STATUS: [OK] CORRECT")
        else:
            print(f"  STATUS: [FAIL] ERROR (difference {diff})")
            
    except Exception as e:
        print(f"  Error during verification: {e}")


def main():
    x = sp.Symbol('x')
    
    test_cases = [
        ("Exponent", sp.exp(x)),
        ("Logarithm", sp.log(x)),
        ("Sine", sp.sin(x)),
        ("Cosine", sp.cos(x)),
        ("Complex equation 1", sp.sin(x) + sp.log(x) - sp.E),
        ("Trigonometric identity (sin^2 + cos^2)", sp.sin(x)**2 + sp.cos(x)**2),
        ("Damped oscillator", sp.exp(-0.5*x) * sp.cos(2*sp.pi*x - sp.pi/4)),
        ("Crazy trigonometry", sp.sin(x)**3 - sp.cos(x/2) * sp.log(x + sp.pi) + sp.sinh(x)*sp.sin(2*x))
    ]
    
    for name, expr in test_cases:
        test_expression(name, expr)


if __name__ == '__main__':
    main()
