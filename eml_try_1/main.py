import sympy as sp
import time
import cmath
import ctypes
import os
from numba import njit, types
from eml_core import to_eml_form, eval_eml

# ============================================================
# Загрузка C-библиотеки (DLL)
# ============================================================
dll_path = os.path.join(os.path.dirname(__file__), 'eml_c.dll')
try:
    # На Windows с MinGW-сборкой DLL может зависеть от libgcc, libwinpthread и т.д.
    # Добавляем каталог MinGW/bin чтобы ctypes нашёл эти зависимости.
    mingw_bin = os.path.join(os.environ.get('MSYSTEM_PREFIX', r'C:\dev\mingw64'), 'bin')
    if os.path.isdir(mingw_bin) and hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(mingw_bin)
    eml_lib = ctypes.CDLL(dll_path)

    # --- Примитивный оператор eml(x,y) для поэлементных вызовов ---
    eml_lib.eml_c.argtypes = [
        ctypes.c_double, ctypes.c_double, 
        ctypes.c_double, ctypes.c_double, 
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)
    ]
    eml_lib.eml_c.restype = None

    def eml_ctypes(x, y):
        """Вызов одного узла eml(x,y) через C/ctypes."""
        cx, cy = complex(x), complex(y)
        out_real = ctypes.c_double()
        out_imag = ctypes.c_double()
        eml_lib.eml_c(cx.real, cx.imag, cy.real, cy.imag,
                       ctypes.byref(out_real), ctypes.byref(out_imag))
        return complex(out_real.value, out_imag.value)

    # --- Полные C-графы (один FFI-вызов на всё выражение) ---
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
        """Вызов полного C-графа по имени. Возвращает complex или None."""
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
    print(f"Ошибка загрузки C-библиотеки: {e}")
    eml_ctypes = None
    call_c_graph = None
    C_LIB_OK = False


# ============================================================
# Numba JIT-компилированный оператор EML
# ============================================================
# ВНИМАНИЕ: cmath внутри Numba поддерживается через внутреннюю
# реализацию Numba, а не через стандартный CPython cmath.
# Это работает для complex128, но:
#   - Поведение может измениться в будущих версиях Numba
#   - Для float32/float64 (не complex) — используйте math, не cmath
#   - При обновлении Numba обязательно проверяйте совместимость
# Если Numba перестанет поддерживать cmath — замените на:
#   return (x.real + 1j*x.imag).__exp__() - ...  (numpy fallback)
@njit(types.complex128(types.complex128, types.complex128), fastmath=True)
def eml_fast(x, y):
    """Оптимизированный через JIT оператор EML."""
    return cmath.exp(x) - cmath.log(y)


# ============================================================
# Маппинг тест-кейсов на имена C-графов
# ============================================================
_TEST_TO_C_GRAPH = {
    "Экспонента":                                     "eml_graph_exp",
    "Логарифм":                                       "eml_graph_log",
    "Синус":                                          "eml_graph_sin",
    "Косинус":                                        "eml_graph_cos",
    "Сложное уравнение 1":                            "eml_graph_complex1",
    "Тригонометрическое тождество (sin^2 + cos^2)":   "eml_graph_trig_identity",
    "Затухающий осциллятор":                           "eml_graph_damped",
    "Безумная тригонометрия":                          "eml_graph_crazy",
}


def test_expression(name, expr, x_val=2.5):
    """
    Тестирует перевод выражения в EML и верифицирует его численно с замером времени.
    Сравнивает: sympy evalf → Numba JIT → C поэлементно → C полный граф.
    """
    x = sp.Symbol('x')
    print("=" * 60)
    print(f"[{name}]")
    print(f"Исходное уравнение: {expr}")
    
    # Переводим в EML
    t0 = time.perf_counter()
    eml_expr = to_eml_form(expr)
    t_conv = time.perf_counter() - t0
    
    print(f"EML форма: {eml_expr}")
    print(f"Время конвертации: {t_conv:.6f} сек")
    
    # Проверка эквивалентности (численная подстановка)
    try:
        # --- Исходное значение через sympy ---
        t1 = time.perf_counter()
        orig_val = expr.subs(x, x_val).evalf()
        t_orig = time.perf_counter() - t1
        
        # --- EML через sympy (развёртка обратно в exp/log) ---
        t2 = time.perf_counter()
        unwrapped_eml = eval_eml(eml_expr)
        eml_val = unwrapped_eml.subs(x, x_val).evalf()
        t_eml = time.perf_counter() - t2
        
        # Очистка мнимой части от ошибок округления
        diff = sp.simplify(orig_val - eml_val)
        is_close = abs(diff) < 1e-9
        
        print(f"Проверка при x={x_val}:")
        print(f"  Исходное (sympy evalf):    {orig_val} ({t_orig:.6f} сек)")
        print(f"  EML (sympy evalf):         {eml_val} ({t_eml:.6f} сек)")
        
        # --- JIT Numba ---
        t3 = time.perf_counter()
        fast_func = sp.lambdify(x, eml_expr, modules=[{'eml': eml_fast}, 'numpy'])
        _ = fast_func(complex(x_val))  # Прогрев JIT
        t_jit_compile = time.perf_counter() - t3
        
        t4 = time.perf_counter()
        jit_val = fast_func(complex(x_val))
        t_jit = time.perf_counter() - t4
        print(f"  EML Numba JIT:             {jit_val} ({t_jit:.9f} сек, "
              f"компиляция: {t_jit_compile:.6f} сек)")
        
        # --- C поэлементно (через ctypes, каждый узел — отдельный FFI-вызов) ---
        if eml_ctypes:
            t5 = time.perf_counter()
            c_func = sp.lambdify(x, eml_expr, modules=[{'eml': eml_ctypes}, 'numpy'])
            t_c_compile = time.perf_counter() - t5
            
            t6 = time.perf_counter()
            c_val = c_func(complex(x_val))
            t_c = time.perf_counter() - t6
            print(f"  EML C поэлементно:         {c_val} ({t_c:.9f} сек, "
                  f"lambdify: {t_c_compile:.6f} сек)")
        
        # --- C полный граф (один FFI-вызов на всё выражение) ---
        c_graph_name = _TEST_TO_C_GRAPH.get(name)
        if C_LIB_OK and c_graph_name and call_c_graph:
            t7 = time.perf_counter()
            c_graph_val = call_c_graph(c_graph_name, x_val)
            t_c_graph = time.perf_counter() - t7
            if c_graph_val is not None:
                print(f"  EML C полный граф:         {c_graph_val} ({t_c_graph:.9f} сек)")
        
        # Итог
        if is_close:
            print("  СТАТУС: [OK] ВЕРНО")
        else:
            print(f"  СТАТУС: [FAIL] ОШИБКА (разница {diff})")
            
    except Exception as e:
        print(f"  Ошибка при верификации: {e}")


def main():
    x = sp.Symbol('x')
    
    test_cases = [
        ("Экспонента", sp.exp(x)),
        ("Логарифм", sp.log(x)),
        ("Синус", sp.sin(x)),
        ("Косинус", sp.cos(x)),
        ("Сложное уравнение 1", sp.sin(x) + sp.log(x) - sp.E),
        ("Тригонометрическое тождество (sin^2 + cos^2)", sp.sin(x)**2 + sp.cos(x)**2),
        ("Затухающий осциллятор", sp.exp(-0.5*x) * sp.cos(2*sp.pi*x - sp.pi/4)),
        ("Безумная тригонометрия", sp.sin(x)**3 - sp.cos(x/2) * sp.log(x + sp.pi) + sp.sinh(x)*sp.sin(2*x))
    ]
    
    for name, expr in test_cases:
        test_expression(name, expr)


if __name__ == '__main__':
    main()
