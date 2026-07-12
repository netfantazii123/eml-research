import sympy as sp

class eml(sp.Function):
    """
    Базовая функция EML (Exponent Minus Logarithm).
    eml(x, y) = exp(x) - ln(y)
    """
    nargs = 2
    
    @classmethod
    def eval(cls, x, y):
        # Оставляем символьным по умолчанию, чтобы не сворачивалось автоматически
        pass

def to_eml_form(expr):
    """
    Рекурсивно переводит математическое выражение expr
    в форму, где все стандартные функции и константы (e, pi, 0)
    заменены на оператор eml.
    """
    if not isinstance(expr, sp.Basic):
        return expr
    
    # 1. Рекурсивно обрабатываем аргументы снизу вверх
    new_args = tuple(to_eml_form(arg) for arg in expr.args)
    if new_args:
        expr = expr.func(*new_args)
        
    # 2. Правила замены констант
    if expr == sp.E:
        return eml(1, 1)
    # ПРИМЕЧАНИЕ: замена 0 убрана намеренно — она порождает экспоненциальный
    # рост дерева (eml(1, eml(eml(1,1),1))), и при рекурсивном обходе
    # может вызвать бесконечное раздувание графа. Ноль остаётся как числовая константа.
    if expr == sp.pi:
        return -sp.I * (1 - eml(0, -1))
        
    # 3. Правила замены базовых функций
    if isinstance(expr, sp.exp):
        x = expr.args[0]
        return eml(x, 1)
    if isinstance(expr, sp.log):
        x = expr.args[0]
        return eml(1, eml(eml(1, x), 1))
    
    # 4. Правила замены тригонометрии
    if isinstance(expr, sp.cos):
        x = expr.args[0]
        return sp.Rational(1, 2) * (eml(sp.I * x, 1) + eml(-sp.I * x, 1))
    if isinstance(expr, sp.sin):
        x = expr.args[0]
        return (eml(sp.I * x, 1) - eml(-sp.I * x, 1)) / (2 * sp.I)
    
    # 5. Правила замены гиперболической тригонометрии
    if isinstance(expr, sp.cosh):
        x = expr.args[0]
        return sp.Rational(1, 2) * (eml(x, 1) + eml(-x, 1))
    if isinstance(expr, sp.sinh):
        x = expr.args[0]
        return sp.Rational(1, 2) * (eml(x, 1) - eml(-x, 1))
        
    return expr

def eval_eml(expr):
    """
    Развертывает функцию eml обратно в стандартные математические операторы,
    чтобы можно было провести верификацию и вычислить численное значение.
    """
    if not isinstance(expr, sp.Basic):
        return expr
    
    new_args = tuple(eval_eml(arg) for arg in expr.args)
    if new_args:
        expr = expr.func(*new_args)
        
    if isinstance(expr, eml):
        x, y = expr.args
        return sp.exp(x) - sp.log(y)
        
    return expr
