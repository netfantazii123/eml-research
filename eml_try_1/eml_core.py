import sympy as sp

class eml(sp.Function):
    """
    The base EML function (Exponent Minus Logarithm).
    eml(x, y) = exp(x) - ln(y)
    """
    nargs = 2

    @classmethod
    def eval(cls, x, y):
        # Keep it symbolic by default so it is not folded automatically
        pass

def to_eml_form(expr):
    """
    Recursively rewrites the mathematical expression expr into a form where all
    standard functions and constants (e, pi, 0) are replaced by the eml
    operator.
    """
    if not isinstance(expr, sp.Basic):
        return expr

    # 1. Process the arguments recursively, bottom up
    new_args = tuple(to_eml_form(arg) for arg in expr.args)
    if new_args:
        expr = expr.func(*new_args)

    # 2. Constant replacement rules
    if expr == sp.E:
        return eml(1, 1)
    # NOTE: the replacement of 0 was removed deliberately — it produces
    # exponential tree growth (eml(1, eml(eml(1,1),1))) and during a recursive
    # traversal can inflate the graph without bound. Zero stays a numeric
    # constant.
    if expr == sp.pi:
        return -sp.I * (1 - eml(0, -1))

    # 3. Base function replacement rules
    if isinstance(expr, sp.exp):
        x = expr.args[0]
        return eml(x, 1)
    if isinstance(expr, sp.log):
        x = expr.args[0]
        return eml(1, eml(eml(1, x), 1))
    
    # 4. Trigonometry replacement rules
    if isinstance(expr, sp.cos):
        x = expr.args[0]
        return sp.Rational(1, 2) * (eml(sp.I * x, 1) + eml(-sp.I * x, 1))
    if isinstance(expr, sp.sin):
        x = expr.args[0]
        return (eml(sp.I * x, 1) - eml(-sp.I * x, 1)) / (2 * sp.I)
    
    # 5. Hyperbolic trigonometry replacement rules
    if isinstance(expr, sp.cosh):
        x = expr.args[0]
        return sp.Rational(1, 2) * (eml(x, 1) + eml(-x, 1))
    if isinstance(expr, sp.sinh):
        x = expr.args[0]
        return sp.Rational(1, 2) * (eml(x, 1) - eml(-x, 1))
        
    return expr

def eval_eml(expr):
    """
    Expands the eml function back into standard mathematical operators, so that
    it can be verified and evaluated numerically.
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
