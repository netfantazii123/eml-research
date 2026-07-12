"""
Компилятор: sympy EML-дерево → байткод для C стек-машины.

Рекурсивно обходит sympy-выражение после to_eml_form() и генерирует
последовательность инструкций (opcodes) для интерпретатора в eml_c.dll.
"""

import sympy as sp
from eml_core import eml

# Opcodes — должны совпадать с #define в eml_c.c
OP_CONST = 0
OP_VAR   = 1
OP_EML   = 2
OP_ADD   = 3
OP_SUB   = 4
OP_MUL   = 5
OP_DIV   = 6
OP_NEG   = 7
OP_POW   = 8


def compile_eml(expr, x_sym=None):
    """
    Компилирует sympy EML-выражение в список инструкций для C VM.
    
    Параметры:
        expr   — sympy-выражение (после to_eml_form)
        x_sym  — символ переменной (по умолчанию sp.Symbol('x'))
    
    Возвращает:
        list of tuples: [(opcode, real, imag), ...]
    """
    if x_sym is None:
        x_sym = sp.Symbol('x')
    
    code = []
    _compile_node(expr, x_sym, code)
    return code


def _compile_node(expr, x_sym, code):
    """Рекурсивный обход дерева → генерация байткода."""
    
    # 1. Переменная x
    if expr == x_sym:
        code.append((OP_VAR, 0.0, 0.0))
        return
    
    # 2. Числовая константа (включая I, Rational, Float, complex)
    #    is_number (lowercase) = True для любого численного значения без символов.
    #    ВАЖНО: is_number может вернуть True для выражений вроде -I*(1-eml(0,-1)),
    #    где все листья — числа, но eml() не вычислен. Поэтому проверяем has(eml).
    if expr.is_number and not expr.has(eml):
        val = complex(expr)
        code.append((OP_CONST, val.real, val.imag))
        return
    
    # 3. Оператор EML: eml(a, b) → compile(a), compile(b), OP_EML
    if isinstance(expr, eml):
        a, b = expr.args
        _compile_node(a, x_sym, code)
        _compile_node(b, x_sym, code)
        code.append((OP_EML, 0.0, 0.0))
        return
    
    # 4. Сложение: Add(a, b, c, ...) → a, b, ADD, c, ADD, ...
    if isinstance(expr, sp.Add):
        args = expr.args
        _compile_node(args[0], x_sym, code)
        for arg in args[1:]:
            _compile_node(arg, x_sym, code)
            code.append((OP_ADD, 0.0, 0.0))
        return
    
    # 5. Умножение: Mul(a, b, c, ...) → a, b, MUL, c, MUL, ...
    if isinstance(expr, sp.Mul):
        args = expr.args
        _compile_node(args[0], x_sym, code)
        for arg in args[1:]:
            _compile_node(arg, x_sym, code)
            code.append((OP_MUL, 0.0, 0.0))
        return
    
    # 6. Степень: Pow(a, b) → compile(a), compile(b), OP_POW
    if isinstance(expr, sp.Pow):
        base, exp = expr.args
        _compile_node(base, x_sym, code)
        _compile_node(exp, x_sym, code)
        code.append((OP_POW, 0.0, 0.0))
        return
    
    # Fallback: если попали сюда — неподдерживаемый узел
    raise ValueError(
        f"Неподдерживаемый узел sympy: {type(expr).__name__} = {expr}\n"
        f"Убедитесь, что выражение полностью преобразовано через to_eml_form()."
    )


def disassemble(bytecode):
    """
    Человекочитаемый дизассемблер байткода (для отладки).
    
    Возвращает строку с листингом инструкций.
    """
    op_names = {
        OP_CONST: "CONST",
        OP_VAR:   "VAR",
        OP_EML:   "EML",
        OP_ADD:   "ADD",
        OP_SUB:   "SUB",
        OP_MUL:   "MUL",
        OP_DIV:   "DIV",
        OP_NEG:   "NEG",
        OP_POW:   "POW",
    }
    
    lines = []
    for i, (op, r, im) in enumerate(bytecode):
        name = op_names.get(op, f"???({op})")
        if op == OP_CONST:
            if im == 0:
                lines.append(f"  {i:3d}  {name}  {r}")
            else:
                lines.append(f"  {i:3d}  {name}  {r:+.6f}{im:+.6f}i")
        else:
            lines.append(f"  {i:3d}  {name}")
    
    return "\n".join(lines)
