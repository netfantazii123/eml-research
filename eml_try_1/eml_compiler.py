"""
Compiler: a sympy EML tree -> bytecode for the C stack machine.

Recursively walks the sympy expression after to_eml_form() and generates a
sequence of instructions (opcodes) for the interpreter in eml_c.dll.
"""

import sympy as sp
from eml_core import eml

# Opcodes — must match the #define values in eml_c.c
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
    Compiles a sympy EML expression into a list of instructions for the C VM.

    Parameters:
        expr   — the sympy expression (after to_eml_form)
        x_sym  — the variable symbol (default sp.Symbol('x'))

    Returns:
        list of tuples: [(opcode, real, imag), ...]
    """
    if x_sym is None:
        x_sym = sp.Symbol('x')
    
    code = []
    _compile_node(expr, x_sym, code)
    return code


def _compile_node(expr, x_sym, code):
    """Recursive tree walk -> bytecode generation."""

    # 1. The variable x
    if expr == x_sym:
        code.append((OP_VAR, 0.0, 0.0))
        return

    # 2. A numeric constant (including I, Rational, Float, complex).
    #    is_number (lowercase) = True for any numeric value without symbols.
    #    IMPORTANT: is_number can return True for expressions such as
    #    -I*(1-eml(0,-1)), where all leaves are numbers but eml() has not been
    #    evaluated. Hence the has(eml) check.
    if expr.is_number and not expr.has(eml):
        val = complex(expr)
        code.append((OP_CONST, val.real, val.imag))
        return
    
    # 3. The EML operator: eml(a, b) -> compile(a), compile(b), OP_EML
    if isinstance(expr, eml):
        a, b = expr.args
        _compile_node(a, x_sym, code)
        _compile_node(b, x_sym, code)
        code.append((OP_EML, 0.0, 0.0))
        return
    
    # 4. Addition: Add(a, b, c, ...) -> a, b, ADD, c, ADD, ...
    if isinstance(expr, sp.Add):
        args = expr.args
        _compile_node(args[0], x_sym, code)
        for arg in args[1:]:
            _compile_node(arg, x_sym, code)
            code.append((OP_ADD, 0.0, 0.0))
        return
    
    # 5. Multiplication: Mul(a, b, c, ...) -> a, b, MUL, c, MUL, ...
    if isinstance(expr, sp.Mul):
        args = expr.args
        _compile_node(args[0], x_sym, code)
        for arg in args[1:]:
            _compile_node(arg, x_sym, code)
            code.append((OP_MUL, 0.0, 0.0))
        return
    
    # 6. Power: Pow(a, b) -> compile(a), compile(b), OP_POW
    if isinstance(expr, sp.Pow):
        base, exp = expr.args
        _compile_node(base, x_sym, code)
        _compile_node(exp, x_sym, code)
        code.append((OP_POW, 0.0, 0.0))
        return
    
    # Fallback: reaching this point means an unsupported node
    raise ValueError(
        f"Unsupported sympy node: {type(expr).__name__} = {expr}\n"
        f"Make sure the expression was fully converted through to_eml_form()."
    )


def disassemble(bytecode):
    """
    A human-readable bytecode disassembler (for debugging).

    Returns a string with the instruction listing.
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
