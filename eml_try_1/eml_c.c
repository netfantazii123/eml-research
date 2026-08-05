#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================
 * EML (Exponent Minus Logarithm) — the base operator
 * eml(x, y) = exp(x) - ln(y)
 * ============================================================ */

static inline double complex eml(double complex x, double complex y) {
    return cexp(x) - clog(y);
}

/* ============================================================
 * DLL export
 * ============================================================ */

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

/* --- Primitive operator (for compatibility with Python ctypes) --- */
EXPORT void eml_c(double real_x, double imag_x,
                  double real_y, double imag_y,
                  double* out_real, double* out_imag)
{
    double complex x = real_x + imag_x * I;
    double complex y = real_y + imag_y * I;
    double complex res = eml(x, y);
    *out_real = creal(res);
    *out_imag = cimag(res);
}

/* ============================================================
 * Stack machine (VM) for evaluating arbitrary EML graphs
 *
 * Python compiles a sympy EML tree into an instruction array,
 * passes it through ctypes, and C interprets it in a single FFI call.
 * ============================================================ */

#define OP_CONST  0   /* push constant (real + imag)          */
#define OP_VAR    1   /* push current value of x              */
#define OP_EML    2   /* pop y, pop x, push eml(x,y)          */
#define OP_ADD    3   /* pop b, pop a, push a+b               */
#define OP_SUB    4   /* pop b, pop a, push a-b               */
#define OP_MUL    5   /* pop b, pop a, push a*b               */
#define OP_DIV    6   /* pop b, pop a, push a/b               */
#define OP_NEG    7   /* pop a, push -a                       */
#define OP_POW    8   /* pop b, pop a, push cpow(a,b)         */

#define VM_STACK_SIZE 256

typedef struct {
    int    opcode;
    double const_real;
    double const_imag;
} EmlInstruction;

/*
 * eml_vm_eval — executes the bytecode on the stack machine.
 *
 * Parameters:
 *   program   — the instruction array (bytecode)
 *   prog_len  — the program length
 *   x_real/x_imag — the value of the variable x
 *   out_real/out_imag — the result (stack top after execution)
 *
 * Returns 0 on success, -1 on error (stack overflow/underflow).
 */
EXPORT int eml_vm_eval(const EmlInstruction* program, int prog_len,
                       double x_real, double x_imag,
                       double* out_real, double* out_imag)
{
    double complex stack[VM_STACK_SIZE];
    int sp = 0;
    double complex x = x_real + x_imag * I;

    for (int i = 0; i < prog_len; i++) {
        switch (program[i].opcode) {
            case OP_CONST:
                if (sp >= VM_STACK_SIZE) return -1;
                stack[sp++] = program[i].const_real
                            + program[i].const_imag * I;
                break;

            case OP_VAR:
                if (sp >= VM_STACK_SIZE) return -1;
                stack[sp++] = x;
                break;

            case OP_EML: {
                if (sp < 2) return -1;
                double complex y  = stack[--sp];
                double complex xx = stack[--sp];
                stack[sp++] = cexp(xx) - clog(y);
                break;
            }
            case OP_ADD: {
                if (sp < 2) return -1;
                double complex b = stack[--sp];
                double complex a = stack[--sp];
                stack[sp++] = a + b;
                break;
            }
            case OP_SUB: {
                if (sp < 2) return -1;
                double complex b = stack[--sp];
                double complex a = stack[--sp];
                stack[sp++] = a - b;
                break;
            }
            case OP_MUL: {
                if (sp < 2) return -1;
                double complex b = stack[--sp];
                double complex a = stack[--sp];
                stack[sp++] = a * b;
                break;
            }
            case OP_DIV: {
                if (sp < 2) return -1;
                double complex b = stack[--sp];
                double complex a = stack[--sp];
                stack[sp++] = a / b;
                break;
            }
            case OP_NEG: {
                if (sp < 1) return -1;
                stack[sp - 1] = -stack[sp - 1];
                break;
            }
            case OP_POW: {
                if (sp < 2) return -1;
                double complex b = stack[--sp];
                double complex a = stack[--sp];
                stack[sp++] = cpow(a, b);
                break;
            }
            default:
                return -1;  /* unknown opcode */
        }
    }

    if (sp != 1) return -1;  /* the stack must hold exactly 1 value */

    *out_real = creal(stack[0]);
    *out_imag = cimag(stack[0]);
    return 0;
}

/* ============================================================
 * Test EML graphs (complete expressions)
 *
 * Every function evaluates a whole mathematical expression,
 * using ONLY the eml() operator recursively.
 * ============================================================ */

/* exp(x)  →  eml(x, 1) */
static double complex eml_test_exp(double complex x) {
    return eml(x, 1);
}

/* log(x)  →  eml(1, eml(eml(1, x), 1)) */
static double complex eml_test_log(double complex x) {
    return eml(1, eml(eml(1, x), 1));
}

/* sin(x)  →  (eml(ix, 1) - eml(-ix, 1)) / (2i) */
static double complex eml_test_sin(double complex x) {
    return (eml(I * x, 1) - eml(-I * x, 1)) / (2.0 * I);
}

/* cos(x)  →  0.5 * (eml(ix, 1) + eml(-ix, 1)) */
static double complex eml_test_cos(double complex x) {
    return 0.5 * (eml(I * x, 1) + eml(-I * x, 1));
}

/* sin(x) + log(x) - e */
static double complex eml_test_complex1(double complex x) {
    double complex s = (eml(I * x, 1) - eml(-I * x, 1)) / (2.0 * I);
    double complex l = eml(1, eml(eml(1, x), 1));
    double complex e = eml(1, 1);
    return s + l - e;
}

/* sin^2(x) + cos^2(x)   (must equal 1) */
static double complex eml_test_trig_identity(double complex x) {
    double complex s = (eml(I * x, 1) - eml(-I * x, 1)) / (2.0 * I);
    double complex c = 0.5 * (eml(I * x, 1) + eml(-I * x, 1));
    return s * s + c * c;
}

/* exp(-0.5x) * cos(2*pi*x - pi/4)  — a damped oscillator */
static double complex eml_test_damped(double complex x) {
    double complex pi_val = -I * (1.0 - eml(0, -1));
    double complex decay = eml(-0.5 * x, 1);
    double complex phase = 2.0 * pi_val * x - pi_val / 4.0;
    double complex osc = 0.5 * (eml(I * phase, 1) + eml(-I * phase, 1));
    return decay * osc;
}

/* sin³(x) - cos(x/2)*log(x+π) + sinh(x)*sin(2x) */
static double complex eml_test_crazy(double complex x) {
    double complex pi_val = -I * (1.0 - eml(0, -1));

    double complex sx  = (eml(I * x, 1) - eml(-I * x, 1)) / (2.0 * I);
    double complex cx2 = 0.5 * (eml(I * x / 2.0, 1) + eml(-I * x / 2.0, 1));

    double complex xpi  = x + pi_val;
    double complex lxpi = eml(1, eml(eml(1, xpi), 1));

    double complex shx = 0.5 * (eml(x, 1) - eml(-x, 1));
    double complex s2x = (eml(2.0 * I * x, 1) - eml(-2.0 * I * x, 1)) / (2.0 * I);

    return sx * sx * sx - cx2 * lxpi + shx * s2x;
}

/* ============================================================
 * DLL export of the complete graphs
 * (lets Python evaluate a whole graph in a single FFI call)
 * ============================================================ */

EXPORT void eml_graph_exp(double rx, double ix, double* out_r, double* out_i) {
    double complex r = eml_test_exp(rx + ix * I);
    *out_r = creal(r); *out_i = cimag(r);
}

EXPORT void eml_graph_log(double rx, double ix, double* out_r, double* out_i) {
    double complex r = eml_test_log(rx + ix * I);
    *out_r = creal(r); *out_i = cimag(r);
}

EXPORT void eml_graph_sin(double rx, double ix, double* out_r, double* out_i) {
    double complex r = eml_test_sin(rx + ix * I);
    *out_r = creal(r); *out_i = cimag(r);
}

EXPORT void eml_graph_cos(double rx, double ix, double* out_r, double* out_i) {
    double complex r = eml_test_cos(rx + ix * I);
    *out_r = creal(r); *out_i = cimag(r);
}

EXPORT void eml_graph_complex1(double rx, double ix, double* out_r, double* out_i) {
    double complex r = eml_test_complex1(rx + ix * I);
    *out_r = creal(r); *out_i = cimag(r);
}

EXPORT void eml_graph_trig_identity(double rx, double ix, double* out_r, double* out_i) {
    double complex r = eml_test_trig_identity(rx + ix * I);
    *out_r = creal(r); *out_i = cimag(r);
}

EXPORT void eml_graph_damped(double rx, double ix, double* out_r, double* out_i) {
    double complex r = eml_test_damped(rx + ix * I);
    *out_r = creal(r); *out_i = cimag(r);
}

EXPORT void eml_graph_crazy(double rx, double ix, double* out_r, double* out_i) {
    double complex r = eml_test_crazy(rx + ix * I);
    *out_r = creal(r); *out_i = cimag(r);
}

/* ============================================================
 * Benchmark utilities
 * ============================================================ */

#ifndef EML_BUILD_DLL

typedef double complex (*test_fn)(double complex);

static void run_test(const char* name, test_fn fn, double complex x,
                     double complex expected)
{
    double complex got = fn(x);
    double err = cabs(got - expected);
    int ok = err < 1e-9;

    printf("  %-28s  EML = %+.10f %+.10fi", name, creal(got), cimag(got));
    if (ok)
        printf("  [OK]\n");
    else
        printf("  [ERR diff=%.2e]  expected %+.10f %+.10fi\n",
               err, creal(expected), cimag(expected));
}

/* ============================================================
 * main() — standalone test runner
 * ============================================================ */

int main(void) {
    double complex x = 2.5 + 0.0 * I;
    double pi_std = acos(-1.0);

    printf("================================================================\n");
    printf("  EML C Native Tests  |  x = %.1f\n", creal(x));
    printf("================================================================\n\n");

    /* Expected values (standard math functions) */
    double complex exp_std  = cexp(x);
    double complex log_std  = clog(x);
    double complex sin_std  = csin(x);
    double complex cos_std  = ccos(x);
    double complex c1_std   = csin(x) + clog(x) - M_E;
    double complex trig_std = 1.0 + 0.0 * I;

    double complex damp_std = cexp(-0.5 * x) *
                              ccos(2.0 * pi_std * x - pi_std / 4.0);

    double complex sx_std   = csin(x);
    double complex cx2_std  = ccos(x / 2.0);
    double complex lxpi_std = clog(x + pi_std);
    double complex shx_std  = csinh(x);
    double complex s2x_std  = csin(2.0 * x);
    double complex crazy_std = sx_std * sx_std * sx_std
                             - cx2_std * lxpi_std
                             + shx_std * s2x_std;

    run_test("exp(x)",                    eml_test_exp,            x, exp_std);
    run_test("log(x)",                    eml_test_log,            x, log_std);
    run_test("sin(x)",                    eml_test_sin,            x, sin_std);
    run_test("cos(x)",                    eml_test_cos,            x, cos_std);
    run_test("sin(x)+log(x)-e",           eml_test_complex1,       x, c1_std);
    run_test("sin^2+cos^2 (=1)",          eml_test_trig_identity,  x, trig_std);
    run_test("damped oscillator",         eml_test_damped,         x, damp_std);
    run_test("crazy trig",                eml_test_crazy,          x, crazy_std);

    /* --- Benchmark: 1M iterations of "crazy trig" --- */
    printf("\n--- Benchmark: 1 000 000 iterations of 'crazy trig' ---\n");
    int N = 1000000;
    clock_t t0 = clock();
    volatile double complex sink;
    for (int i = 0; i < N; i++) {
        sink = eml_test_crazy(x);
    }
    clock_t t1 = clock();
    double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;
    printf("  Total: %.4f sec  |  Per call: %.3f us\n", elapsed, elapsed / N * 1e6);
    (void)sink;

    printf("\nDone.\n");
    return 0;
}

#endif /* EML_BUILD_DLL */
