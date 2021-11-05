import matplotlib.pyplot as plt
from math import sin, cos, pi, exp
import streamlit as st
import numpy as np
from numpy import linspace, array
from numpy.linalg import solve
from typing import List
from typing import Callable as func


def simpson_integrate(f: func, a: float, b: float, n: int) -> float:
    h, res = (b - a) / n, 0
    for i in range(0, n, 2):
        x1, x2, x3 = i * h, (i + 1) * h, (i + 2) * h
        f1, f2, f3 = f(x1), f(x2), f(x3)
        res += f1 + 4 * f2 + f3
    return h / 3 * res


# Возвращает значение полинома Лежандра n-го порядка
def legendre(n: int, x: float) -> float:
    if n < 2: return [1, x][n]
    P1, P2 = legendre(n - 1, x), legendre(n - 2, x)
    return ((2 * n - 1) * x * P1 - (n - 1) * P2) / n


# возвращает значение производной полинома Лежандра
def legendre_prime(n: int, x: float) -> float:
    P1, P2 = legendre(n - 1, x), legendre(n, x)
    return n / (1 - x * x) * (P1 - x * P2)


# Нахождение корней полинома Лежандра n-го порядка
def legendre_roots(n: int, eps: float = 1e-12) -> List[float]:
    roots = [cos(pi * (4 * i + 3) / (4 * n + 2)) for i in range(n)]
    for i, root in enumerate(roots):  # уточнение корней
        root_val = legendre(n, root)
        while abs(root_val) > eps:
            root -= root_val / legendre_prime(n, root)
            root_val = legendre(n, root)
        roots[i] = root
    return roots


# Метод Гаусса для численного интегрирования на [-1; 1]
def gauss_integrate_norm(f: func, n: int) -> float:
    t = legendre_roots(n)
    T = array([[t_i**k for t_i in t] for k in range(n)])

    int_tk = lambda k: 2 / (k + 1) if k % 2 == 0 else 0
    b = array([int_tk(k) for k in range(n)])
    A = solve(T, b)  # решение системы линейных уравнений

    return sum(A_i * f(t_i) for A_i, t_i in zip(A, t))


# Метод Гаусса для произвольного промежутка [a; b]
def gauss_integrate(f: func, a: float, b: float, n: int) -> float:
    mean, diff = (a + b) / 2, (b - a) / 2
    g = lambda t: f(mean + diff * t)
    return diff * gauss_integrate_norm(g, n)


def composite_integrate(f: func, a1: float, b1: float, a2: float, b2: float,
                        method_1: func, method_2: func, n1: int, n2: int) -> float:
    F = lambda y: method_1(lambda x: f(x, y), a1, b1, n1)
    return method_2(F, a2, b2, n2)


def function_integrator(f: func, a: float, b: float, c: float, d: float, n: int, m: int) -> float:
    return composite_integrate(f, a, b, c, d, gauss_integrate, simpson_integrate, n, m)


def function(t: float, n: int, m: int) -> float:
    L_R = lambda theta, phi: 2 * cos(theta) / (1 - sin(theta)**2 * cos(phi)**2)
    f = lambda theta, phi: (1 - exp(-t * L_R(theta, phi))) * cos(theta) * sin(theta)

    return 4 / pi * function_integrator(f, 0, pi / 2, 0, pi / 2, n, m)


def test_function(func):
    def test_f(x):
        return 0.3 * (x - 1) ** 2 - 0.1 * (x + 4) ** 2 - x

    def act_int_f(x):
        return 0.1 * (x - 1) ** 3 - 0.1 / 3 * (x + 4) ** 3 - 0.5 * x ** 2

    X = linspace(-10, 10, 100)
    Y = act_int_f(X) - act_int_f(0 * X)
    F = lambda n: [func(test_f, 0, x, n) for x in X]

    plt.plot(X, Y)
    plt.plot(X, F(1))
    st.pyplot(plt)


def main():
    st.markdown("### Лабораторная работа №2")
    st.markdown("**Тема:** Построение и программная реализация алгоритмов численного интегрирования.")
    st.markdown("""**Цель работы:** Получение навыков построения алгоритма вычисления двукратного 
        интеграла с использованием квадратурных формул Гаусса и Симпсона.""")

    st.write("---")

    c0, c1, c2 = st.columns(3)
    tau = c0.number_input("Введите значение параметра τ:", min_value=0, max_value=1, value=1)
    N = c1.number_input("Введите значение N:", min_value=0, max_value=100, value=3, step=1)
    M = c2.number_input("Введите значение M:", min_value=0, max_value=100, value=12, step=1)

    # c3, c4 = st.columns(2)
    # ext_method = c3.selectbox("Выберите внешний метод", ("Гаусс", "Симпсон"))
    # int_method = c4.selectbox("Выберите внутренний метод", ("Гаусс", "Симпсон"))
    # func1 = gauss if ext_method == "Гаусс" else simpson
    # func2 = gauss if int_method == "Гаусс" else simpson

    # integr_func = lambda tau_: double_integreation(tau_func(tau), [[0, pi/2], [0, pi/2]], [N, M], [func1, func2])
    integr_func = function(tau, N, M)
    st.write(f"Результат интегрирования: {integr_func}")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    major_ticks = linspace(0, 10, 6)
    minor_ticks = linspace(0, 10, 21)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(linspace(0, 1, 6))
    ax.set_yticks(linspace(0, 1, 11), minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.4)
    ax.grid(which='major', alpha=0.8)

    tao = linspace(0.05, 10, 100)
    eps = [function(t, N, M) for t in tao]
    ax.plot(tao, eps, label="ɛ(τ)")

    plt.legend()
    plt.xlabel("τ")
    plt.ylabel("ɛ").set_rotation(0)
    st.pyplot(plt)

    test = st.selectbox("Выберите метод тестирования", ("Гаусс", "Симпсон"))
    if test == "Симпсон":
        test_function(simpson_integrate)
    elif test == "Гаусс":
        test_function(gauss_integrate)


if __name__ == "__main__":
    main()
