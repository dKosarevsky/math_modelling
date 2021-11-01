import matplotlib.pyplot as plt
from math import sin, cos, pi, exp
import streamlit as st
import pandas as pd
import numpy as np
from math import fabs

from st_aggrid import AgGrid


# FIXME
def mul_polynoms(pol1, pol2):
    res_pol = [0 for i in range(len(pol1) + len(pol2) - 1)]

    for i in range(len(pol1)):
        for j in range(len(pol2)):
            res_pol[i + j] += pol1[i] * pol2[j]

    return res_pol


def diff_polynoms(pol1, pol2):
    max_len = max(len(pol1), len(pol2))
    res_pol = [0 for i in range(max_len)]

    for i in range(len(pol1)):
        res_pol[i + len(res_pol) - len(pol1)] += pol1[i]

    for i in range(len(pol2)):
        res_pol[i + len(res_pol) - len(pol2)] -= pol2[i]

    return res_pol


def polynom_value(pol, x):
    res = 0
    for i in range(len(pol)):
        res += pol[i] * (x ** (len(pol) - i - 1))

    return res


def find_legandr_pol(n):
    leg_pol_0 = [1]  # 1
    leg_pol_1 = [1, 0]  # x

    prev = leg_pol_0
    res_pol = leg_pol_1

    for power in range(2, n + 1):
        coef_1 = [(2 * power - 1) / power, 0]  # (2 ^ power - 1 / power) * x
        coef_2 = [(power - 1) / power]  # (power - 1) / power
        mul_pol1 = mul_polynoms(coef_1, res_pol)
        mul_pol2 = mul_polynoms(coef_2, prev)
        tmp = [x for x in res_pol]
        res_pol = diff_polynoms(mul_polynoms(coef_1, res_pol), mul_polynoms(coef_2, prev))
        prev = [x for x in tmp]

    return res_pol


def half_mid_division(pol, left, right):
    mid = left + (right - left) / 2
    res = polynom_value(pol, mid)
    while fabs(res) > 1e-5:
        if res * polynom_value(pol, left) < 0:
            right = mid
        else:
            left = mid
        mid = left + (right - left) / 2
        res = polynom_value(pol, mid)

    return mid


def find_roots(leg_pol):
    n = len(leg_pol) - 1
    parts = 2 * n
    is_find_segments = False

    while not is_find_segments:
        segments = []
        step = 2 / parts

        x = -1
        for i in range(parts - 1):
            if polynom_value(leg_pol, x) * polynom_value(leg_pol, x + step) < 0 or polynom_value(leg_pol, x) == 0:
                segments.append([x, x + step])
            x += step
        if polynom_value(leg_pol, x) * polynom_value(leg_pol, 1) < 0 or polynom_value(leg_pol, x) == 0:
            segments.append([x, 1])

        if len(segments) == n:
            is_find_segments = True

    return [half_mid_division(leg_pol, seg[0], seg[1]) for seg in segments]


def solve_slau(slau):
    for i in range(len(slau)):
        tmp = slau[i][i]
        for j in range(len(slau[0])):
            slau[i][j] /= tmp

        for j in range(i + 1, len(slau)):
            tmp = slau[j][i]
            for k in range(len(slau[0])):
                slau[j][k] -= slau[i][k] * tmp

    coefs = []

    for i in range(len(slau) - 1, -1, -1):
        coef = slau[i][len(slau[0]) - 1]

        for j in range(len(coefs)):
            coef -= coefs[j] * slau[i][i + j + 1]

        coefs.insert(0, coef)

    return coefs


def find_args(roots):
    mtr = []
    for i in range(len(roots)):
        row = [root ** i for root in roots]
        if i % 2 == 1:
            row.append(0)
        else:
            row.append(2 / (i + 1))
        mtr.append(row)

    return solve_slau(mtr)


def convert_arg(t, a, b):
    return (b - a) / 2 * t + (b + a) / 2


def gauss(func, a, b, node_count):
    leg_pol = find_legandr_pol(node_count)
    roots = find_roots(leg_pol)
    args = find_args(roots)

    res = 0
    for i in range(node_count):
        res += (b - a) / 2 * args[i] * func(convert_arg(roots[i], a, b))

    return res


# FIXME
def simpson(func, a, b, nodes_count):
    h = (b - a) / (nodes_count - 1)
    x = a
    res = 0

    for i in range((nodes_count - 1) // 2):
        res += func(x) + 4 * func(x + h) + func(x + 2 * h)
        x += 2 * h

    return h / 3 * res


# FIXME
def tau_func(tau):
    sub_func = lambda phi, teta: ((2 * cos(teta)) / (1 - sin(teta) * sin(teta) * cos(phi) * cos(phi)))
    return lambda phi, teta: (4 / pi * (1 - exp(-tau * sub_func(phi, teta))) * cos(teta) * sin(teta))


def func_2_to_1(func, value):
    return lambda y: func(value, y)


def double_integreation(double_func, limits, nodes_counts, integreate_funcs):
    F = lambda y: integreate_funcs[1](func_2_to_1(double_func, y), limits[1][0], limits[1][1], nodes_counts[1])
    return integreate_funcs[0](F, limits[0][0], limits[0][1], nodes_counts[0])


def plot_graphic(integr_func, tau_start, tau_end, tau_step, label):
    plt.figure(1)

    x = []
    tau = tau_start
    while tau <= tau_end:
        x.append(tau)
        tau += tau_step

    plt.plot(x, [integr_func(tau) for tau in x], label=label)

    plt.xlabel("tau")
    plt.ylabel("Интеграция")
    plt.grid(True)
    plt.legend(loc=0)
    st.pyplot(plt)


def gen_label(func1, func2, N, M):
    label = ""
    if func1 == gauss:
        label += "ext - gauss\n"
    else:
        label += "ext - simpson\n"
    if func2 == gauss:
        label += "int - gauss\n"
    else:
        label += "int - simpson\n"

    label += f"ext nodes - {N}\nint nodes - {M}"

    return label


def test_simpson():
    func1 = gauss
    func2 = simpson

    int_func1 = lambda tau: double_integreation(tau_func(tau), [[0, pi / 2], [0, pi / 2]], [10, 9], [func1, func2])
    int_func2 = lambda tau: double_integreation(tau_func(tau), [[0, pi / 2], [0, pi / 2]], [10, 3], [func1, func2])
    int_func3 = lambda tau: double_integreation(tau_func(tau), [[0, pi / 2], [0, pi / 2]], [9, 10], [func2, func1])
    int_func4 = lambda tau: double_integreation(tau_func(tau), [[0, pi / 2], [0, pi / 2]], [3, 10], [func2, func1])

    plot_graphic(int_func1, 0.05, 10, 0.05, gen_label(func1, func2, 10, 9))
    plot_graphic(int_func2, 0.05, 10, 0.05, gen_label(func1, func2, 10, 3))
    plot_graphic(int_func3, 0.05, 10, 0.05, gen_label(func2, func1, 9, 10))
    plot_graphic(int_func4, 0.05, 10, 0.05, gen_label(func2, func1, 3, 10))


def test_gauss():
    func1 = gauss
    func2 = simpson

    int_func1 = lambda tau: double_integreation(tau_func(tau), [[0, pi / 2], [0, pi / 2]], [3, 9], [func1, func2])
    int_func2 = lambda tau: double_integreation(tau_func(tau), [[0, pi / 2], [0, pi / 2]], [10, 9], [func1, func2])
    int_func3 = lambda tau: double_integreation(tau_func(tau), [[0, pi / 2], [0, pi / 2]], [9, 3], [func2, func1])
    int_func4 = lambda tau: double_integreation(tau_func(tau), [[0, pi / 2], [0, pi / 2]], [9, 10], [func2, func1])

    plot_graphic(int_func1, 0.05, 10, 0.05, gen_label(func1, func2, 3, 9))
    plot_graphic(int_func2, 0.05, 10, 0.05, gen_label(func1, func2, 10, 9))
    plot_graphic(int_func3, 0.05, 10, 0.05, gen_label(func2, func1, 9, 3))
    plot_graphic(int_func4, 0.05, 10, 0.05, gen_label(func2, func1, 9, 10))


def main():
    st.markdown("### Лабораторная работа №2")
    st.markdown("**Тема:** Построение и программная реализация алгоритмов численного интегрирования.")
    st.markdown("""**Цель работы:** Получение навыков построения алгоритма вычисления двукратного 
        интеграла с использованием квадратурных формул Гаусса и Симпсона.""")

    st.write("---")

    c0, c1, c2 = st.columns(3)
    tau = c0.number_input("Введите значение параметра τ:", min_value=.0, max_value=1., value=.545, format="%.3f")
    N = c1.number_input("Введите значение N:", min_value=0, max_value=7, value=3, step=1)
    M = c2.number_input("Введите значение M:", min_value=0, max_value=7, value=3, step=1)

    c3, c4 = st.columns(2)
    ext_method = c3.selectbox("Выберите внешний метод", ("Гаусс", "Симпсон"))
    int_method = c4.selectbox("Выберите внутренний метод", ("Гаусс", "Симпсон"))
    func1 = gauss if ext_method == "Гаусс" else simpson
    func2 = gauss if int_method == "Гаусс" else simpson

    integr_func = lambda tau_: double_integreation(tau_func(tau), [[0, pi/2], [0, pi/2]], [N, M], [func1, func2])
    st.write(f"Результат интегрирования: {integr_func(tau)}")

    test = st.selectbox("Выберите метод тестирования", ("Гаусс", "Симпсон", "Не тестировать"))
    if test == "Не тестировать":
        plot_graphic(integr_func, 0.05, 10, 0.01,
                     gen_label(func1, func2, N, M))
    elif test == "Симпсон":
        test_simpson()
    elif test == "Гаусс":
        test_gauss()

    st.pyplot(plt)


if __name__ == "__main__":
    main()
