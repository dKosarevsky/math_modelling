import streamlit as st
import numpy as np
import plotly.graph_objects as go


def k(x, a, b):
    """ коэффициенты теплопроводности и теплоотдачи """
    return a / (x - b)


def alpha(x, c, d):
    return c / (x - d)


def p(x, c, d, R):
    return 2 * alpha(x, c, d) / R


def f(x, c, d, T_0, R):
    return 2 * alpha(x, c, d) * T_0 / R


def plus_half(x, a, b, h):
    """ метод средних """
    return (k(x, a, b) + k(x + h, a, b)) / 2


def minus_half(x, a, b, h):
    """ метод средних """
    return (k(x, a, b) + k(x - h, a, b)) / 2


def A(x, a, b, h):
    """ коэффициенты разностной схемы """
    return plus_half(x, a, b, h) / h


def B(x, a, b, c, d, R, h):
    return A(x, a, b, h) + C(x, a, b, h) + p(x, c, d, R) * h


def C(x, a, b, h):
    return minus_half(x, a, b, h) / h


def D(x, c, d, T_0, R, h):
    return f(x, c, d, T_0, R) * h


def left_boundary_condition(h, a, b, c, d, R, F_0, T_0):
    """ левое граничное условие """
    k_0 = plus_half(0, a, b, h) + h * h * (p(0, c, d, R) + p(h, c, d, R)) / 16 + h * h * p(0, c, d, R) / 4
    M0 = -plus_half(0, a, b, h) + h * h * (p(0, c, d, R) + p(h, c, d, R)) / 16
    P0 = h * F_0 + h * h / 4 * ((f(0, c, d, T_0, R) + f(h, c, d, T_0, R)) / 2 + f(0, c, d, T_0, R))
    return k_0, M0, P0


def right_boundary_condition(l, a, b, c, d, h, alpha_n, R, T_0):
    """ правое граничное условие """
    k_n = -minus_half(l, a, b, h) / h - alpha_n - p(l, c, d, R) * h / 4 - ((p(l, c, d, R) + p(l - h, c, d, R)) * h) / 16
    MN = minus_half(l, a, b, h) / h - ((p(l, c, d, R) + p(l - h, c, d, R)) * h) / 16
    PN = - alpha_n * T_0 - h * (f(l, c, d, T_0, R) + f(l - h, c, d, T_0, R) + f(l, c, d, T_0, R)) / 8
    return k_n, MN, PN


def plot(x: np.array, t: np.array):
    """ отрисовка графика """
    st.markdown("---")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=t,
        mode='lines',
    ))

    fig.update_layout(
        title_text=f"График зависимости температуры T(x) от координаты x:",
        xaxis_title="Длина, см",
        yaxis_title="Температура, K",
    )

    st.write(fig)
    st.markdown("---")


def main():
    st.markdown("### Лабораторная работа №4")
    st.markdown("""**Тема:**
    Программно-алгоритмическая реализация моделей на основе ОДУ второго порядка с краевыми условиями II и III рода.""")
    st.markdown("""**Цель работы:** 
    Получение навыков разработки алгоритмов решения краевой задачи при реализации моделей, 
    построенных на ОДУ второго порядка.""")

    a1, a2, a3 = st.columns(3)
    b1, b2, b3 = st.columns(3)
    c1, c2, c3 = st.columns(3)
    d1, d2, d3 = st.columns(3)

    k_0 = a1.number_input("k₀ (Вт/см К):", min_value=0., max_value=1., value=.4)
    k_n = b1.number_input("kₙ (Вт/см К):", min_value=0., max_value=1., value=.1)
    alpha_0 = c1.number_input("α₀ (Вт/см² К):", min_value=0., max_value=1., value=.05)
    alpha_n = d1.number_input("αₙ (Вт/см² К):", min_value=0., max_value=1., value=.01)

    l = a2.number_input("l (см):", min_value=1, max_value=100, value=10)
    T_0 = b2.number_input("T₀ (К):", min_value=1, max_value=1000, value=300)
    R = c2.number_input("R (см):", min_value=0., max_value=1., value=.5)
    F_0 = d2.number_input("F₀ (Вт/см²):", min_value=0, max_value=100, value=50)

    h = a3.number_input("h:", min_value=.00001, max_value=1., value=.0001, format="%.5f")

    # параметры коэффициентов теплопроводности и теплоотдачи
    b = (k_n * l) / (k_n - k_0)
    a = -k_0 * b
    d = (alpha_n * l) / (alpha_n - alpha_0)
    c = -alpha_0 * d

    k_0, M0, P0 = left_boundary_condition(h, a, b, c, d, R, F_0, T_0)
    k_n, MN, PN = right_boundary_condition(l, a, b, c, d, h, alpha_n, R, T_0)

    # прямой ход
    # массивы прогоночных коэффициентов
    eps = [0]
    eta = [0]

    eps1 = -M0 / k_0
    eta1 = P0 / k_0

    eps.append(eps1)
    eta.append(eta1)

    x = h
    n = 1
    while x + h < l:
        eps.append(C(x, a, b, h) / (B(x, a, b, c, d, R, h) - A(x, a, b, h) * eps[n]))
        eta.append((A(x, a, b, h) * eta[n] + D(x, c, d, T_0, R, h)) / (B(x, a, b, c, d, R, h) - A(x, a, b, h) * eps[n]))
        n += 1
        x += h

    # обратный ход
    t = [0] * (n + 1)

    # значение функции в последней точке
    t[n] = (PN - MN * eta[n]) / (k_n + MN * eps[n])

    for i in range(n - 1, -1, -1):
        t[i] = eps[i + 1] * t[i + 1] + eta[i + 1]

    x = [i for i in np.arange(0, l, h)]

    plot(x, t[: -1])


if __name__ == "__main__":
    main()
