import plotly.graph_objects as go
import streamlit as st
import numpy as np

from typing import List, Callable


def plot(func: Callable, n: int, m: int, test: bool = False, test_func: Callable = None, test_func2: Callable = None):
    """ отрисовка графика """
    st.markdown("---")
    x = np.arange(.05, 10, .001) if not test else np.arange(-10, 10, .001)
    y = [func(t, n, m) for t in x] if not test else test_func2(x) - test_func2(0 * x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
    ))

    if test:
        def test_y(n_test):
            return [func(test_func, 0, x_test, n_test) for x_test in x]

        fig.add_trace(go.Scatter(
            x=x,
            y=test_y(1),
            mode='lines',
        ))

    fig.update_layout(
        title_text=f"График бла бла" if not test else "График бла",
        xaxis_title="Икс" if not test else "",
        yaxis_title="Игрек" if not test else "",
        showlegend=False
    )

    st.write(fig)
    st.markdown("---")


def main():
    st.markdown("### Лабораторная работа №3")
    st.markdown("**Тема:** Построение и программная реализация алгоритмов численного дифференцирования.")
    st.markdown("""**Цель работы:** 
    Получение навыков построения алгоритма вычисления производных от сеточных функций.""")

    st.write("---")

    # c0, c1, c2 = st.columns(3)
    # tau = c0.number_input("Введите значение параметра τ:", min_value=.0, max_value=1., value=.808, format="%.3f")
    # N = c1.number_input("Введите значение N:", min_value=1, max_value=100, value=4, step=1)
    # M = c2.number_input("Введите значение M:", min_value=1, max_value=100, value=5, step=1)
    #
    # result = integrate_function(tau, N, M)
    # st.write(f"Результат интегрирования: {round(result, 5)}")

    # plot(integrate_function, N, M)


if __name__ == "__main__":
    main()
