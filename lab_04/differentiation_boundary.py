import streamlit as st
import pandas as pd

from typing import List


def main():
    st.markdown("### Лабораторная работа №4")
    st.markdown("""**Тема:**
    Программно-алгоритмическая реализация моделей на основе ОДУ второго порядка с краевыми условиями II и III рода.""")
    st.markdown("""**Цель работы:** 
    Получение навыков разработки алгоритмов решения краевой задачи при реализации моделей, 
    построенных на ОДУ второго порядка.""")

    # TODO read task: https://drive.google.com/file/d/1R022ThJfPbRxiwUXq3x9ZGEUex5_j_dz/view

    # FIXME all:
    # formula = r"""
    # $$
    # y = \frac{a_0x}{a_1 + a_2x}
    # $$
    # """
    # st.markdown("""**Задание:**
    #     Задана табличная (сеточная) функция.
    #     Имеется информация, что закономерность представленная таблицей, может быть описана формулой""")
    # st.write(formula)
    # st.write("Параметры функции:")

    # a1, a2, a3 = st.columns(3)
    # b1, b2, b3 = st.columns(3)
    # c1, c2, c3 = st.columns(3)
    # d1, d2, d3 = st.columns(3)
    # e1, e2, e3 = st.columns(3)
    # f1, f2, f3 = st.columns(3)
    #
    # x1 = a1.number_input("x₁", min_value=1, max_value=10, value=1, step=1)
    # x2 = b1.number_input("x₂", min_value=1, max_value=10, value=2, step=1)
    # x3 = c1.number_input("x₃", min_value=1, max_value=10, value=3, step=1)
    # x4 = d1.number_input("x₄", min_value=1, max_value=10, value=4, step=1)
    # x5 = e1.number_input("x₅", min_value=1, max_value=10, value=5, step=1)
    # x6 = f1.number_input("x₆", min_value=1, max_value=10, value=6, step=1)
    #
    # y1 = a2.number_input("y₁:", min_value=.001, max_value=10., value=.571, format="%.4f")
    # y2 = b2.number_input("y₂", min_value=.001, max_value=10., value=.889, format="%.4f")
    # y3 = c2.number_input("y₃", min_value=.001, max_value=10., value=1.091, format="%.4f")
    # y4 = d2.number_input("y₄", min_value=.001, max_value=10., value=1.231, format="%.4f")
    # y5 = e2.number_input("y₅", min_value=.001, max_value=10., value=1.333, format="%.4f")
    # y6 = f2.number_input("y₆", min_value=.001, max_value=10., value=1.412, format="%.4f")
    #
    # h = a3.number_input("h:", min_value=1, max_value=10, value=1, step=1)
    #
    # x_array = [x1, x2, x3, x4, x5, x6]
    # y_array = [y1, y2, y3, y4, y5, y6]
    # x_array_len = len(x_array)
    # x_array_range = range(x_array_len)
    #
    # st.write("---")
    # st.write("Итоговая таблица:")
    # result_data = {
    #     "x": x_array,
    #     "y": y_array,
    #     "1": [one_sided_derivative(ln, x_array_len, y_array, h) for ln in x_array_range],
    #     "2": [central_derivative(ln, x_array_len, y_array, h) for ln in x_array_range],
    #     "3": [runge_derivative(ln, x_array_len, y_array, h) for ln in x_array_range],
    #     "4": [alignment_variables(ln, x_array_len, y_array, x_array) for ln in x_array_range],
    #     "5": [second_derivative(ln, x_array_len, y_array, h) for ln in x_array_range],
    # }
    # result = pd.DataFrame(data=result_data).applymap("{0:.4f}".format)
    # st.table(result.assign(hack="").set_index("hack"))


if __name__ == "__main__":
    main()
