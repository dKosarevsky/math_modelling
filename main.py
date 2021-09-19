import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lab_01 import polynomial_interpolation

# st.set_page_config(initial_sidebar_state="collapsed")
st.sidebar.image('logo.png', width=300)


def header():
    author = """
        made by [Kosarevsky Dmitry](https://github.com/dKosarevsky) 
        for Modelling [labs](https://github.com/dKosarevsky/)
        in [BMSTU](https://bmstu.ru)
    """
    st.markdown("# МГТУ им. Баумана. Кафедра ИУ7")
    st.markdown("## Моделирование")
    st.markdown("**Преподаватель:** Градов В.М.")
    st.markdown("**Студент:** Косаревский Д.П.")
    st.sidebar.markdown(author)


def main():
    header()
    lab = st.sidebar.radio(
        "Выберите Лабораторную работу", (
            "1. Полиномиальная интерполяция табличных функций.",
            "2. Численное интегрирование.",
            "3. Численное дифференцирование.",
            "4. Модели на основе ОДУ второго порядка с краевыми условиями II и III рода.",
        ),
        index=0
    )

    if lab[:1] == "1":
        polynomial_interpolation.main()

    # elif lab[:1] == "2":
    #     _.main()
    #
    # elif lab[:1] == "3":
    #     _.main()
    #
    # elif lab[:1] == "4":
    #     _.main()


if __name__ == "__main__":
    main()

