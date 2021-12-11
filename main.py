import streamlit as st

from lab_01 import polynomial_interpolation
from lab_02 import numerical_integration
from lab_03 import numerical_differentiation
from lab_04 import differentiation_boundary
from nirs import nirs

# st.set_page_config(initial_sidebar_state="collapsed")
st.sidebar.image('logo.png', width=300)


def header():
    author = """
        made by [Kosarevsky Dmitry](https://github.com/dKosarevsky) 
        for [Modelling](https://github.com/dKosarevsky/iu7/blob/master/7sem/modeling.md) labs
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
        "Выберите Лабораторную работу или НИРС:", (
            "1. Полиномиальная интерполяция табличных функций.",
            "2. Численное интегрирование.",
            "3. Численное дифференцирование.",
            "4. Модели на основе ОДУ второго порядка с краевыми условиями II и III рода.",
            "5. НИРС.",
        ),
        index=4
    )

    if lab[:1] == "1":
        polynomial_interpolation.main()

    elif lab[:1] == "2":
        numerical_integration.main()

    elif lab[:1] == "3":
        numerical_differentiation.main()

    elif lab[:1] == "4":
        differentiation_boundary.main()

    elif lab[:1] == "5":
        nirs.main()


if __name__ == "__main__":
    main()

