import streamlit as st
import numpy as np
import plotly.graph_objects as go


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
        title_text=f"График зависимости ____ от ____:",
        xaxis_title="____",
        yaxis_title="____",
    )

    st.write(fig)
    st.markdown("---")


def main():
    st.markdown("### НИРС")
    st.markdown("""**Тема:**
    Алгоритм и программная реализация метода Монте-Карло при вычислении однократных и двойных интегралов.""")
    st.markdown("""**Цель работы:** 
    Получение навыков программной реализации метода Монте-Карло при вычислении однократных и двойных интегралов.""")

    functions = """
    1. f(x) = 3 + 2x - 3x² + 2x³
    на отрезке [0, 1]

    2. f(x) = 1 / (3cos(x) + 2)

    3. f(x, y) = 4 + 2y + 2x + xy + x³y
    x -> от -1 до 1
    y -> от 0 до 3

    4. f(x, y) = y²/x²
    """

    show_schema = st.checkbox("Показать функции")
    if show_schema:
        st.code(functions)


if __name__ == "__main__":
    main()
