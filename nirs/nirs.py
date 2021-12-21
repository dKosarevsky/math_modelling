import numpy as np
import plotly.graph_objects as go
import streamlit as st


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


def equation_0(x):
    """f(x) = sin(x)"""
    return np.sin(x)


def equation_1(x):
    """f(x) = 3 + 2x - 3x² + 2x³"""
    return 3 + 2 * x - 3 * x ** 2 + 2 * x ** 3


def equation_2(x):
    """f(x) = 1 / (3cos(x) + 2)"""
    return 1 / (3 * np.cos(x) + 2)


def equation_3(x, y):
    """f(x, y) = 4 + 2y + 2x + xy + x³y"""
    return 4 + 2 * y + 2 * x + x * y + x ** 3 * y


def equation_4(x, y):
    """f(x, y) = y²/x²"""
    return y ** 2 / x ** 2


def generate_random(a, b, num_samples):
    return np.random.uniform(a, b, num_samples)


def calculate_average(list_, num_samples, func):
    sum_ = 0
    for i in range(0, num_samples):
        sum_ += func(list_[i])

    return sum_ / num_samples


def calculate(a, b, num_samples, func):
    list_random_uniform_nums = generate_random(a, b, num_samples)
    average = calculate_average(list_random_uniform_nums, num_samples, func)
    integral = (b - a) * average

    return integral


def calculate_integral(a, b, num_samples, num_iter, func):
    avg_sum = 0
    areas = []
    for i in range(0, num_iter):
        integral = calculate(a, b, num_samples, func)
        avg_sum += integral
        areas.append(integral)
    avg_integral = avg_sum / num_iter
    st.markdown(f"""
        Интеграл функции {func.__doc__} на интервале от {a} до {b} 
        с использованием {num_samples} примеров и {num_iter} итераций равен: **{avg_integral}**
    """)

    return areas


def main():
    st.markdown("### НИРС")
    st.markdown("""**Тема:**
    Алгоритм и программная реализация метода Монте-Карло при вычислении однократных и двойных интегралов.""")
    st.markdown("""**Цель работы:** 
    Получение навыков программной реализации метода Монте-Карло при вычислении однократных и двойных интегралов.""")

    description = """
    Пусть интеграл (a, b, f (x)) представляет собой интеграл от a до b функции f (x) по x
    Пусть avg (f (x), a, b) представляет собой среднее значение функции на отрезке a, b
    Используя вторую фундаментальную теорему исчисления avg (f (x), a, b) = 1 (b-a) интеграл (a, b, f (x))
    преобразовывая функцию, указанную во второй фундаментальной теореме исчисления
    мы получаем (b-a) avg (f (x), a, b) = интеграл (a, b, f (x))
    
    Мы вычисляем среднее значение функции на интервале, используя несколько случайных выборок в данном интервале.
    Затем мы используем это вычисленное среднее значение, чтобы найти интеграл, используя приведенное выше уравнение.
    
    Функции:
    1. f(x) = 3 + 2x - 3x² + 2x³
    на отрезке [0, 1]

    2. f(x) = 1 / (3cos(x) + 2)

    3. f(x, y) = 4 + 2y + 2x + xy + x³y
    x -> от -1 до 1
    y -> от 0 до 3

    4. f(x, y) = y²/x²
    """

    show_schema = st.checkbox("Показать описание")
    if show_schema:
        st.code(description)

    st.markdown("---")
    selected_fun = st.radio("Выберите функцию", (
        f"0. {equation_0.__doc__}",
        f"1. {equation_1.__doc__}",
        f"2. {equation_2.__doc__}",
        f"3. {equation_3.__doc__}",
        f"4. {equation_4.__doc__}",
    ))

    function = ""
    if selected_fun[:1] == "1":
        function = equation_1
    elif selected_fun[:1] == "2":
        function = equation_2
    elif selected_fun[:1] == "3":
        function = equation_3
    elif selected_fun[:1] == "4":
        function = equation_4
    elif selected_fun[:1] == "0":
        function = equation_0

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c4, c5 = st.columns(2)
    c6, c7 = st.columns(2)
    a = c1.number_input("Введите нижний предел:", value=-1.)
    b = c2.number_input("Введите верхний предел:", value=1.)
    sub_intervals = c3.number_input("Введите шаг:", min_value=.00000000001, value=.01, format="%.8f")
    precision = c4.number_input("Введите точность:", value=.1, format="%.8f")
    epsilon = c5.number_input("Введите эпсилон (ε):", min_value=.00000000001, value=1.0 * (10 ** - 8), format="%.8f")
    num_samples = c6.number_input("Введите количество примеров:", min_value=1, max_value=10, value=5, step=1)
    num_iterations = c7.number_input("Введите количество итераций:", min_value=1, max_value=100000, value=1000, step=1)

    st.markdown("---")
    # calc_type = st.radio(
    #     "Выберите тип вычисления", (
    #         "1. Вычисление однократного интеграла",
    #         "2. Вычисление двойного интеграла",
    #     )
    # )
    # if calc_type[:1] == "1":
    #     st.write(calc_type[3:])
    #
    # elif calc_type[:1] == "2":
    #     st.write(calc_type[3:])

    if abs(a % np.pi) < epsilon:
        a = int(a / np.pi) * np.pi
    elif abs(a % np.e) < epsilon:
        a = int(a / np.e) * np.e
    if abs(b % np.pi) < epsilon:
        b = int(b / np.pi) * np.pi
    elif abs(b % np.e) < epsilon:
        b = int(b / np.e) * np.e

    all_areas = calculate_integral(a, b, num_samples, num_iterations, function)

    show_areas = st.checkbox("Показать подробные результаты вычислений")
    if show_areas:
        st.code(all_areas)

    fig = px.histogram(
        all_areas,
        opacity=0.85,
        marginal="box",
        title="Распределение рассчитанных интегралов(площадей)",
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
