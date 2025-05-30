import math
import numpy as np
from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from scipy.stats import t
from scipy.stats import f

a,b = None, None

def lin_regression(X, ni, mi, yx, xy):
    sum_x_sqr_n = sum([X[i] ** 2 * ni[i] for i in range(len(X))])
    sum_x_n = sum([X[i] * ni[i] for i in range(len(X))])
    sum_n = sum(ni)
    sum_x_n_yx = sum([X[i] * ni[i] * yx[i] for i in range(len(X))])
    sum_n_yx = sum([ni[i] * yx[i] for i in range(len(X))])

    a, b = np.linalg.solve(np.array([[sum_x_sqr_n, sum_x_n], [sum_x_n, sum_n]]), np.array([sum_x_n_yx, sum_n_yx]))
    y_star = [a * X[i] + b for i in range(len(X))]

    n = sum(ni)
    m = sum(mi)
    y_avg = sum([yx[i] * ni[i] for i in range(len(yx))]) / n

    Q = sum([(yx[i] - y_avg) ** 2 * ni[i] for i in range(len(ni))])
    Qp = sum([(y_star[i] - y_avg) ** 2 * ni[i] for i in range(len(ni))])
    Qo = sum([(yx[i] - y_star[i]) ** 2 * ni[i] for i in range(len(ni))])


    R_sqr = 1 - Qo / Q

    m = 2
    F_emp = (Qp * (n - m)) / (Qo * (m - 1))

    def check(win, alpha, F_emp):
        F_crit = f.ppf(1 - alpha, m - 1, n - m)
        Label(win, text=f'Fемп = {F_emp}\tFкр = {F_crit}', font=font).grid(row=6, column=1)
        if F_emp < F_crit:
            Label(win, text=f'Оскільки Fемп < Fкр, то гіпотезу H0 приймаємо', font=font).grid(row=7, column=1)
        else:
            Label(win, text=f'Оскільки Fемп > Fкр, то гіпотезу H0 відхиляємо', font=font).grid(row=7, column=1)

    lin_regr_win = Toplevel()
    font = ('Arial', 14, 'bold')

    Label(lin_regr_win, text='Лінійна регресія', font=('Arial', 20, 'bold')).grid(row=0)
    Label(lin_regr_win, text=f'Рівняння регресії: y = {round(a, 2)}x + {round(b, 2)}', font=font).grid(row=1, column=1)
    Label(lin_regr_win, text=f'Q = {Q}\tQp = {Qp}\tQo = {Qo}', font=font).grid(row=2, column=1)
    Label(lin_regr_win, text=f'R^2 = {R_sqr}', font=font).grid(row=3, column=1)
    Label(lin_regr_win, text=f'H0: модель регресії не є значущою\tH1: модель регресії є значущою', font=font).grid(row=4, column=1)
    alpha_frame = Frame(lin_regr_win)
    Label(alpha_frame, text='Введіть рівень значущості:', font=font).grid(column=0, row=0)

    alpha_var = StringVar()
    Entry(alpha_frame, textvariable=alpha_var, font=font).grid(column=1, row=0)
    Button(alpha_frame, text='Перевірити гіпотезу', command=lambda: check(lin_regr_win, float(alpha_var.get()), F_emp), font=font).grid(column=3, row=0)
    alpha_frame.grid(row=5, column=1)

    fig = Figure(figsize=(5, 5), dpi=100)
    plot1 = fig.add_subplot(111)
    plot1.plot(X, yx, label='y(x) (спостереження)')
    plot1.plot(X, y_star, label='y*(x) (прогноз)')
    plot1.set_title('Лінійна регресія')
    plot1.set_xlabel('X')
    plot1.set_ylabel('Y')
    plot1.legend()

    graph = Frame(lin_regr_win)
    graph.grid(row=1, column=0, rowspan=10)

    canvas1 = FigureCanvasTkAgg(fig, master=graph)
    canvas1.get_tk_widget().pack()

    toolbar = NavigationToolbar2Tk(canvas1, graph)
    toolbar.update()
    toolbar.pack()

    canvas1.get_tk_widget().pack()

def nonlin_regretion(X, ni, yx):
    global a, b
    n = sum(ni)

    # sum_one_x_n = sum([1/X[i] * ni[i] for i in range(len(X))])
    # sum_one_x_sqr_n = sum([1/X[i]**2 * ni[i] for i in range(len(X))])
    # sum_one_x_yx_n = sum([1/X[i] * yx[i] * ni[i] for i in range(len(X))])
    # sum_n_yx = sum([ni[i] * yx[i] for i in range(len(X))])
    # sum_n_x_4 = sum([ni[i] * X[i] ** 4 for i in range(len(X))])
    # sum_n_x_3 = sum([ni[i] * X[i] ** 3 for i in range(len(X))])
    # sum_n_yx_x_2 = sum([ni[i] * yx[i] * X[i] ** 2 for i in range(len(yx))])
    # sum_n_yx_x = sum([ni[i] * yx[i] * X[i] for i in range(len(yx))])
    # sum_n_x_2 = sum([ni[i] * X[i] ** 2 for i in range(len(X))])
    # sum_n_lg_yx = sum([ni[i] * math.log10(yx[i]) for i in range(len(X))])
    # sum_n_x_lg_yx = sum([ni[i] * math.log10(yx[i]) * X[i] for i in range(len(X))])

    sum_yx_n = sum([yx[i] * ni[i] for i in range(len(yx))])
    sum_n_x = sum([ni[i] * X[i] for i in range(len(X))])
    sum_n_root_x = sum([ni[i]*math.sqrt(X[i]) for i in range(len(X))])
    sum_n_yx_root_x = sum([ni[i] * yx[i] * math.sqrt(X[i]) for i in range(len(X))])

    # a, b = np.linalg.solve(np.array([[sum_one_x_n, n],[sum_one_x_sqr_n, sum_one_x_n]]), np.array([sum_yx_n, sum_one_x_yx_n]))
    # def fun(x):
    #     return a / x + b
    #
    # a, b, c = np.linalg.solve(np.array([[sum_n_x_4, sum_n_x_3, sum_n_x_2], [sum_n_x_3, sum_n_x_3, sum_n_x], [sum_n_x_2, sum_n_x, n]]), np.array([sum_n_yx_x_2, sum_n_yx_x, sum_n_yx]))
    # def fun(x):
    #     return a * x ** 2 + b * x + c


    # A = np.array([[n, sum_n_x], [sum_n_x, sum_n_x_2]])
    # B = np.array([sum_n_lg_yx, sum_n_x_lg_yx])
    # c, d = np.linalg.solve(A, B)
    # a = 10 ** d
    # b = 10 ** c
    # # def fun(x, a=a, b=b):
    #     return b * a ** x

    a, b = np.linalg.solve([[sum_n_root_x, n],[sum_n_x, sum_n_root_x]], [sum_yx_n, sum_n_yx_root_x])
    def fun(x, a=a, b=b):
        return a*math.sqrt(x)+b

    n_total = sum(sum(row) for row in table)
    y_total_sum = 0
    for i in range(len(Y)):
        for j in range(len(X)):
            y_total_sum += Y[i] * table[i][j]
    y_avg = y_total_sum / n_total
    y_star = [fun(X[i]) for i in range(len(X))]

    Qp = sum([(y_star[i] - y_avg) ** 2 * ni[i] for i in range(len(ni))])
    Qo = sum([(yx[i] - y_star[i]) ** 2 * ni[i] for i in range(len(ni))])
    Q = sum([(yx[i] - y_avg) ** 2 * ni[i] for i in range(len(ni))])
    # Q = Qp + Qo
    R_sqr = 1 - Qo / Q

    m = 2
    F_emp = (Qp * (n - m)) / (Qo * (m - 1))


    def check(win, alpha, F_emp):
        F_crit = f.ppf(1 - alpha, m-1, n-m)
        Label(win, text=f'Fемп = {F_emp}\tFкр = {F_crit}', font=font).grid(row=6, column=1)
        if F_emp < F_crit:
            Label(win, text=f'Оскільки Fемп < Fкр, то гіпотезу H0 приймаємо', font=font).grid(row=7,
                                                                                                          column=1)
        else:
            Label(win, text=f'Оскільки Fемп > Fкр, то гіпотезу H0 відхиляємо', font=font).grid(row=7,
                                                                                                           column=1)

    nonlin_regr_win = Toplevel()
    font = ('Arial', 14, 'bold')
    Label(nonlin_regr_win, text='Нелінійна регресія (коренева кореляція)', font=('Arial', 20, 'bold')).grid(row=0)
    Label(nonlin_regr_win, text=f'Рівняння регресії: y = {round(a, 2)} * √x + ({round(b, 2)})', font=font).grid(row=1, column=1)
    Label(nonlin_regr_win, text=f'Q = {Q}\tQp = {Qp}\tQo = {Qo}', font=font).grid(row=2, column=1)
    if Qp <= Q and Qo <= Q:
        Label(nonlin_regr_win, text=f'R^2 = {R_sqr}', font=font).grid(row=3, column=1)
        Label(nonlin_regr_win, text=f'H0: модель регресії не є значущою\tH1: модель регресії є значущою', font=font).grid(
            row=4, column=1)
        alpha_frame = Frame(nonlin_regr_win)
        Label(alpha_frame, text='Введіть рівень значущості:', font=font).grid(column=0, row=0)
        alpha_var = StringVar()
        Entry(alpha_frame, textvariable=alpha_var, font=font).grid(column=1, row=0)
        Button(alpha_frame, text='Перевірити гіпотезу',
               command=lambda: check(nonlin_regr_win, float(alpha_var.get()), F_emp), font=font).grid(column=3, row=0)
        alpha_frame.grid(row=5, column=1)


    fig = Figure(figsize=(5, 5), dpi=100)
    plot1 = fig.add_subplot(111)
    plot1.plot(X, yx, label='y(x) (спостереження)')
    plot1.plot(X, y_star, label='y*(x) (коренева модель)')
    plot1.set_title('Нелінійна (коренева) регресія')
    plot1.set_xlabel('X')
    plot1.set_ylabel('Y')
    plot1.legend()

    graph = Frame(nonlin_regr_win)
    graph.grid(row=1, column=0, rowspan=10)

    canvas1 = FigureCanvasTkAgg(fig, master=graph)
    canvas1.get_tk_widget().pack()

    toolbar = NavigationToolbar2Tk(canvas1, graph)
    toolbar.update()
    toolbar.pack()

    canvas1.get_tk_widget().pack()

def lin_coef(X, Y, ni, mi):
    n = sum(ni)
    m = sum(mi)
    y_avg = sum([yx[i] * ni[i] for i in range(len(yx))]) / n
    x_avg = sum([xy[i] * mi[i] for i in range(len(xy))]) / m

    s = 0
    for i in range(len(Y)):
        for j in range(len(X)):
            s += table[i][j] * (X[j] - x_avg) * (Y[i] - y_avg)

    c12 = 1 / (n - 1) * s
    s1 = math.sqrt(1 / (n - 1) * sum([ni[i] * (X[i] - x_avg) ** 2 for i in range(len(X))]))
    s2 = math.sqrt(1 / (n - 1) * sum([mi[i] * (Y[i] - y_avg) ** 2 for i in range(len(Y))]))

    r12 = c12 / (s1 * s2)

    t_emp = (r12 * math.sqrt(n - 2)) / math.sqrt(1 - r12 ** 2)

    def check(win, alpha, t_emp):
        t_crit = t.ppf(1 - alpha / 2, n - 2)
        Label(win, text=f'tемп = {t_emp}\ntкр = {t_crit}', font=font).grid(row=5)
        if abs(t_emp) < t_crit:
            Label(win, text=f'Оскільки |tемп| < tкр, то гіпотезу H0 приймаємо', font=font).grid(row=6)
        else:
            Label(win, text=f'Оскільки |tемп| > tкр, то гіпотезу H0 відхиляємо', font=font).grid(row=6)

    nonlin_regr_win = Toplevel()
    font = ('Arial', 14, 'bold')
    Label(nonlin_regr_win, text='Вибірковий лінійний коефіцієнт кореляці', font=('Arial', 20, 'bold')).grid(row=0)
    Label(nonlin_regr_win, text=f'c12 = {c12}\ts1 = {s1}\ts2 = {s2}', font=font).grid(row=1)
    Label(nonlin_regr_win, text=f'r12 = {r12}', font=font).grid(row=2)
    Label(nonlin_regr_win, text=f'H0: ρ=0\tH1: ρ≠0', font=font).grid(row=3)
    alpha_frame = Frame(nonlin_regr_win)
    Label(alpha_frame, text='Введіть рівень значущості:', font=font).grid(column=0, row=0)
    alpha_var = StringVar()
    Entry(alpha_frame, textvariable=alpha_var, font=font).grid(column=1, row=0)
    Button(alpha_frame, text='Перевірити гіпотезу', command=lambda: check(nonlin_regr_win, float(alpha_var.get()), t_emp),
           font=font).grid(column=3, row=0)
    alpha_frame.grid(row=4)


def predict(w, x_star):
        global a, b
        x_star = float(x_star)
        y = a * math.sqrt(x_star) + b
        Label(w, text=f'y* = {y}', font=('Arial', 14, 'bold')).grid(row=3, column=0, columnspan=3)
def prediction():
    predict_win = Toplevel()
    font = ('Arial', 14, 'bold')
    Label(predict_win, text='Прогнозоване значення y*', font=('Arial', 20, 'bold')).grid(row=0,columnspan=3)
    Label(predict_win, text=f'Рівняння регресії: y* = {round(a, 2)} * x* + ({round(b, 2)})', font=font).grid(row=1, column=0,columnspan=3)
    Label(predict_win, text=f'Введіть x*:', font=font).grid(row=2, column=0)
    x_star = StringVar()
    Entry(predict_win, textvariable=x_star).grid(row=2, column=1)
    Button(predict_win, text='Прогнзувати', command=lambda: predict(predict_win, x_star.get())).grid(row=2, column=3)

X = [2,3,5,8,10,11,13]
Y = [3,4,6,8,10,12]
table = [[0, 0, 0, 0, 0, 19, 2],
         [0, 0, 0,3, 31, 2, 0],
         [0, 0, 1, 16, 3, 0, 0],
         [0, 2, 21, 4, 0, 0, 0],
         [3, 31, 5, 0, 0, 0, 0],
         [30, 2, 0, 0, 0, 0, 0]]

ni = [sum([table[i][j] for i in range(len(table))]) for j in range(len(table[0]))]
mi = [sum(table[i]) for i in range(len(table))]


yx = []
for i in range(len(table[0])):
    s = 0
    for j in range(len(table)):
        s += Y[j] * table[j][i] / ni[i]
    yx.append((s))

xy = []
for i in range(len(table)):
    s = 0
    for j in range(len(table[0])):
        s += X[j] * table[i][j] / mi[i]
    xy.append((s))

main = Tk()
main.state("zoomed")
font = ('Arial', 15, 'bold')
Label(main, text='Кореляційна таблиця:', font=font).pack(pady=(50,0))
table_image = PhotoImage(file='img.png')
table_and_graph = Frame(main)
table_and_graph.pack()
Label(table_and_graph, image=table_image).grid(row=0, column=0)
Label(main, text=f'Умовні середні yxi: {yx}', font=font).pack()
buttons = Frame(main)
Button(buttons, text='Лінійна регресія', font=font, command=lambda: lin_regression(X, ni, mi, yx, xy)).grid(row=0,column=0)
Button(buttons, text='Нелінійна регресія', font=font, command=lambda: nonlin_regretion(X, ni, yx)).grid(row=0,column=1)
Button(buttons, text='Вибірковий лінійний коефіцієнт кореляції', font=font, command=lambda: lin_coef(X, Y, ni, mi)).grid(row=1,column=0)
Button(buttons, text='Прогноз', font=font, command=lambda: prediction()).grid(row=1,column=1)
buttons.pack(pady=(20,0))
fig = Figure(figsize=(4, 4), dpi=100)
plot1 = fig.add_subplot(111)
plot1.plot(X, yx, marker='o', linestyle='-')
plot1.set_title('Емпірична лінія регресії')
plot1.set_xlabel('X')
plot1.set_ylabel('y|x')

graph = Frame(table_and_graph)
graph.grid(row=0, column=1)

canvas1 = FigureCanvasTkAgg(fig, master=graph)
canvas1.get_tk_widget().pack()

toolbar = NavigationToolbar2Tk(canvas1, graph)
toolbar.update()
toolbar.pack()

canvas1.get_tk_widget().pack()
main.mainloop()
