import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)
import numpy as np

from tkinter import *
import random
import math


def create_probab_for_nep_emp_f(intervals, wi):
    def outer(wi, h0, h1, om):
        def inner(x):
            return wi / (h1 - h0) * (x - h0) + om
        return inner

    nprob_list = [0]
    om = 0
    for i in range(len(intervals)):
        nprob = outer(wi[i], intervals[i][0], intervals[i][1], om)
        nprob_list.append(nprob)
        om = round(om + wi[i], 4)
    nprob_list.append(1)
    return nprob_list

def generate_nep(e, main_frame):
    n = int(e.get())
    if n < 50:
        return
    vib = sorted(np.random.uniform(6, 16, n))
    for i in range(n):
        vib[i] = round(vib[i].item(), 4)


    int_roz_n = round(1 + 3.322 * math.log10(n))
    h = round((vib[-1] - vib[0]) / int_roz_n, 4)



    intervals = []
    t = vib[0]
    for _ in range(int_roz_n):
        intervals.append([round(t, 4), round(t + h, 4)])
        t += h
    intervals[-1][1] = vib[-1]
    ni = []
    t = 0
    count = 0
    ind = 0
    while ind < n:
        if vib[ind] < intervals[t][1]:
            count += 1
        elif vib[ind] == intervals[-1][1] and t == len(intervals) - 1:
            count += 1
        else:
            ni.append(count)
            count = 0
            ind -= 1
            t += 1
        ind += 1
    ni.append(count)
    nih = [round(i / h, 4) for i in ni]
    wi = [round(i / n, 4) for i in ni]

    for widget in main_frame.winfo_children():
        for children in widget.winfo_children():
            children.destroy()

    var_row = Frame(main_frame)
    Label(var_row, text='Варіаційний ряд:', font=('Arial', 15, 'bold')).pack(side=LEFT)
    row_scroller = Scrollbar(var_row, orient=HORIZONTAL)
    row = Text(var_row, wrap=NONE, xscrollcommand=row_scroller.set, height=1, font=('Arial', 15, 'bold'))
    row.pack(side=TOP, fill=BOTH, expand=True)
    row_scroller.config(command=row.xview)
    row_scroller.pack(side=BOTTOM, fill=X)
    row.insert(END, str(vib)[1:-1])
    var_row.place(x=110, y=0)

    kh_frame = Frame(main_frame)
    kh = Label(kh_frame, text=f'Згрупування даних: k = {int_roz_n}; h = {h}', font=('Arial', 20, 'bold'))
    kh.pack()
    kh_frame.place(x=400, y=100)

    int_roz(main_frame, intervals, ni, nih, wi).place(x=0, y=180)
    histogram(main_frame, intervals, h, nih).place(x=720, y=400)
    emp_nep_graph(main_frame, intervals, wi).place(x=0, y=400)
    chis_char_inter(main_frame, intervals, ni, n).place(x=250, y=1000)


def generate_disc(e, main_frame,):
    def chis_char(main_frame, ):
        if n % 2 == 0:
            median = (vib[int(n / 2)] + vib[int(n / 2) - 1]) / 2
        else:
            median = vib[int(n / 2)]

        moda = []
        max_n = max(table.values())
        for i in table.keys():
            if table[i] == max_n:
                moda.append(i)

        ser_ar = 0
        for i in vib: ser_ar += i
        ser_ar /= n

        rozmah = vib[-1] - vib[0]

        dev = 0
        for i in vib: dev += (i - ser_ar) ** 2

        variansa = dev / (n - 1)

        standart = math.sqrt(variansa)

        vib_dis = dev / n

        ser_kvad_vid = math.sqrt(vib_dis)

        variatsia = standart / ser_ar

        moment2, moment3, moment4 = 0, 0, 0
        for i in vib:
            moment2 += (i - ser_ar) ** 2
            moment3 += (i - ser_ar) ** 3
            moment4 += (i - ser_ar) ** 4
        moment2 *= 1 / n
        moment3 *= 1 / n
        moment4 *= 1 / n

        asym = moment3 / moment2 ** (3 / 2)

        exscess = moment4 / moment2 ** 2 - 3

        kvart = []
        oct = []
        dets = []
        tsent = []
        milil = []

        if n % 4 == 0:
            ind = int(n / 4)
            for i in range(ind - 1, n - 1, ind):
                kvart.append(vib[i])
        if n % 8 == 0:
            ind = int(n / 8)
            for i in range(ind - 1, n - 1, ind):
                oct.append(vib[i])
        if n % 10 == 0:
            ind = int(n / 10)
            for i in range(ind - 1, n - 1, ind):
                dets.append(vib[i])
        if n % 100 == 0:
            ind = int(n / 100)
            tsent.append(vib[ind - 1])
            tsent.append(vib[n - 1 - ind])
        if n % 1000 == 0:
            ind = int(n / 1000)
            milil.append(vib[ind - 1])
            milil.append(vib[n - 1 - ind])

        char_frame = Frame(main_frame)
        char_frame.place(x=50, y=1400)


        Label(char_frame, text='Числові характеристики', font=('Arial', 25, 'bold')).grid(row=0, column=0, columnspan=9)
        for i, el in enumerate(
                [median, moda, ser_ar, rozmah, dev, variansa, standart, vib_dis, ser_kvad_vid, variatsia, asym,
                 exscess]):
            Label(char_frame,
                  text=f'{["Медіана: ", "Мода: ", "Сер. арифм.: ", "Розмах: ", "Девіація: ", "Варіанса: ", "Стандарт: ", "Виб. дисперсія: ", "Середньокв. відх.: ", "Варіація: ", "Асиметрія: ", "Ексцесс: "][i]}',
                  font=('Arial', 15, 'bold')).grid(row=i % 6 + 1, column=int(i / 6) * 2, padx=(20, 0))
            Label(char_frame, text=str(el)[1:-1] if isinstance(el, list) else str(round(el, 4)),
                  font=('Arial', 12, 'bold')).grid(row=i % 6 + 1, column=int(i / 6) * 2 + 1, padx=(0, 20))

        Label(char_frame, text=f'Квартилі: ', font=('Arial', 15, 'bold')).grid(row=1, column=4, padx=(40, 0),
                                                                               columnspan=2, sticky='w')
        Label(char_frame, text=f'Октилі: ', font=('Arial', 15, 'bold')).grid(row=2, column=4, padx=(40, 0),
                                                                             columnspan=2, sticky='w')
        Label(char_frame, text=f'Децилі: ', font=('Arial', 15, 'bold')).grid(row=4, column=4, padx=(40, 0),
                                                                             columnspan=2, sticky='w')
        Label(char_frame, text=f'Центилі: ', font=('Arial', 15, 'bold')).grid(row=6, column=4, padx=(40, 0))
        Label(char_frame, text=f'Мілілі: ', font=('Arial', 15, 'bold')).grid(row=6, column=6)

        if n % 4 == 0:
            Label(char_frame, text=f'Q1={kvart[0]}, Q2={kvart[1]}, Q3={kvart[2]}', font=('Arial', 12, 'bold')).grid(
                row=1, column=6, columnspan=2, sticky='e')
            Label(char_frame, text=f'Інтерквартильна: {kvart[-1] - kvart[0]}', font=('Arial', 12, 'bold')).grid(row=2,
                                                                                                                column=8,
                                                                                                                padx=(
                                                                                                                40, 0),
                                                                                                                sticky='w')
        else:
            Label(char_frame, text='-', font=('Arial', 12, 'bold')).grid(row=1, column=6, columnspan=2)
            Label(char_frame, text='Інтерквартильна: -', font=('Arial', 12, 'bold')).grid(row=2, column=8, padx=(40, 0),
                                                                                          sticky='w')

        if n % 8 == 0:
            Label(char_frame, text=f'O1={oct[0]}, O2={oct[1]}, O3={oct[2]}', font=('Arial', 12, 'bold')).grid(row=2,
                                                                                                              column=6,
                                                                                                              columnspan=2,
                                                                                                              sticky='e')
            Label(char_frame, text=f'O4={oct[3]}, O5={oct[4]}, O6={oct[5]}, O7={oct[6]}',
                  font=('Arial', 12, 'bold')).grid(row=3, column=5, columnspan=3, sticky='e')
            Label(char_frame, text=f'Інтероктильна: {oct[-1] - oct[0]}', font=('Arial', 12, 'bold')).grid(row=3,
                                                                                                          column=8,
                                                                                                          padx=(40, 0),
                                                                                                          sticky='w')
        else:
            Label(char_frame, text='-', font=('Arial', 12, 'bold')).grid(row=2, column=6, columnspan=2)
            Label(char_frame, text='Інтероктильна: -', font=('Arial', 12, 'bold')).grid(row=3, column=8, padx=(40, 0),
                                                                                        sticky='w')

        if n % 10 == 0:
            Label(char_frame, text=f'D1={dets[0]}, D2={dets[1]}, D3={dets[2]}, D4={dets[3]},',
                  font=('Arial', 12, 'bold')).grid(row=4, column=6, columnspan=2, sticky='e')
            Label(char_frame, text=f'D5={dets[4]}, D6={dets[5]}, D7={dets[6]}, D8={dets[7]}, D9={dets[8]}',
                  font=('Arial', 12, 'bold')).grid(row=5, column=5, columnspan=3, sticky='e')
            Label(char_frame, text=f'Інтердецильна: {dets[-1] - dets[0]}', font=('Arial', 12, 'bold')).grid(row=4,
                                                                                                            column=8,
                                                                                                            padx=(
                                                                                                            40, 0),
                                                                                                            sticky='w')
        else:
            Label(char_frame, text='-', font=('Arial', 12, 'bold')).grid(row=4, column=6, columnspan=2)
            Label(char_frame, text='Інтердецильна: -', font=('Arial', 12, 'bold')).grid(row=4, column=8, padx=(40, 0),
                                                                                        sticky='w')

        if n % 100 == 0:
            Label(char_frame, text=f'С01={tsent[0]}, C99={tsent[1]}', font=('Arial', 12, 'bold')).grid(row=6, column=5)
            Label(char_frame, text=f'Інтерцентильна: {tsent[-1] - tsent[0]}', font=('Arial', 12, 'bold')).grid(row=5,
                                                                                                               column=8,
                                                                                                               padx=(
                                                                                                               40, 0),
                                                                                                               sticky='w')
        else:
            Label(char_frame, text='-', font=('Arial', 12, 'bold')).grid(row=6, column=5)
            Label(char_frame, text='Інтерцентильна: -', font=('Arial', 12, 'bold')).grid(row=5, column=8, padx=(40, 0),
                                                                                         sticky='w')

        if n % 1000 == 0:
            Label(char_frame, text=f'M001={milil[0]}, M999={milil[1]}', font=('Arial', 12, 'bold')).grid(row=6,
                                                                                                         column=7)
            Label(char_frame, text=f'Інтермілільна: {milil[-1] - milil[0]}', font=('Arial', 12, 'bold')).grid(row=6,
                                                                                                              column=8,
                                                                                                              padx=(
                                                                                                              40, 0),
                                                                                                              sticky='w')
        else:
            Label(char_frame, text='-', font=('Arial', 12, 'bold')).grid(row=6, column=7)
            Label(char_frame, text='Інтермілільна: -', font=('Arial', 12, 'bold')).grid(row=6, column=8, padx=(40, 0),
                                                                                        sticky='w')

        Label(char_frame, text='Широти', font=('Arial', 15, 'bold')).grid(row=1, column=8, columnspan=2)

    def show_table(main_frame):

        ta = pd.DataFrame({'xi': xi, 'ni': table.values()}).T
        rows = len(ta)
        columns = len(ta.iloc[0])

        d_table = Frame(main_frame)
        d_table.place(x=160, y=70)

        for i in range(rows):
            for j in range(columns):
                e = Label(d_table, width=5, text=ta.iloc[i, j], font=('Arial', 20, 'bold'))
                e.grid(row=i + 1, column=j + 1)
        Label(d_table, text='Частотна таблиця', font=('Arial', 25, 'bold')).grid(row=0, column=0, columnspan=columns)
        Label(d_table, text='xi', font=('Arial', 20, 'bold')).grid(row=1, column=0)
        Label(d_table, text='ni', font=('Arial', 20, 'bold')).grid(row=2, column=0)
        d_table.update()

    def show_poligon(main_frame, ):

        fig = Figure(figsize=(5, 5), dpi=100)
        plot1 = fig.add_subplot(111)
        plot1.plot(xi, table.values(), color='black')
        plot1.set_xlim(xi[0] - 1, xi[-1] + 1)
        plot1.axhline(y=0, color='black', linewidth=2)
        plot1.axvline(x=0, color='black', linewidth=2)
        plot1.set_xticks(xi)
        if max(table.values()) > 20:
            plot1.set_yticks([min(table.values()) - min(5, min(table.values()))] + list(table.values()))
            plot1.set_ylim(min(table.values()) - min(5, min(table.values())), max(table.values()) + 2)
        else:
            plot1.set_yticks(np.arange(0, max(table.values()) + 1, 1))
        plot1.set_title('Полігон')
        plot1.set_xlabel('x')
        plot1.set_ylabel('n')

        poligon = Frame(main_frame)
        poligon.place(x=100, y=200)

        canvas1 = FigureCanvasTkAgg(fig,
                                    master=poligon)
        canvas1.get_tk_widget().pack()

        toolbar = NavigationToolbar2Tk(canvas1,
                                       poligon)
        toolbar.update()
        toolbar.pack()

        canvas1.get_tk_widget().pack()

    def show_diagram(main_frame, ):

        fig = Figure(figsize=(5, 5), dpi=100)
        plot1 = fig.add_subplot(111)
        plot1.stem(xi, table.values(), linefmt='black', markerfmt='black')
        plot1.set_xlim(xi[0] - 1, xi[-1] + 1)
        plot1.axhline(y=0, color='black', linewidth=2)
        plot1.axvline(x=0, color='black', linewidth=2)
        plot1.set_xticks(xi)
        if max(table.values()) > 20:
            plot1.set_yticks([min(table.values()) - min(5, min(table.values()))] + list(table.values()))
            plot1.set_ylim(min(table.values()) - min(5, min(table.values())), max(table.values()) + 2)
        else:
            plot1.set_yticks(np.arange(0, max(table.values()) + 1, 1))
        plot1.set_title('Діаграма')
        plot1.set_xlabel('x')
        plot1.set_ylabel('n')

        diagram = Frame(main_frame)
        diagram.place(x=700, y=200)

        canvas1 = FigureCanvasTkAgg(fig,
                                    master=diagram)
        canvas1.draw()
        canvas1.get_tk_widget().pack()

        toolbar = NavigationToolbar2Tk(canvas1,
                                       diagram)
        toolbar.update()
        toolbar.pack()
        canvas1.get_tk_widget().pack()

    def emp_func(main_frame, ):

        emp_f = Frame(main_frame)
        emp_f.place(x=200, y=825)

        Label(emp_f, text='Емпірична функція', font=('Arial', 23, 'bold')).grid(row=0, column=0, columnspan=3)
        func_font = ('Arial', 20, 'bold')
        Label(emp_f, text='F(x)=', font=func_font).grid(row=1, column=0, rowspan=len(dprob_list))
        Label(emp_f, text='0', font=func_font).grid(row=1, column=1)
        Label(emp_f, text=f', x<{xi[0]}', font=func_font).grid(row=1, column=2)
        for i in range(len(dprob_list) - 2):
            Label(emp_f, text=f"{dprob_list[i + 1]}", font=func_font).grid(row=i + 2, column=1)
            Label(emp_f, text=f", {xi[i]}<=x<{xi[i + 1]}", font=func_font).grid(row=i + 2, column=2)
        Label(emp_f, text='1', font=func_font).grid(row=len(dprob_list), column=1)
        Label(emp_f, text=f', x>={xi[-1]}', font=func_font).grid(row=len(dprob_list), column=2)
        emp_f.update()

    def emp_f_graph(main_frame, ):
        fig = Figure(figsize=(7, 5), dpi=100)
        plot1 = fig.add_subplot(111)

        plot1.set_title('Графік емпіричної функції')
        plot1.plot([-100, xi[0]], [0, 0], color='black', linewidth=3)
        plot1.plot([xi[-1], xi[-1] + 100], [1, 1], color='black')
        for i in range(len(xi) - 1):
            plot1.arrow(xi[i], dprob_list[i + 1], xi[i + 1] - xi[i], 0, length_includes_head=True,
                        head_width=0.02, head_length=0.2, color='black')

        plot1.set_xlim(xi[0] - 1, xi[-1] + 1)
        plot1.axhline(y=0, color='black', linewidth=2)
        plot1.axvline(x=0, color='black', linewidth=2)
        plot1.set_xticks(np.arange(xi[0], xi[-1] + 1, 1))
        plot1.set_yticks([round(i, 2) for i in dprob_list[1:]])
        plot1.set_xlabel('x')
        plot1.set_ylabel('F')

        emp_f_gr = Frame(main_frame)
        emp_f_gr.place(x=600, y=800)

        canvas1 = FigureCanvasTkAgg(fig,
                                    master=emp_f_gr)
        canvas1.draw()
        canvas1.get_tk_widget().pack()

        toolbar = NavigationToolbar2Tk(canvas1,
                                       emp_f_gr)
        toolbar.update()
        toolbar.pack()
        canvas1.get_tk_widget().pack()



    n = int(e.get())
    if n < 50:
        return
    vib = sorted([random.randint(6, 16) for _ in range(n)])
    table = {}
    for i in vib:
        table[i] = table.get(i, 0) + 1

    xi = list(table.keys())

    dprob = 0
    dprob_list = [0]
    for i in range(len(xi) - 1):
        dprob = round(dprob + table[xi[i]] / n, 4)
        dprob_list.append(dprob)
    dprob_list.append(1)

    int_roz_n = round(1 + 3.322 * math.log10(n))
    h = round((xi[-1] - xi[0]) / int_roz_n, 4)

    intervals = []
    t = xi[0]
    for _ in range(int_roz_n):
        intervals.append([round(t, 4), round(t + h, 4)])
        t += h
    intervals[-1][1] = round(intervals[-1][1])
    ni = []
    t = 0
    count = 0
    ind = 0
    while ind < n:
        if vib[ind] < intervals[t][1]:
            count += 1
        elif vib[ind] == intervals[-1][1] and t == len(intervals)-1:
            count += 1
        else:
            ni.append(count)
            count = 0
            ind -= 1
            t += 1
        ind += 1
    ni.append(count)
    nih = [round(i / h, 4) for i in ni]
    wi = [round(i / n, 4) for i in ni]



    for widget in main_frame.winfo_children():
        for children in widget.winfo_children():
            children.destroy()
    var_row = Frame(main_frame)
    Label(var_row, text='Варіаційний ряд:', font=('Arial', 15, 'bold')).pack(side=LEFT)
    row_scroller = Scrollbar(var_row, orient=HORIZONTAL)
    row = Text(var_row, wrap=NONE, xscrollcommand=row_scroller.set, height=1, font=('Arial', 15, 'bold'))
    row.pack(side=TOP, fill=BOTH, expand=True)
    row_scroller.config(command=row.xview)
    row_scroller.pack(side=BOTTOM, fill=X)
    row.insert(END, str(vib)[1:-1])
    var_row.place(x=110, y=0)

    kh_frame = Frame(main_frame)
    kh = Label(kh_frame, text=f'Згрупування даних: k = {int_roz_n}; h = {h}', font=('Arial', 20, 'bold'))
    kh.pack()
    kh_frame.place(x=400, y=1700)

    show_table(main_frame)
    show_poligon(main_frame,)
    show_diagram(main_frame,)
    emp_func(main_frame,)
    emp_f_graph(main_frame,)
    int_roz(main_frame, intervals, ni, nih, wi).place(x=80, y=1800)
    histogram(main_frame, intervals, h, nih).place(x=730, y=2000)
    chis_char(main_frame,)
    emp_nep_graph(main_frame, intervals, wi).place(x=0, y=2000)
    chis_char_inter(main_frame, intervals, ni, n).place(x=300, y=2600)



def int_roz(main_frame, intervals, ni, nih, wi):
    table = pd.DataFrame({'[Zi-1; Zi)': intervals, 'ni': ni, 'ni/h': nih, 'wi': wi}).T
    table = pd.DataFrame({'[Zi-1; Zi)': intervals, 'ni': ni, 'ni/h': nih, 'wi': wi}).T
    rows = len(table)
    columns = len(table.iloc[0])

    int_roz_table = Frame(main_frame)


    font = ('Arial', 14+7-columns, 'bold')
    Label(int_roz_table, text='Інтервальний розподіл варіанти', font=('Arial', 20, 'bold')).grid(row=0, column=0,
                                                                                                columnspan=columns + 1)
    for i in range(columns): Label(int_roz_table,
                                   text=str(table.iloc[0, i])[:-1] + f'{"]" if i == columns - 1 else ")"}',
                                   font=font).grid(row=1, column=i + 1)
    for i in range(1, rows):
        for j in range(columns):
            e = Label(int_roz_table, text=table.iloc[i, j], font=font)
            e.grid(row=i + 1, column=j + 1)
    Label(int_roz_table, text='[Zi-1; Zi)', font=font).grid(row=1, column=0)
    Label(int_roz_table, text='ni', font=font).grid(row=2, column=0)
    Label(int_roz_table, text='ni/h', font=font).grid(row=3, column=0)
    Label(int_roz_table, text='wi', font=font).grid(row=4, column=0)



    return int_roz_table


def emp_nep_graph(main_frame, intervals, wi):
    probabilities = create_probab_for_nep_emp_f(intervals, wi)
    fig = Figure(figsize=(7, 5), dpi=100)
    plot1 = fig.add_subplot(111)
    for i in range(len(intervals)):
        x = np.linspace(intervals[i][0], intervals[i][1], 50)
        plot1.plot(x, probabilities[i + 1](x), color='black')
        plot1.scatter(intervals[i][1], probabilities[i + 1](intervals[i][1]), color='black', facecolors='none')
        plot1.plot([intervals[i][1], intervals[i][1]], [0, probabilities[i + 1](intervals[i][1])], linestyle='--',
                   color='black', linewidth=0.5)
        plot1.plot([0, intervals[i][1]], [probabilities[i + 1](intervals[i][1]), probabilities[i + 1](intervals[i][1])],
                   linestyle='--', color='black', linewidth=0.5)
    plot1.plot([0, intervals[0][0]], [0, 0], color='black', linewidth=3)
    plot1.plot([intervals[-1][1], intervals[-1][1] + 2], [1, 1], color='black')
    plot1.set_xlim(intervals[0][0] - 1, intervals[-1][1] + 1)
    plot1.axhline(y=0, color='k')
    plot1.axvline(x=0, color='k')
    plot1.set_title('Графік емпіричної функції')
    plot1.set_xticks([i[0] for i in intervals] + [intervals[-1][1]])
    plot1.set_yticks([round(probabilities[i + 1](intervals[i][1]), 2) for i in range(len(intervals))])
    plot1.set_xlabel('z')
    plot1.set_ylabel('F')

    emp_f_nep_graph = Frame(main_frame)


    canvas1 = FigureCanvasTkAgg(fig, master=emp_f_nep_graph)
    canvas1.draw()
    canvas1.get_tk_widget().pack()
    toolbar = NavigationToolbar2Tk(canvas1, emp_f_nep_graph)
    toolbar.update()
    toolbar.pack()
    canvas1.get_tk_widget().pack()
    return emp_f_nep_graph

def histogram(main_frame, intervals, h, nih):

    fig = Figure(figsize=(6, 5), dpi=100)
    plot1 = fig.add_subplot(111)

    plot1.set_title('Гістограма')
    plot1.bar([i[0] for i in intervals], nih, width=h, align='edge', edgecolor='black')
    plot1.set_xlim(intervals[0][0] - 1, intervals[-1][1] + 1)
    plot1.axhline(y=0, color='black', linewidth=2)
    plot1.axvline(x=0, color='black', linewidth=2)
    plot1.set_xticks([i[0] for i in intervals] + [intervals[-1][1]])
    plot1.set_yticks(nih)
    plot1.set_xlabel('x')
    plot1.set_ylabel('ni/h')

    histogram_frame = Frame(main_frame)


    canvas1 = FigureCanvasTkAgg(fig, master=histogram_frame)
    canvas1.draw()
    canvas1.get_tk_widget().pack()
    if min(nih) > 20:
        plot1.set_ylim(min(nih) - 5, max(nih) + 2)
        plot1.set_yticks([min(nih) - 5] + nih)
    else:
        plot1.set_yticks(nih)
    toolbar = NavigationToolbar2Tk(canvas1, histogram_frame)
    toolbar.update()
    toolbar.pack()

    canvas1.get_tk_widget().pack()

    return  histogram_frame

def chis_char_inter(main_frame, intervals, ni, n):
    k = len(ni)
    half = n / 2
    nak = 0
    med_int = 0
    med_n = 0
    for i in range(k):
        if (nak + ni[i] >= half):
            med_int = intervals[i]
            med_n = ni[i]
            break
        nak += ni[i]
    median = med_int[0] + (med_int[1] - med_int[0]) * (half - nak) / med_n

    moda = []
    max_n = max(ni)
    for i in range(k):
        if ni[i] == max_n:
            n_next = 0
            n_prev = 0
            if i != 0:
                n_prev = ni[i - 1]
            if i != len(ni) - 1:
                n_next = ni[i + 1]
            moda.append(round(intervals[i][0] + (ni[i] - n_prev) * (intervals[i][1] - intervals[i][0]) / (
                        (ni[i] - n_prev) + (ni[i] - n_next)), 4))

    ser = 0
    for i in range(k):
        ser += (intervals[i][1] + intervals[i][0]) / 2 * ni[i]
    ser /= n

    rozmah = intervals[-1][1] - intervals[0][0]

    dev = 0
    for i in range(k):
        dev += ni[i] * ((intervals[i][0] + intervals[i][1]) / 2 - ser) ** 2

    varianca = round(dev / (n - 1), 4)

    standart = round(math.sqrt(varianca), 4)

    variacia = round(standart / ser, 4)

    dispersia = round(dev / n, 4)

    ser_kv_v = round(math.sqrt(dispersia), 4)

    m2 = dispersia
    m3, m4 = 0, 0
    for i in range(k):
        x = (intervals[i][0] + intervals[i][1]) / 2
        m3 += ni[i] * (x - ser) ** 3
        m4 += ni[i] * (x - ser) ** 4
    m3 /= n
    m4 /= n

    asimetria = m3 / m2 ** (3 / 2)
    excess = m4 / m2 ** 2 - 3

    int_char_frame = Frame(main_frame)


    Label(int_char_frame, text='Числові характеристики інтервального розподілу', font=('Arial', 25, 'bold')).grid(
        row=0, column=0, columnspan=4)
    for i, el in enumerate(
            [median, moda, ser, rozmah, dev, varianca, standart, dispersia, ser_kv_v, variacia, asimetria, excess]):
        Label(int_char_frame,
              text=f'{["Медіана: ", "Мода: ", "Сер. арифм.: ", "Розмах: ", "Девіація: ", "Варіанса: ", "Стандарт: ", "Виб. дисперсія: ", "Середньокв. відх.: ", "Варіація: ", "Асиметрія: ", "Ексцесс: "][i]}',
              font=('Arial', 15, 'bold')).grid(row=i % 6 + 1, column=int(i / 6) * 2, padx=(20, 0))
        Label(int_char_frame, text=str(el)[1:-1] if isinstance(el, list) else str(round(el, 4)),
              font=('Arial', 12, 'bold')).grid(row=i % 6 + 1, column=int(i / 6) * 2 + 1, padx=(0, 20))

    return int_char_frame


def disc():

    main.destroy()
    dwindow = Tk()
    dwindow.state('zoomed')

    n = StringVar()
    canvas = Canvas(dwindow)
    scroll_frame = Frame(canvas,  width=1520, height=4000)

    scrollbar = Scrollbar(dwindow, command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack( side = RIGHT, fill=Y )
    canvas.pack(side=LEFT, fill=BOTH, expand=True)

    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    scroll_frame.bind("<Configure>", on_frame_configure)

    d_size_frame = Frame(scroll_frame)
    main_frame = Frame(scroll_frame, height=3000, width=1300)


    Label(scroll_frame, text='Дискретна статистична змінна', font=('Arial', 25, 'bold')).place(x=750, y=50, anchor="center")
    Label(d_size_frame, text='Введіть об\'єм вибірки n:', font=('Arial', 20, 'bold')).grid(row=0)
    e = Entry(d_size_frame, textvariable=n, width=5, font=('Arial', 16, 'bold'))
    e.grid(row=0, column=1)
    d_size_frame.place(x=750, y=120, anchor="center")
    main_frame.place(x=760, y=1700, anchor="center")
    Button(d_size_frame, text='Згенерувати вибірку', command=lambda: generate_disc(e, main_frame), font=('Arial', 16, 'bold')).grid(row=2,column=0, columnspan=2)
    dwindow.mainloop()

def nep():
    main.destroy()
    nwindow = Tk()
    nwindow.state('zoomed')

    n = StringVar()
    k = StringVar()
    h = StringVar()

    canvas = Canvas(nwindow)
    scroll_frame = Frame(canvas, width=1520, height=4000)

    scrollbar = Scrollbar(nwindow, command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side=RIGHT, fill=Y)
    canvas.pack(side=LEFT, fill=BOTH, expand=True)

    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    scroll_frame.bind("<Configure>", on_frame_configure)

    d_size_frame = Frame(scroll_frame)
    main_frame = Frame(scroll_frame, height=3000, width=1300)

    Label(scroll_frame, text='Неперервна статистична змінна', font=('Arial', 25, 'bold')).place(x=750, y=50, anchor="center")
    Label(d_size_frame, text='Введіть об\'єм вибірки n:', font=('Arial', 20, 'bold')).grid(row=0)
    e = Entry(d_size_frame, textvariable=n, width=5, font=('Arial', 16, 'bold'))
    e.grid(row=0, column=1)
    d_size_frame.place(x=750, y=120, anchor="center")
    main_frame.place(x=760, y=1700, anchor="center")
    Button(d_size_frame, text='Згенерувати вибірку', command=lambda: generate_nep(e, main_frame),
           font=('Arial', 16, 'bold')).grid(row=2, column=0, columnspan=2)
    nwindow.mainloop()


main = Tk()
main.geometry('400x200')
Label(main, text='Оберіть тип змінної для вибірки', font=('Arial', 15, 'bold')).place(x=40, y=40)
Button(text='Дискретна', command=disc, font=('Arial', 12)).place(x=80, y=100)
Button(text='Неперервна', command=nep, font=('Arial', 12)).place(x=220, y=100)
main.mainloop()