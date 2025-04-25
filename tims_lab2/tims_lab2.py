from tkinter import *
import math
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)
from scipy.stats import chi2


def F(x, lam):
      return 1 - math.e ** (-lam * x)

def check_hip(alpha, lambda_, x_avr, intervals, ni, table):
    n = sum(ni)
    param_n = 1
    if alpha.get() == '':
        alpha = 0.05
    else:
        alpha = float(alpha.get())
    if lambda_.get() == '':
        lambda_ = 1/x_avr
    else:
        lambda_ = float(lambda_.get())
        param_n = 0
    pi = []
    for inter in intervals[:-1]:
        pi.append(F(inter[1], lambda_) - F(inter[0], lambda_))

    pi.append(1-sum(pi))
    npi = [n*i for i in pi]

    table.loc['pi'] = [i for i in pi]
    table.loc['npi'] = [i for i in npi]
    rows = len(table)
    columns = len(table.iloc[0])

    hip_window = Tk()

    int_roz_table = Frame(hip_window)
    for i in range(columns): Label(int_roz_table,
                                   text=str(table.iloc[0, i])[:-1] + f'{"]" if i == columns - 1 else ")"}',
                                   font=font).grid(row=1, column=i + 1)
    for i in range(1, rows):
        for j in range(columns):
            e = Label(int_roz_table, text=round(table.iloc[i, j], 4), font=font)
            e.grid(row=i + 1, column=j + 1)
    Label(int_roz_table, text='[Zi-1; Zi)', font=font).grid(row=1, column=0)
    Label(int_roz_table, text='ni', font=font).grid(row=2, column=0)
    Label(int_roz_table, text='pi', font=font).grid(row=3, column=0)
    Label(int_roz_table, text='npi', font=font).grid(row=4, column=0)
    int_roz_table.grid(row=0, column=0, pady=(50, 0), padx=(30,0))#place(x=50, y=50)


    ind = 0
    while ind < columns-1:
        if table.iloc[1][ind] < 5 or table.iloc[3][ind] < 5:

            inter_list = table.iloc[0].tolist()
            ni_list = table.iloc[1].tolist()
            pi_list = table.iloc[2].tolist()
            npi_list = table.iloc[3].tolist()

            inter_list[ind] = [inter_list[ind][0], inter_list[ind+1][1]]
            ni_list[ind] += ni_list[ind+1]
            pi_list[ind] += pi_list[ind+1]
            npi_list[ind] += npi_list[ind+1]

            inter_list.pop(ind+1)
            ni_list.pop(ind+1)
            pi_list.pop(ind+1)
            npi_list.pop(ind+1)

            table=pd.DataFrame({"[Zi-1; Zi)" : inter_list, "ni": ni_list, "pi": pi_list, "npi": npi_list}).T
            columns = len(table.iloc[0])
        else:
            ind += 1
    if table.iloc[1][len(table.iloc[0])-1] < 5 or table.iloc[3][len(table.iloc[0])-1] < 5:
        inter_list = table.iloc[0].tolist()
        ni_list = table.iloc[1].tolist()
        pi_list = table.iloc[2].tolist()
        npi_list = table.iloc[3].tolist()

        inter_list[-2] = [inter_list[-2][0], inter_list[-1][1]]
        ni_list[-2] += ni_list[-1]
        pi_list[-2] += pi_list[-1]
        npi_list[-2] += npi_list[-1]

        inter_list.pop(-1)
        ni_list.pop(-1)
        pi_list.pop(-1)
        npi_list.pop(-1)

        table = pd.DataFrame({"[Zi-1; Zi)": inter_list, "ni": ni_list, "pi": pi_list, "npi": npi_list}).T
        columns = len(table.iloc[0])

    hi_emp = 0
    for i in range(len(table.iloc[0])):
        hi_emp += (table.iloc[1][i] - table.iloc[3][i])**2 / table.iloc[3][i]
    hi_crit = chi2.ppf(1-alpha, len(table.iloc[0])-1-param_n)
    hip_window.geometry("1400x600")
    int_roz_table_after = Frame(hip_window)
    for i in range(columns): Label(int_roz_table_after,
                                   text=str(table.iloc[0, i])[:-1] + f'{"]" if i == columns - 1 else ")"}',
                                   font=font).grid(row=1, column=i + 1)
    for i in range(1, rows):
        for j in range(columns):
            e = Label(int_roz_table_after, text=round(table.iloc[i, j], 4), font=font)
            e.grid(row=i + 1, column=j + 1)
    Label(int_roz_table_after, text='[Zi-1; Zi)', font=font).grid(row=1, column=0)
    Label(int_roz_table_after, text='ni', font=font).grid(row=2, column=0)
    Label(int_roz_table_after, text='pi', font=font).grid(row=3, column=0)
    Label(int_roz_table_after, text='npi', font=font).grid(row=4, column=0)
    int_roz_table_after.grid(row=1, column=0, pady=(50, 0))#place(x=0, y=250)

    results = Frame(hip_window)
    Label(results, text=f"X²емп = {round(hi_emp, 4)}", font=font).grid(row=0, column=0, padx=(0, 50))
    Label(results, text=f"X²кр |α = {alpha}; df = {columns-1-param_n}| = {round(hi_crit, 4)}", font=font).grid(row=0, column=1)
    if hi_emp < hi_crit:
        Label(results, text="Гіпотезу H0 приймаємо", font=font).grid(row=1, column=0, columnspan=2, pady=(50, 0))
    else:
        Label(results, text="Гіпотезу H0 відхиляємо", font=font).grid(row=1, column=0, columnspan=2, pady=(50, 0))
    results.grid(row=2, column=0, pady=(70, 0))

    hip_window.mainloop()


file = open("sample6.txt", "r")
vib = list(map(float, file.read().split()))
vib.sort()
n = len(vib)

int_roz_n = int(1 + 3.322 * math.log10(n))
h = vib[-1]  / int_roz_n



intervals = []
t = 0
for _ in range(int_roz_n):
    intervals.append([round(t, 4), round(t + h, 4)])
    t += h




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


x_avr = 0
for i, interv in enumerate(intervals):
    x_avr += (interv[0]+interv[1])/2 * ni[i]
x_avr = x_avr / n

print(n)


main = Tk()
main.state("zoomed")
var_row_frame = Frame(main)

Label(var_row_frame, text='Варіаційний ряд:', font=('Arial', 15, 'bold')).pack(side=LEFT)
row_scroller = Scrollbar(var_row_frame, orient=HORIZONTAL)
row = Text(var_row_frame, wrap=NONE, xscrollcommand=row_scroller.set, height=1, font=('Arial', 15, 'bold'))
row.pack(side=TOP, fill=BOTH, expand=True)
row_scroller.config(command=row.xview)
row_scroller.pack(side=BOTTOM, fill=X)
row.insert(END, str(vib)[1:-1])
var_row_frame.place(x=230, y=50)

kh_frame = Frame(main)
kh = Label(kh_frame, text=f'Згрупування даних: k = {int_roz_n}; h = {round(h, 4) }', font=('Arial', 20, 'bold'))
kh.pack()
kh_frame.place(x=500, y=130)


table = pd.DataFrame({'[Zi-1; Zi)': intervals, 'ni': ni}).T
rows = len(table)
columns = len(table.iloc[0])

int_roz_table = Frame(main)


font = ('Arial', 12, 'bold')
Label(int_roz_table, text='Інтервальний розподіл', font=('Arial', 20, 'bold')).grid(row=0, column=0,
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
int_roz_table.place(x=150, y=200)



fig = Figure(figsize=(8, 4), dpi=100)
plot1 = fig.add_subplot(111)

plot1.set_title('Гістограма')
plot1.bar([i[0] for i in intervals], ni, width=h, align='edge', edgecolor='black')
plot1.set_xlim(intervals[0][0] - 0.5, intervals[-1][1] + 0.5)
plot1.axhline(y=0, color='black', linewidth=2)
plot1.axvline(x=0, color='black', linewidth=2)
plot1.set_xticks([i[0] for i in intervals] + [intervals[-1][1]])
plot1.set_xlabel('x')
plot1.set_ylabel('ni')

histogram_frame = Frame(main, width=5000)

canvas1 = FigureCanvasTkAgg(fig, master=histogram_frame)
canvas1.draw()
canvas1.get_tk_widget().pack()
plot1.set_yticks([i * 50 for i in range(11)])
toolbar = NavigationToolbar2Tk(canvas1, histogram_frame)
toolbar.update()
toolbar.pack()

canvas1.get_tk_widget().pack()
histogram_frame.place(x=50, y = 300)



results = Frame(main)

alpha = StringVar()
lambda_ = StringVar()

Label(results, text="З вигляду гістограми висуваємо гіпотезу", font=("Arial", 16, "bold")).grid(row=0, columnspan = 2)
Label(results, text="H0: генеральна сукупність має експонентний закон розподілу", font=("Arial", 14, "bold")).grid(row=1, columnspan = 2)
Label(results, text="Функція розподілу:", font=("Arial", 14)).grid(row=2, columnspan = 2, pady=(50, 0))
Label(results, text="F(x) = ", font=("Arial", 14)).grid(row=3, rowspan=2, padx=(200, 0))
Label(results, text="0,                x < 0", font=("Arial", 14)).grid(row=3, column=1, padx=(0, 200))
Label(results, text="1 - e^(-λx),   x >= 0", font=("Arial", 14)).grid(row=4, column=1, padx=(0, 200))
alpha_frame = Frame(results)
Label(alpha_frame, text="Рівень значущості α = ", font=("Arial", 12)).grid(row=0, column=0)
Entry(alpha_frame, textvariable=alpha).grid(row=0, column=1)
Label(alpha_frame, text="Значення за замовчуванням", font=("Arial", 9)).grid(row=1, column=0)
Label(alpha_frame, text="0.05", font=("Arial", 9)).grid(row=1, column=1)
alpha_frame.grid(row=5, column=0, pady=(50, 0))
lambda_frame = Frame(results)
Label(lambda_frame, text="Параметр λ = ", font=("Arial", 12)).grid(row=0, column=0)
Entry(lambda_frame, textvariable=lambda_).grid(row=0, column=1)
Label(lambda_frame, text="1/x̄ = ", font=("Arial", 9)).grid(row=1, column=0)
Label(lambda_frame, text=round(1/x_avr , 4), font=("Arial", 9)).grid(row=1, column=1)
lambda_frame.grid(row=5, column=1, pady=(50, 0))

Button(results, command=lambda: check_hip(alpha, lambda_, x_avr, intervals, ni, table), text="Перевірити гіпотезу H0", height=2, width=20).grid(row=6, column=0, columnspan=2, pady=(40, 0))

results.place(x=900, y=350)

main.mainloop()
