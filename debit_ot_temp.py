import math as mt
import matplotlib.pyplot as plt
import numpy as np

t1 = float(input(r'Введите температуру 1:'))
t2 = float(input(r'Введите температуру 2:'))
v1 = float(input(r'Введите скорость при температуре 1:'))
v2 = float(input(r'Введите скорость при температуре 2:'))
ro = float(input(r'Введите плотность нефти при 293 Кельвинах:'))
K = float(input(r'Введите эффективную проницаемость нефти:'))
h = float(input(r'Введите эффективную мощность пласта:'))
Pr = float(input(r'Введите среднепластовое давление:'))
Pwf = float(input(r'Введите забивное давление:'))
B0 = float(input(r'Введите объемный коэффициент нефти:'))
re = float(input(r'Введите радиус дринирования:'))
rw = float(input(r'Введите радиус скважины:'))
S = float(input(r'Введите скин:'))
b = mt.log(mt.log(v1 + 0.8) / mt.log(v2 + 0.8)) / mt.log(t1 / t2)
a = mt.log(mt.log(v1 + 0.8)) - b * mt.log(t1)
ksi = 1.825 - 13.14 ** (-4) * ro


# Расчёт взякости нефти
def calc_mu(a: float, b: float, ro_t: float, t: float) -> float:
    mu = (mt.exp(mt.exp(a - b * mt.log(t))) - 0.8) * ro_t
    return mu


t_array = np.linspace(293, 793, 5)
q0_array = []
# Расчёт дебита
for t in t_array:
    ro_t = ro - ksi * (t - 293)
    mu = calc_mu(a, b, ro_t, t)
    q0 = (K * h * (Pr - Pwf)) / (18.41 * mu * B0 * (mt.log(re / rw) - 0.75 + S))
    q0_array.append(q0)

plt.plot(t_array, q0_array)
plt.xlabel(r't')
plt.ylabel(r'q0')

plt.show()  # Построение графика зависимости дебита от темпиратуры
