import numpy as np
import matplotlib.pyplot as plt

def rad_pres_calc (r:float, p_w = 99, q = 118, mu = 3, k = 1000, h = 9, r_w = 0.14, s=0) -> float:
    rad_pres = p_w + 18.41*q*mu/(k*h)*(np.log(r/r_w)+s)
    return rad_pres

print(f'Давление на расстоянии x1={18} м, составляет примерно {round(rad_pres_calc(18), 1)} атм')
print(f'Давление на расстоянии x2={49} м, составляет примерно {round(rad_pres_calc(49), 1)} атм')

x = np.arange(0.1, 100, 1)

plt.figure(figsize=(5, 5))
plt.title('Распределение давления вдоль радиуса')
plt.xlabel('Расстояние от скважины (м)')
plt.ylabel('Давление (атм)')
plt.grid(which='major')
plt.minorticks_on()
plt.grid(which='minor', linestyle='--', linewidth=0.5)
plt.plot(x, rad_pres_calc(x), c='black')
plt.show()

def deb_calc(k=1000, h=9, p_e=102.5, p_w=99, r_e=40, r_w=0.14, mu=3, s=0):
    deb = (k * h * (p_e - p_w)) / (18.41 * mu * (np.log(r_e / r_w) + s))
    return deb

s_arr = np.linspace(-5, 7, 100)
k_arr = np.linspace(100,12000,1000)
p_e_arr = np.linspace(41, 141, 100)
mu_arr = np.linspace(1, 100, 100)
r_e_arr = np.linspace(10, 1000, 1000)
p_w_arr = np.linspace(1, 41, 100)

plt.figure(figsize=(5,5))
plt.plot(s_arr, [deb_calc(s=s) for s in s_arr], c = 'red')
plt.title('Чувствительность к скин-фактору')
plt.xlabel('Скин-фактор')
plt.ylabel('Дебит жидкости (м^3/сут)')
plt.grid()
plt.minorticks_on()
plt.grid(which='minor', linestyle='--', linewidth=0.4)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(k_arr, [deb_calc(k=k) for k in k_arr], c = 'orange')
plt.title('Чувствительность к проницаемости')
plt.xlabel('Проницаемость (мД)')
plt.ylabel('Дебит жидкости (м^3/сут)')
plt.grid()
plt.minorticks_on()
plt.grid(which='minor', linestyle='--', linewidth=0.5)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(p_e_arr, [deb_calc(p_e=p_e) for p_e in p_e_arr], c = 'green')
plt.title('Чувствительность к пластовому давлению')
plt.xlabel('Пластовое давление (атм)')
plt.ylabel('Дебит жидкости (м^3/сут)')
plt.grid()
plt.minorticks_on()
plt.grid(which='minor', linestyle='--', linewidth=0.5)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(mu_arr, [deb_calc(mu=mu) for mu in mu_arr], c = 'pink')
plt.title('Чувствительность к вязкости жидкости')
plt.xlabel('Вязкость жидкости, сПз')
plt.ylabel('Дебит жидкости, м^3/сут')
plt.grid()
plt.minorticks_on()
plt.grid(which='minor', linestyle='--', linewidth=0.5)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(r_e_arr, [deb_calc(r_e=r_e) for r_e in r_e_arr], c = 'gray')
plt.title('Чувствительность к радиусу пласта \n (зоны дренирования)')
plt.xlabel('Радиус пласта (зоны дренирования) (м)')
plt.ylabel('Дебит жидкости (м^3/сут)')
plt.grid()
plt.minorticks_on()
plt.grid(which='minor', linestyle='--', linewidth=0.5)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(p_w_arr, [deb_calc(p_w=p_w) for p_w in p_w_arr], c = 'blue')
plt.title('Чувствительность к забойному давлению')
plt.xlabel('Забойное давление (атм)')
plt.ylabel('Дебит жидкости (м^3/сут)')
plt.grid()
plt.minorticks_on()
plt.grid(which='minor', linestyle='--', linewidth=0.5)
plt.show()
