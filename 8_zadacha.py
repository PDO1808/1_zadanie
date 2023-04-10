import numpy as np
import matplotlib.pyplot as plt

mesh_w = 5
mesh_h = 1000
dt = 10
dt *= 24 * 60 * 60
m = 0.15
mu = 5e-3
k = 5e-3 * (1e-6) ** 2
dx = 100
compr = 2.4e-9
a_square = 10000
p_initial = 18e6
q_left = 9/(24 * 60 * 60)
p_right = 20e6
dimless_mult = (dt / dx ** 2) * (k / (m * mu * compr))

p_explicit = np.zeros([mesh_h, mesh_w])

p_explicit[0][:] = p_initial

for t_ind in range(1, mesh_h):
    p_explicit[t_ind][0] = p_explicit[t_ind - 1][0] + \
                           dimless_mult * (p_explicit[t_ind - 1][1] - p_explicit[t_ind - 1][0] + q_left * mu / (
                a_square * k) * dx)
    p_explicit[t_ind][-1] = p_explicit[t_ind - 1][-1] + \
                            4 / 3 * dimless_mult * (
                                        2 * p_right - 3 * p_explicit[t_ind - 1][-1] + p_explicit[t_ind - 1][-2])

    for x_ind in range(1, mesh_w - 1):
        p_explicit[t_ind][x_ind] = p_explicit[t_ind - 1][x_ind] + \
                                   dimless_mult * (p_explicit[t_ind - 1][x_ind + 1] - 2 * p_explicit[t_ind - 1][x_ind] +
                                                   p_explicit[t_ind - 1][x_ind - 1])

p_explicit /= 1e6
fig = plt.figure(figsize=(15,10))
for p_distribution in p_explicit[::7]:
    plt.plot(range(1, 6), p_distribution)

plt.title('Распределение давления вдоль пласта \n в различные моменты времени')
plt.xlabel('Номер ячейки')
plt.ylabel('Давление (МПа)')

plt.axhline(y=20.0, color='black', linestyle='--', linewidth=0.5)
plt.grid(linewidth=0.5)
plt.show()

#Критерий стабильности
print(f'dt <= {round(1 / 2 * m * mu * compr / k, 3)} * dx^2')
print(dt <= 1 / 2 * (m * mu * compr / k) * dx ** 2)

