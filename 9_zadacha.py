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

gamma = (m * mu * compr / k) * (dx ** 2 / dt)
p_implicit = np.zeros([mesh_h, mesh_w])
p_implicit[0][:] = p_initial
#print(p_implicit)

for t_ind in range(1, mesh_h):
    alpha = np.zeros(mesh_w)
    beta = np.zeros(mesh_w)
    for x_ind in range(0, mesh_w):
        if x_ind == 0:
            a = 0
            b = -1 - gamma
            c = 1
            d = -q_left * dx * mu / (a_square * k) - gamma * p_implicit[t_ind - 1][x_ind]
        elif x_ind == mesh_w - 1:
            a = 1
            b = -3 - 3 / 4 * gamma
            c = 0
            d = -3 / 4 * gamma * p_implicit[t_ind - 1][x_ind] - 2 * p_right
        else:
            a = 1
            b = -2 - gamma
            c = 1
            d = -gamma * p_implicit[t_ind - 1][x_ind]
        if x_ind == 0:
            alpha[x_ind] = - c / b
            beta[x_ind] = d / b
        else:
            alpha[x_ind] = - c / (a * alpha[x_ind - 1] + b)
            beta[x_ind] = (d - a * beta[x_ind - 1]) / (a * alpha[x_ind - 1] + b)
    for x_ind in range(mesh_w - 1, -1, -1):
        if x_ind == mesh_w - 1:
            p_implicit[t_ind][x_ind] = beta[x_ind]
        else:
            p_implicit[t_ind][x_ind] = alpha[x_ind] * p_implicit[t_ind][x_ind + 1] + beta[x_ind]

p_implicit /= 1e6
print(p_implicit)

fig = plt.figure(figsize=(15,10))
for p_distribution in p_implicit[::7]:
    plt.plot(range(1, 6), p_distribution)

plt.title('Распределение давления вдоль пласта \n в различные моменты времени')
plt.xlabel('Номер ячейки')
plt.ylabel('Давление (МПа)')

plt.axhline(y=20.0, color='black', linestyle='--', linewidth=0.5)
plt.grid(linewidth=0.5)
plt.show()


