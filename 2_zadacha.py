import numpy as np
import matplotlib.pyplot as plt

def explor_radius(kt: float, fi=0.14, mu=8, c_t=3e-4) -> float:
    explor = 0.037 * np.sqrt(kt / (fi * mu * c_t))
    return explor

k = 94
t = 7593
print(f'За {t} мин давление распространится примерно на {round(explor_radius(k * t), 2)} метра')

kt = np.arange(0,500000,10)
plt.figure()
plt.title('Радиус исследования')
plt.xlabel('kt (мД*мин)')
plt.ylabel('Радиус исследования (м)')
plt.grid(which='major')
plt.minorticks_on()
plt.grid(which='minor', linestyle='--', linewidth=0.5)
plt.plot(kt, explor_radius(kt), c='purple')
plt.show()

def twg_explor_radius(r_inv: float, k=94, fi=0.14, mu=8, c_t=3e-4) -> float:
    twg_explor = fi * mu * c_t / k * (r_inv / 0.037) ** 2
    return twg_explor

r_inv = 100
print(f'При радиусе исследования r_inv={r_inv} м время исследования')
print(f'будет составлять примерно {round(twg_explor_radius(r_inv), 1)} мин')

