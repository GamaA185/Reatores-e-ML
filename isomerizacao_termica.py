import matplotlib.pyplot as plt 
import numpy as np
import overreact as rx
from overreact import _constants as constants
from overreact import core, simulate, api

scheme = rx.parse_reactions("Ciclopropano -> Propeno")

kf = 1.5 * 1e-4 # 1/s (T = 763 K)

dydt = simulate.get_dydt(scheme, np.array([kf]))

y, r = simulate.get_y(dydt, y0=[1, 0], t_span=(0, 5e4), method="Radau", atol=1e-10, rtol=1e-8)
t = np.linspace(y.t_min, 5e4)

fig, ax = plt.subplots()
ax.plot(t, y(t)[0], label="Ciclopropano")
ax.plot(t, y(t)[1], label="Propeno")
ax.legend()
ax.set_title("Thermal isomerization between cyclopropane and propene")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Concentration (M)")
plt.show()
