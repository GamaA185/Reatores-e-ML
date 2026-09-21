import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

# Inversão enzimática da sacarose em reator piloto
# DOI: 10.1002/bit.260260409

'''
Adaptado de Monsan e Combes (1984), que estudaram a hidrólise contínua
de solução concentrada de sacarose usando invertase imobilizada.
A sacarose forma glicose + frutose; B representa 
os açúcares invertidos totais (glicose + frutose).
'''

# Parâmetros cinéticos da invertase
k1 = 5000.0  # L/(mol·s), formação do complexo sacarose-invertase
k2 = 1.0      # 1/s, dissociação do complexo
k3 = 20.0     # 1/s, conversão do complexo em açúcares invertidos + enzima

# --- Parâmetros do CSTR piloto do artigo ---
V = 17.6       # volume útil do reator piloto (L)
F = 0.0038     # vazão volumétrica (L/s), aproximadamente 13,7 L/h (F = 0.0038 L/s * 3600 s/h = 13.68 L/h)
tau = V / F  # tempo de residência (s), aproximadamente 4.6 h (tau = 17.6 L / 0.0038 L/s = 4621 s ≈ 1.28 h)

# Concentrações na alimentação (mol/L)
sacarose_in = 2.50       # solução concentrada, próxima de 69% m/m
complexo_in = 0.0      
acucares_invertidos_in = 0.0

# Concentração efetiva equivalente para representar a invertase imobilizada.
invertase_in = 1.95e-5   # mol/L equivalente, calibrada para cerca de 72% de conversão
fracao_massica_sacarose = 0.69
densidade_alimentacao = 1.34  

# Condições iniciais no reator (reator inicialmente carregado com a alimentação)
sacarose_0 = sacarose_in
complexo_0 = 0.0
acucares_invertidos_0 = 0.0
invertase_total_0 = invertase_in  

y0 = [sacarose_0, complexo_0, acucares_invertidos_0, invertase_total_0]

def dydt(t, y):
    sacarose, complexo, acucares_invertidos, invertase_total = y
    invertase_livre = invertase_total - complexo
    v_formacao = k1 * sacarose * invertase_livre
    v_dissociacao = k2 * complexo
    v_reacao = k3 * complexo
    d_sacarose_dt = (F/V) * (sacarose_in - sacarose) - v_formacao + v_dissociacao
    d_complexo_dt = (F/V) * (complexo_in - complexo) + v_formacao - v_dissociacao - v_reacao
    d_acucares_invertidos_dt = (F/V) * (acucares_invertidos_in - acucares_invertidos) + v_reacao
    d_invertase_total_dt = (F/V) * (invertase_in - invertase_total)  # perda por arraste
    return [d_sacarose_dt, d_complexo_dt, d_acucares_invertidos_dt, d_invertase_total_dt]

# Simulação: seis tempos de residência para aproximar o regime estacionário
t_span = (0, 6 * tau)
t_eval = np.linspace(t_span[0], t_span[1], 1000)
sol = solve_ivp(dydt, t_span, y0, t_eval=t_eval, method='Radau',
                rtol=1e-8, atol=1e-10)

sacarose = sol.y[0]
complexo = sol.y[1]
acucares_invertidos = sol.y[2]
invertase_total = sol.y[3]
invertase_livre = invertase_total - complexo

conversao = (sacarose_in - sacarose[-1]) / sacarose_in
sacarose_hidrolisada_kg_h = (
    F * 3_600 * densidade_alimentacao * fracao_massica_sacarose * conversao
)
print(f'Conversao de sacarose: {100 * conversao:.1f}%')
print(f'Sacarose hidrolisada: {sacarose_hidrolisada_kg_h:.2f} kg/h')

# Gráfico
plt.figure(figsize=(9, 5))
plt.plot(sol.t, sacarose, label='Sacarose')
plt.plot(sol.t, complexo, label='Complexo sacarose-invertase')
plt.plot(sol.t, invertase_livre, label='Invertase livre')
plt.plot(sol.t, invertase_total, '--', label='Invertase total')
plt.plot(sol.t, acucares_invertidos, label='Açúcares invertidos')
plt.xlabel('Tempo (s)')
plt.ylabel('Concentração (M)')
plt.title(f'Inversão da sacarose')
plt.legend()
plt.grid(True)
plt.show()
