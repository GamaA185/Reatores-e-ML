# Reator líquido em batelada ou CSTR

Simulador transiente da rede **A + B ⇌ C → I → P**, com cinco balanços molares e um balanço de energia acoplados. As espécies e os parâmetros padrão são ilustrativos. Os livros ao fim desta página fundamentam as equações.

## Como executar

Na raiz do projeto, com Python 3.10 ou superior:

```sh
python -m pip install numpy scipy matplotlib pandas
python -m reator_flex.main
python -m unittest discover -s tests -v
```

A execução padrão gera o relatório do CSTR e os gráficos de concentração e temperatura. Para investigar uma batelada adiabática:

```python
from reator_flex import ReactorParams, audit_balances, simulate, summarize

p = ReactorParams(mode="BATCH", UA=0)
sol = simulate(p, t_final=800, n_points=201)
print(summarize(sol, p, display=False)["metrics"])
print(audit_balances(sol, p))
```

## Hipóteses e unidades

O líquido é ideal, incompressível e perfeitamente misturado. Volume, densidade, capacidade térmica volumétrica e entalpias de reação são constantes. A alimentação e o líquido no tanque têm a mesma capacidade térmica volumétrica. As propriedades da saída são as do tanque. A camisa tem temperatura prescrita. Desprezam-se calor de mistura, trabalho de eixo e variações de energia cinética e potencial.

| Grandeza | Símbolo no código | Unidade |
|---|---|---|
| Volume do líquido e vazão volumétrica | `V`, `F` | L, L/s |
| Concentrações e temperaturas absolutas | `CA…CP`, `T` | mol/L, K |
| Capacidade térmica por volume | `rho_cp` | J/(L K) |
| Condutância térmica global da camisa | `UA` | J/(s K) |
| Energias de ativação e entalpias de reação | `E*`, `dH*` | J/mol de reação como escrita |
| Constantes cinéticas | `k1f` e `k1r, k2, k3` | L/(mol s) e 1/s, respectivamente |

No CSTR, as vazões de entrada e saída são iguais a `F` e definem a taxa de diluição `D = F/V` em 1/s. Na batelada, `D = 0` independentemente do valor armazenado em `F`.

## Balanços de espécies

As velocidades por volume de líquido, em mol/(L s), são:

```text
r1 = k1f CA CB − k1r CC
r2 = k2 CC
r3 = k3 CI
```

Para cada espécie, o acúmulo molar no tanque é entrada menos saída mais geração líquida. Dividindo pelo volume constante:

```text
dCA/dt = D (CA_in − CA) − r1
dCB/dt = D (CB_in − CB) − r1
dCC/dt = D (CC_in − CC) + r1 − r2
dCI/dt = D (CI_in − CI) + r2 − r3
dCP/dt = D (CP_in − CP) + r3
```

Cada mol de C, I ou P contém um equivalente de A e um de B. Assim, para cada um dos inventários por volume, `S_A = CA + CC + CI + CP` e `S_B = CB + CC + CI + CP`, a combinação das equações acima dá `dS/dt = D (S_in − S)`. A solução exata é `S(t) = S_in + (S(0) − S_in) exp(−Dt)`. Em batelada ambos permanecem constantes. Somar simplesmente as cinco concentrações contaria a passagem de dois mols de reagentes para um mol de C como uma perda. Para converter os inventários em massa, são necessárias massas molares compatíveis com a rede, isto é, `M_C = M_I = M_P = M_A + M_B`.

## Balanço de energia

Aplicando a primeira lei ao conteúdo do tanque e usando as hipóteses acima:

```text
rho_cp V dT/dt = rho_cp V D (T_in − T)
                 + UA (T_coolant − T)
                 − V (dH1 r1 + dH2 r2 + dH3 r3)
```

Cada termo tem unidade J/s. A corrente aquece o líquido quando entra mais quente; uma camisa mais fria retira calor. Uma etapa exotérmica tem entalpia negativa e libera calor quando avança. Se a etapa 1 correr no sentido inverso, a taxa líquida `r1` troca de sinal e a contribuição térmica também. A condição adiabática é `UA = 0`; ainda pode haver variação de temperatura por reação ou alimentação.

### Equilíbrio da etapa reversível

As constantes seguem Arrhenius em relação a `T_ref`. No equilíbrio **da etapa 1**, `CC/(CA CB) = Kc = k1f/k1r` (L/mol). Para a solução ideal, com estado padrão de concentração fixo e entalpia constante, a dependência térmica segue:

```text
ln[Kc(T)/Kc(T_ref)] = (dH1/R) (1/T_ref − 1/T)
E1r = E1f − dH1
```

Mantêm-se os dois valores de referência da cinética. Com os dados padrão, a barreira reversa é 95 kJ/mol e o aquecimento desfavorece a formação de C no equilíbrio da etapa. Ao migrar um caso antigo, omita `E1r` para derivá-lo automaticamente. Se fornecido, ele precisa ser consistente com `E1f` e `dH1`.

## Como conferir o resultado

`audit_balances(sol, p)` usa os mesmos parâmetros da simulação. Nos passos internos aceitos pelo integrador, compara os dois inventários do tanque (mol) com a solução exata de diluição. Para verificar energia por uma segunda via, adota entalpias químicas relativas nulas para A e B e reconstrói a entalpia total do líquido por volume:

```text
h = rho_cp (T − T_ref) + dH1 (CC + CI + CP) + dH2 (CI + CP) + dH3 CP
H = V h
dH/dt = V D (h_in − h) + UA (T_coolant − T)
```

A auditoria compara `H(t) − H(0)` com a integral da potência das correntes e da camisa. A energia química já está em `h`; as velocidades de reação não entram nessa integral. A quadratura de Gauss usa três pontos por passo sobre a interpolação do BDF. O retorno contém o maior resíduo absoluto observado para A, B e entalpia total.

O integrador BDF é implícito, com `rtol=1e-8` e `atol=1e-10`. Os máximos de I e T consideram os extremos do intervalo e eventos de derivada passando de positiva a negativa nos passos internos. `n_points` controla os pontos salvos para os gráficos. Como em qualquer detecção por mudança de sinal, múltiplos extremos dentro de um único passo podem escapar.

No CSTR com vazão positiva, o relatório calcula a conversão aparente de A em relação à entrada e a vazão molar bruta de produto `F CP`. O diagnóstico de regime permanente compara as derivadas em 51 instantes do último tempo de residência com tolerâncias de 1e-8 mol/(L s) e 1e-6 K/s. As tolerâncias são configuráveis em `summarize`. Sem uma janela inteira, o diagnóstico fica indeterminado. Em operação fechada a conversão se refere à concentração inicial de A; com referência zero, fica indefinida.

| Caso padrão do CSTR | Resultado aproximado |
|---|---:|
| Tempo de residência | 100 s |
| Conversão aparente de A aos 2.000 s | 49,42% |
| Pico de I | 0,21574 mol/L aos 162,43 s |
| Pico de T | 336,655 K aos 227,55 s |
| Vazão molar de P aos 2.000 s | 0,11469 mol/s |

## Referências

Paginação impressa nas edições consultadas; o número entre parênteses indica a página do arquivo PDF, contando a capa.

1. Charles G. Hill Jr., *An Introduction to Chemical Engineering Kinetics & Reactor Design*. §2.5, p. 10 (PDF 19), dependência térmica do equilíbrio. §10.1, p. 350 (PDF 359), primeira lei com acúmulo em volume de controle. §10.2, pp. 352–353 (PDF 361–362), batelada incompressível. §10.3, p. 357 (PDF 366), CSTR com reações múltiplas.
2. Mark E. Davis e Robert J. Davis, *Fundamentals of Chemical Reaction Engineering*. §§3.2–3.3, pp. 65 e 71 (PDF 81 e 87), balanços molares. §9.3, p. 291, eq. 9.3.11 (PDF 307), balanço térmico com camisa na batelada. §9.5, p. 303, eq. 9.5.1 (PDF 319), limite estacionário do CSTR.
3. A. R. Secchi e E. C. Biscaia Jr., *Métodos Numéricos para Engenheiros Químicos – Algoritmos e Aplicações*, 2020, §§7.4–7.5, pp. 214–217 (mesma paginação no PDF), integração de sistemas de EDOs, BDF e rigidez.
