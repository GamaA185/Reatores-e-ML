"""Exemplo de reação próxima a dados experimentais para o pacote reator_flex.

Este arquivo ilustra um caso inspirado em esterificação ácido-acético + etanol,
que é uma reação bem documentada na literatura e se encaixa bem na estrutura
atual do modelo do pacote:

    A + B <-> C -> I -> P

O objetivo aqui não é reproduzir a cinética química exata da esterificação,
mas sim fornecer um caso razoável e experimentalmente plausível, mantendo o
formato do reator_flex já implementado.

Uso:
    python -m reator_flex.esterification_example
"""

from reator_flex.model import ReactorParams
from reator_flex.reporting import summarize
from reator_flex.simulation import simulate


def make_esterification_like_params(mode: str = "CSTR") -> ReactorParams:
    """Cria um conjunto de parâmetros inspirado em esterificação.

    O caso é uma aproximação de uma reação bimolecular reversível com calor
    moderado, típica de sistemas ácido-alcool em CSTR ou batch.
    """
    return ReactorParams(
        mode=mode,
        # Operação do reator
        V=1.0,
        F=0.01 if mode.upper() == "CSTR" else 0.0,
        rho_cp=3600.0,
        UA=50.0,
        T_coolant=298.0,
        # Alimentação do CSTR / condições iniciais para batch
        CA_in=0.8,
        CB_in=2.0,
        CC_in=0.0,
        CI_in=0.0,
        CP_in=0.0,
        T_in=323.15,
        CA0=0.8,
        CB0=2.0,
        CC0=0.0,
        CI0=0.0,
        CP0=0.0,
        T0=323.15,
        # Temperatura de referência e cinética
        # Estes valores foram escolhidos para gerar uma conversão plausível,
        # na faixa esperada para uma esterificação em CSTR ao redor de 50-60°C.
        T_ref=323.15,
        k1f_ref=0.018,
        k1r_ref=0.0045,
        k2_ref=0.001,
        k3_ref=0.0002,
        # Energias de ativação (J/mol)
        E1f=52_000.0,
        E1r=48_000.0,
        E2=80_000.0,
        E3=85_000.0,
        # Entalpias (J/mol)
        dH1=-11_000.0,
        dH2=-5_000.0,
        dH3=-3_000.0,
    )


def run_example(mode: str = "CSTR", t_final: float = 6000.0, n_points: int = 2000):
    """Executa um caso próximo a dados experimentais e retorna resultados."""
    params = make_esterification_like_params(mode=mode)
    sol = simulate(params, t_final=t_final, n_points=n_points)
    report = summarize(sol, params)

    print("\n=== Exemplo inspirado em esterificação (aproximação experimental) ===")
    print(f"Modo: {mode.upper()}")
    print(f"Tempo final: {sol.t[-1]:.1f} s")
    print(f"Conversão aparente de A: {report['metrics']['conversion_A']:.4f}")
    print(f"Temperatura máxima: {report['metrics']['max_temperature']:.2f} K")
    print(f"Maior acúmulo de I: {report['metrics']['max_I']:.4e} mol/L")

    return params, sol, report


if __name__ == "__main__":
    run_example()
