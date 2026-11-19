"""Testes de balanço independentes do relatório e das rotinas de diagnóstico."""

import unittest
from dataclasses import replace
from math import exp

import numpy as np

from reator_flex.model import R, ReactorParams, kinetic_constants, reactor_rhs
from reator_flex.simulation import simulate


class TestBalancos(unittest.TestCase):
    def setUp(self):
        # C já está presente e A e B têm estoques diferentes.
        self.mistura = ReactorParams(
            CA0=0.7, CB0=0.15, CC0=0.4, CI0=0.06, CP0=0.05, T0=335.0,
            CA_in=0.17, CB_in=0.9, CC_in=0.12, CI_in=0.02, CP_in=0.07,
            T_in=312.0, V=80.0, F=1.25,
        )

    def test_inventarios_batelada_e_equivalencia_cstr_sem_vazao(self):
        p = replace(self.mistura, mode="BATCH")
        sol = simulate(p, t_final=350, n_points=101)
        soma_produtos = sol.y[2:5].sum(axis=0)
        for i, reagente in enumerate((p.CA0, p.CB0)):
            esperado = reagente + p.CC0 + p.CI0 + p.CP0
            np.testing.assert_allclose(sol.y[i] + soma_produtos, esperado, atol=2e-9, rtol=0)
        fechado = simulate(replace(p, mode="CSTR", F=0), t_final=350, n_points=101)
        np.testing.assert_allclose(fechado.y, sol.y, atol=2e-9, rtol=1e-10)

    def test_inventarios_cstr_com_entrada_e_saida(self):
        p = self.mistura
        sol = simulate(p, t_final=350, n_points=101)
        for i, (c0, c_in) in enumerate(((p.CA0, p.CA_in), (p.CB0, p.CB_in))):
            estoque_inicial = c0 + p.CC0 + p.CI0 + p.CP0
            estoque_alimentado = c_in + p.CC_in + p.CI_in + p.CP_in
            esperado = estoque_alimentado + (estoque_inicial - estoque_alimentado) * np.exp(-p.F * sol.t / p.V)
            observado = sol.y[i] + sol.y[2:5].sum(axis=0)
            np.testing.assert_allclose(observado, esperado, atol=2e-8, rtol=2e-7)

    def test_entalpia_adiabatica_com_reacao(self):
        for modo in ("BATCH", "CSTR"):
            for ca0, cb0, cc0 in ((0.7, 0.9, 0.04), (0.0, 0.0, 0.8)):
                with self.subTest(modo=modo, ca0=ca0):
                    p = replace(self.mistura, mode=modo, UA=0.0, CA0=ca0, CB0=cb0, CC0=cc0)
                    sol = simulate(p, t_final=350, n_points=101)
                    # Entalpias molares relativas de A e B nulas na referência.
                    h = np.array([0, 0, p.dH1, p.dH1 + p.dH2, p.dH1 + p.dH2 + p.dH3])
                    observado = p.rho_cp * (sol.y[5] - p.T_ref) + h @ sol.y[:5]
                    h0 = p.rho_cp * (p.T0 - p.T_ref) + h @ [ca0, cb0, cc0, p.CI0, p.CP0]
                    h_in = p.rho_cp * (p.T_in - p.T_ref) + h @ [p.CA_in, p.CB_in, p.CC_in, p.CI_in, p.CP_in]
                    diluicao = p.F / p.V if modo == "CSTR" else 0.0
                    esperado = h_in + (h0 - h_in) * np.exp(-diluicao * sol.t)
                    np.testing.assert_allclose(observado, esperado, rtol=3e-7, atol=0.01)

    def test_camisa_e_entalpia_da_alimentacao_sem_reacao(self):
        for modo in ("BATCH", "CSTR"):
            with self.subTest(modo=modo):
                p = ReactorParams(
                    mode=modo, CA0=0, CB0=0, CA_in=0, CB_in=0,
                    CP0=0.2, CP_in=0.65, T0=345, T_in=325, T_coolant=293,
                    V=40, F=0.15, UA=680, rho_cp=2900,
                )
                sol = simulate(p, t_final=250, n_points=61)
                diluicao = p.F / p.V if modo == "CSTR" else 0.0
                troca = p.UA / (p.rho_cp * p.V)
                t_estacionaria = (diluicao * p.T_in + troca * p.T_coolant) / (diluicao + troca)
                esperado = t_estacionaria + (p.T0 - t_estacionaria) * np.exp(-(diluicao + troca) * sol.t)
                np.testing.assert_allclose(sol.y[5], esperado, atol=6e-5, rtol=0)

    def test_valores_dimensionais_do_balanco_local(self):
        p = ReactorParams(
            V=125, F=0.75, UA=300, rho_cp=4000,
            CA_in=1.5, CB_in=1.0, CC_in=0.1, CI_in=0.05, CP_in=0.15,
            T_in=340, T_coolant=315,
        )
        # A 330 K, r1 = 0.002, r2 = 0.012, r3 = 0.003 mol/(L s).
        y = [2, 0.2, 1, 0.75, 0.35, 330]
        estequiometria = np.array([-0.002, -0.002, -0.010, 0.009, 0.003])
        correntes = (0.75 / 125) * np.array([-0.5, 0.8, -0.9, -0.70, -0.20])
        # Os termos térmicos são reação (+0.130), camisa (-0.009) e entrada
        # (+0.060), todos em K/s.
        esperado = np.r_[estequiometria + correntes, 0.181]
        np.testing.assert_allclose(reactor_rhs(0, y, p), esperado, rtol=1e-13, atol=1e-14)

    def test_reacao_reversa_isolada_resfria_batelada(self):
        p = ReactorParams(mode="BATCH", UA=0, k2_ref=0, k3_ref=0)
        y = [0, 0, 1, 0, 0, 330]
        # Só C reage; a dissociação de 0,006 mol/(L s) absorve 300 J/(L s).
        esperado = [0.006, 0.006, -0.006, 0, 0, -300 / p.rho_cp]
        np.testing.assert_allclose(reactor_rhs(0, y, p), esperado, rtol=1e-13, atol=1e-14)

    def test_equilibrio_da_etapa_reversivel_em_tres_temperaturas(self):
        p = ReactorParams(mode="BATCH", UA=0, k2_ref=0, k3_ref=0)
        for T in (305.0, p.T_ref, 350.0):
            with self.subTest(T=T):
                k = kinetic_constants(T, p)
                kc = (p.k1f_ref / p.k1r_ref) * exp((p.dH1 / R) * (1 / p.T_ref - 1 / T))
                self.assertAlmostEqual(k["k1f"] / k["k1r"], kc, places=11)
                equilibrado = [0.35, 0.42, kc * 0.35 * 0.42, 0, 0, T]
                np.testing.assert_allclose(reactor_rhs(0, equilibrado, p), 0, atol=3e-15)

    def test_rejeita_parametros_inconsistentes(self):
        for mudanca in ({"E1r": 40000}, {"V": 0}, {"CI_in": -0.3}, {"T0": 0},
                        {"rho_cp": float("nan")}, {"F": float("inf")}):
            with self.subTest(mudanca=mudanca), self.assertRaises(ValueError):
                replace(self.mistura, **mudanca)
        replace(self.mistura, E1r=self.mistura.E1f - self.mistura.dH1)


if __name__ == "__main__":
    unittest.main()
