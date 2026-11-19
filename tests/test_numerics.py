"""Testes da integração e das métricas de saída dos reatores."""

from dataclasses import replace
import unittest

import numpy as np

from reator_flex import ReactorParams, reactor_rhs, simulate, summarize


class NumericalReportingTests(unittest.TestCase):
    def test_interior_peaks_on_two_point_mesh(self):
        p = ReactorParams()
        coarse = simulate(p, n_points=2)
        fine = simulate(p, n_points=1200)
        a = summarize(coarse, p, display=False)["metrics"]
        b = summarize(fine, p, display=False)["metrics"]
        for index, variable in [(3, "I"), (5, "temperature")]:
            with self.subTest(variable=variable):
                value = a["max_" + variable]
                time = a["time_max_" + variable]
                self.assertGreater(value, coarse.y[index].max())
                self.assertAlmostEqual(value, b["max_" + variable], places=8)
                self.assertAlmostEqual(time, b["time_max_" + variable], places=6)
                self.assertAlmostEqual(reactor_rhs(time, coarse.sol(time), p)[index], 0., places=9)
        self.assertIs(a["steady_state"], True)
        self.assertIs(b["steady_state"], True)
        self.assertAlmostEqual(a["product_outlet_mol_s"], p.F * coarse.y[4, -1])

    def test_endpoint_peaks_on_inert_batch(self):
        # Sem reagentes, T se aproxima monotonicamente da temperatura da camisa.
        for jacket, expected_time in [(300., 0.), (350., 80.), (330., 0.)]:
            with self.subTest(jacket=jacket):
                p = ReactorParams(mode="BATCH", CA0=0., CB0=0., T0=330., T_coolant=jacket)
                sol = simulate(p, t_final=80., n_points=2)
                metrics = summarize(sol, p, display=False)["metrics"]
                self.assertEqual(metrics["time_max_temperature"], expected_time)
                self.assertEqual(metrics["time_max_I"], 0.)
                self.assertEqual(metrics["max_I"], 0.)

    def test_steady_state_uses_the_entire_last_residence_time(self):
        # Lavagem de P inerte, CP=exp(-t/tau). Após 1 tau, |dCP/dt|
        # vale 0,00368. No início da janela ainda valia 0,01 mol/(L s).
        p = ReactorParams(CA0=0., CB0=0., CA_in=0., CB_in=0., CP0=1.,
                          T_coolant=330.)
        for end, expected in [(50., None), (100., False), (200., True)]:
            with self.subTest(end=end):
                sol = simulate(p, t_final=end, n_points=2)
                metrics = summarize(sol, p, display=False, concentration_rate_tol=0.005)["metrics"]
                self.assertIs(metrics["steady_state"], expected)
                self.assertEqual(metrics["residence_time_s"], 100.)
                if expected is not None:
                    window_start = end - 100.
                    expected_max_rate = np.exp(-window_start / 100.) / 100.
                    self.assertAlmostEqual(metrics["max_concentration_rate_mol_L_s"],
                                           expected_max_rate, places=8)

    def test_closed_cstr_and_batch_metrics_match(self):
        # Alimentação distinta da carga para expor o denominador da conversão.
        closed = ReactorParams(F=0., CA_in=2.)
        batch = replace(closed, mode="BATCH", F=1.)
        metrics = []
        for p in (closed, batch):
            sol = simulate(p, t_final=120., n_points=3)
            result = summarize(sol, p, display=False)["metrics"]
            metrics.append(result)
            self.assertIsNone(result["steady_state"])
            self.assertIsNone(result["residence_time_s"])
            self.assertEqual(result["product_outlet_mol_s"], 0.)
            self.assertAlmostEqual(result["conversion_A"], 1. - sol.y[0, -1] / p.CA0)
        for key in ("conversion_A", "max_temperature", "max_I"):
            self.assertAlmostEqual(metrics[0][key], metrics[1][key], places=9)
