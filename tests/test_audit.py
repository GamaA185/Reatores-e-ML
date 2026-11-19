"""Fechamento integral independente da malha de saída e das taxas cinéticas."""

from dataclasses import replace
import unittest

from reator_flex import ReactorParams, audit_balances, simulate


class BalanceAuditTests(unittest.TestCase):
    def test_jacket_only(self):
        # Só P no tanque, sem alimentação: a camisa retira calor sensível.
        p = ReactorParams(mode="BATCH", CA0=0, CB0=0, CP0=1, UA=800,
                          T0=355, T_coolant=300)
        sol = simulate(p, t_final=400, n_points=2)
        residuals = audit_balances(sol, p)
        exchanged = p.rho_cp * p.V * abs(sol.y[5, -1] - p.T0)
        self.assertLess(residuals["max_enthalpy_residual_J"], 3e-6 * exchanged)
        self.assertLess(residuals["max_inventory_A_residual_mol"], 1e-9)
        self.assertLess(residuals["max_inventory_B_residual_mol"], 1e-9)
        # Uma camisa omitida na auditoria deixa à vista toda a energia retirada.
        missing_jacket = audit_balances(sol, replace(p, UA=0))
        self.assertAlmostEqual(missing_jacket["max_enthalpy_residual_J"], exchanged, places=5)

    def test_open_reacting_system(self):
        p = ReactorParams(CA0=0.4, CB0=1.2, CC0=0.1, CI_in=0.12, CP_in=0.08,
                          T_in=340, T_coolant=295)
        errors = audit_balances(simulate(p, t_final=1200, n_points=2), p)
        self.assertLess(errors["max_inventory_A_residual_mol"], 2e-5)
        self.assertLess(errors["max_inventory_B_residual_mol"], 2e-5)
        self.assertLess(errors["max_enthalpy_residual_J"], 3.0)
