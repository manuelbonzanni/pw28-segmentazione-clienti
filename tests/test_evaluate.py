"""Test delle funzioni di valutazione (evaluate.py): simulazione di campagna, scenario trial, stabilità, allineamento"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config as C          # noqa: E402
import evaluate as E        # noqa: E402
import segmentation as S    # noqa: E402


def small_profile():
    """Profilo sintetico di tre segmenti con numeri facili da ricalcolare a mano"""
    return pd.DataFrame({
        "segmento": ["Clienti fedeli ad alto valore", "Clienti occasionali", "Inattivi / a rischio abbandono"],
        "clienti": [100, 200, 100],
    })


class TestCampaignSimulation(unittest.TestCase):
    def test_numbers_match_hand_calculation(self):
        r = E.campaign_simulation(small_profile())
        # A: 400 contatti * 4% * 25 EUR = 400 di incremento; costo 400 * 0,40 = 160
        self.assertAlmostEqual(r["A_incremento"], 400.0)
        self.assertAlmostEqual(r["A_costo"], 160.0)
        self.assertEqual(r["A_contatti"], 400)
        # B: fedeli esclusi; occasionali 200 * 8% * 24 = 384; inattivi 100 * 6% * 40 = 240; costo 300 * 0,40 = 120
        self.assertAlmostEqual(r["B_incremento"], 624.0)
        self.assertAlmostEqual(r["B_costo"], 120.0)
        self.assertEqual(r["B_contatti"], 300)
        self.assertAlmostEqual(r["B_netto"], 504.0)
        self.assertAlmostEqual(r["rapporto_netto_B_su_A"], 504 / 240)

    def test_breakeven_rate_makes_scenarios_equal(self):
        r = E.campaign_simulation(small_profile())
        r2 = E.campaign_simulation(small_profile(), mass_rate=r["tasso_A_di_pareggio"])
        self.assertAlmostEqual(r2["A_netto"], r2["B_netto"], places=6)

    def test_rate_factor_scales_only_b(self):
        full, half = E.campaign_simulation(small_profile()), E.campaign_simulation(small_profile(), rate_factor=0.5)
        self.assertAlmostEqual(half["B_incremento"], full["B_incremento"] / 2)
        self.assertAlmostEqual(half["A_incremento"], full["A_incremento"])

    def test_sensitivity_table_shape(self):
        t = E.campaign_sensitivity(small_profile(), rate_factors=(0.5, 1.0), costs=(0.2, 0.4))
        self.assertEqual(len(t), 4)
        self.assertTrue((t["B_netto"].diff().dropna() != 0).any())


class TestTrialScenario(unittest.TestCase):
    def test_values_follow_assumptions(self):
        out = pd.DataFrame({"piano": ["Trial"] * 10 + ["Plus"] * 5 + ["Pay-per-use"] * 5,
                            "importo_totale_90g": [50.0] * 10 + [300.0] * 5 + [100.0] * 5})
        r = E.trial_scenario(out, base_conv=0.20, uplift_pp=0.05)
        self.assertEqual(r["n_trial"], 10)
        self.assertAlmostEqual(r["conversioni_attese_oggi"], 2.0)
        self.assertAlmostEqual(r["conversioni_aggiuntive"], 0.5)
        self.assertAlmostEqual(r["valore_trimestrale"], 0.5 * 250.0)


class TestStabilityAndAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, _ = S.clean_data(S.load_data())
        cls.X = S.Preprocessor().fit(cls.df).transform(cls.df)

    def test_adopted_configuration_is_seed_stable(self):
        mean, mn = E.stability(self.X, 4, n_runs=5)
        self.assertAlmostEqual(mean, 1.0)
        self.assertAlmostEqual(mn, 1.0)

    def test_single_init_is_not_more_stable_than_adopted(self):
        mean_adopted, _ = E.stability(self.X, 4, n_runs=5)
        mean_single, _ = E.stability(self.X, 4, n_runs=5, n_init=1, init="random")
        self.assertLessEqual(mean_single, mean_adopted + 1e-9)

    def test_latent_profiles_follow_customer_id_not_row_order(self):
        shuffled = self.df.sample(frac=1, random_state=3).reset_index(drop=True)
        a = E.latent_profiles(self.df)
        b = E.latent_profiles(shuffled)
        # stesso cliente -> stesso profilo, qualunque sia l'ordine delle righe
        m = dict(zip(self.df["customer_id"], a))
        self.assertTrue(all(m[c] == p for c, p in zip(shuffled["customer_id"], b)))
        self.assertEqual(len(a), len(self.df))


if __name__ == "__main__":
    unittest.main()
