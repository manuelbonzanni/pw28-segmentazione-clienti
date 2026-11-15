"""Test della pipeline: generazione dati, pulizia, pre-processing, segmentazione, assegnazione nuovi clienti
   e visualizzazione (generate_data.py, segmentation.py, plots.py)"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config as C          # noqa: E402
import generate_data as G   # noqa: E402
import segmentation as S    # noqa: E402


class TestGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = G.generate()

    def test_size_in_range(self):
        self.assertTrue(300 <= len(self.df) <= 600)

    def test_deterministic(self):
        pd.testing.assert_frame_equal(self.df, G.generate())

    def test_different_seed_differs(self):
        self.assertFalse(self.df["importo_totale_90g"].equals(G.generate(seed=7)["importo_totale_90g"]))

    def test_unique_ids_no_pii(self):
        self.assertTrue(self.df["customer_id"].is_unique)
        pii = {"nome", "cognome", "email", "telefono", "indirizzo"}
        self.assertFalse(pii & set(self.df.columns))

    def test_behavioural_coherence(self):
        d = self.df.dropna(subset=["giorni_ultima_richiesta"])
        self.assertTrue(((d["richieste_90g"] == 0) <= (d["giorni_ultima_richiesta"] > C.WINDOW_DAYS)).all())
        self.assertTrue(((d["richieste_90g"] > 0) <= (d["giorni_ultima_richiesta"] <= C.WINDOW_DAYS)).all())

    def test_has_all_behavioural_types(self):
        self.assertEqual(set(self.df["_profilo_latente"]), set(G.ARCHETYPES))

    def test_missing_values_injected(self):
        self.assertGreater(self.df[C.RFM_FEATURES].isna().sum().sum(), 0)


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = G.generate()
        raw["data_iscrizione"] = pd.to_datetime(raw["data_iscrizione"])
        cls.df, cls.report = S.clean_data(raw.drop(columns="_profilo_latente"))
        cls.pre = S.Preprocessor().fit(cls.df)
        cls.X = cls.pre.transform(cls.df)

    def test_cleaning_removes_nan(self):
        self.assertEqual(int(self.df[C.FEATURES].isna().sum().sum()), 0)
        self.assertEqual(self.report["righe_finali"], 480)

    def test_cleaning_drops_duplicates_and_negatives(self):
        dirty = pd.concat([self.df, self.df.iloc[:3]])
        dirty.loc[dirty.index[0], "importo_totale_90g"] = -5
        clean, rep = S.clean_data(dirty)
        self.assertEqual(rep["duplicati_rimossi"], 3)
        self.assertEqual(rep["negativi_importo_totale_90g"], 1)
        self.assertTrue((clean["importo_totale_90g"] >= 0).all())

    def test_negative_requests_are_imputed_and_pipeline_runs(self):
        """Un valore negativo di richieste diventa mancante e viene imputato: la pipeline non deve fallire"""
        dirty = self.df.copy()
        i_out = dirty.index[dirty["giorni_ultima_richiesta"] > C.WINDOW_DAYS][0]   # ultima richiesta fuori finestra
        i_in = dirty.index[dirty["giorni_ultima_richiesta"] <= C.WINDOW_DAYS][0]   # ultima richiesta in finestra
        dirty.loc[i_out, "richieste_90g"] = -1
        dirty.loc[i_in, "richieste_90g"] = -3
        clean, rep = S.clean_data(dirty)
        self.assertEqual(rep["negativi_richieste_90g"], 2)
        self.assertEqual(rep["mancanti_richieste"], 2)
        self.assertEqual(int(clean[C.FEATURES].isna().sum().sum()), 0)
        self.assertEqual(clean.loc[i_out, "richieste_90g"], 0)          # fuori finestra -> 0 richieste
        self.assertGreater(clean.loc[i_in, "richieste_90g"], 0)         # in finestra -> mediana degli attivi
        seg = S.run_segmentation(clean, k=4)                            # prima della correzione: ValueError (NaN)
        self.assertEqual(len(seg.labels), len(clean))

    def test_segment_names_for_each_allowed_k(self):
        """Le regole di denominazione danno nomi distinti e sensati per ogni k ammesso"""
        for k in C.K_ALLOWED:
            names = set(S.run_segmentation(self.df, k=k).names.values())
            self.assertEqual(len(names), k, names)
            self.assertIn("Inattivi / a rischio abbandono", names)
            self.assertIn("Clienti fedeli ad alto valore", names)
            self.assertFalse(any(n.startswith("Gruppo ") for n in names), names)
        self.assertIn("Nuovi clienti in fase di prova", S.run_segmentation(self.df, k=4).names.values())

    def test_scaling_zero_mean_unit_std(self):
        self.assertTrue(np.allclose(self.X.mean(axis=0), 0, atol=1e-9))
        self.assertTrue(np.allclose(self.X.std(axis=0), 1, atol=1e-9))

    def test_k_chosen_within_assignment_range(self):
        scan = S.scan_k(self.X)
        k = S.choose_k(scan)
        self.assertIn(k, C.K_ALLOWED)
        sub = scan[scan["k"].isin(C.K_ALLOWED)]
        self.assertEqual(k, int(sub.loc[sub["silhouette"].idxmax(), "k"]))  # regola -> silhouette massima fra 3 e 5

    def test_reproducible(self):
        a = S.run_segmentation(self.df, k=4).labels
        b = S.run_segmentation(self.df, k=4).labels
        self.assertTrue((a == b).all())

    def test_silhouette_in_valid_range_and_decent(self):
        scan = S.scan_k(self.X)
        self.assertTrue(scan["silhouette"].between(-1, 1).all())
        self.assertGreater(scan.loc[scan["k"] == 4, "silhouette"].iloc[0], 0.4)

    def test_profile_consistency(self):
        seg = S.run_segmentation(self.df, k=5)
        self.assertEqual(int(seg.profile["clienti"].sum()), len(self.df))
        self.assertAlmostEqual(seg.profile["quota_ricavi_%"].sum(), 100, places=6)
        self.assertEqual(len(set(seg.names.values())), 5)

    def test_predict_matches_training_label(self):
        seg = S.run_segmentation(self.df, k=5)
        lab, _ = seg.predict(self.df.head(50))
        self.assertTrue((lab == seg.labels[:50]).all())

    def test_new_customers_get_sensible_segment(self):
        seg = S.run_segmentation(self.df, k=5)
        new = pd.DataFrame([
            dict(richieste_90g=12, importo_totale_90g=650, giorni_ultima_richiesta=3, anzianita_giorni=190),
            dict(richieste_90g=0, importo_totale_90g=0, giorni_ultima_richiesta=200, anzianita_giorni=230),
        ])
        _, names = seg.predict(new)
        self.assertEqual(names[0], "Clienti fedeli ad alto valore")
        self.assertEqual(names[1], "Inattivi / a rischio abbandono")

    def test_tenure_distinguishes_new_customers(self):
        """Stesso RFM ma anzianità diversa: solo l'iscritto da pochi giorni è un nuovo cliente"""
        seg = S.run_segmentation(self.df, k=5)
        base = dict(richieste_90g=2, importo_totale_90g=70, giorni_ultima_richiesta=5)
        _, names = seg.predict(pd.DataFrame([dict(base, anzianita_giorni=10), dict(base, anzianita_giorni=220)]))
        self.assertEqual(names[0], "Nuovi clienti in fase di prova")
        self.assertNotEqual(names[1], "Nuovi clienti in fase di prova")

    def test_predict_derives_tenure_from_signup_date(self):
        seg = S.run_segmentation(self.df, k=5)
        rec = dict(richieste_90g=2, importo_totale_90g=70, giorni_ultima_richiesta=5)
        a, _ = seg.predict(pd.DataFrame([dict(rec, data_iscrizione=pd.Timestamp("2026-09-20"))]))
        b, _ = seg.predict(pd.DataFrame([dict(rec, anzianita_giorni=10)]))
        self.assertEqual(int(a[0]), int(b[0]))

    def test_every_axis_choice_of_the_dashboard_can_be_plotted(self):
        """Ogni variabile scelta come asse nella dashboard deve esistere nei dati"""
        import plots as P
        seg = S.run_segmentation(self.df, k=5)
        data = S.assign_dataframe(self.df, seg)
        cmap = P.color_map(data["segmento"])
        for x in P.LABELS:
            for y in P.LABELS:
                if x != y:
                    fig = P.scatter_fig(data, x, y, cmap)
                    P.describe_scatter(data, x, y)
                    fig.clf()

    def test_filtered_subsets_with_missing_clusters_do_not_crash(self):
        import plots as P
        seg = S.run_segmentation(self.df, k=5)
        data = S.assign_dataframe(self.df, seg)
        cmap = P.color_map(data["segmento"])
        one = data[data["cluster"] == data["cluster"].iloc[0]]
        for sub in (one, one.head(10), data[data["cluster"].isin([0, 1])], data.head(10), data.iloc[0:0]):
            prof = S.profile_segments(sub, sub["cluster"].values, names=seg.names)
            if len(sub):
                P.size_revenue_fig(prof.assign(segmento=prof.index.map(seg.names)), cmap).clf()
                P.scatter_fig(sub, "giorni_ultima_richiesta", "importo_totale_90g", cmap).clf()
                P.describe_scatter(sub, "giorni_ultima_richiesta", "importo_totale_90g")
        self.assertEqual(S.name_segments(S.profile_segments(data.iloc[0:0], [])), {})

    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_other_algorithms_run(self):
        for algo in ("gmm", "agglomerative"):
            seg = S.run_segmentation(self.df, k=4, algo=algo)
            self.assertEqual(len(seg.labels), len(self.df))
            lab, _ = seg.predict(self.df.head(5))
            self.assertEqual(len(lab), 5)


if __name__ == "__main__":
    unittest.main()
