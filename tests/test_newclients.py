"""Test sui nuovi clienti e sul riaddestramento"""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import newclients as N  # noqa: E402
import segmentation as S  # noqa: E402


def rec(**kw):
    base = dict(zona="Centro", data_iscrizione=pd.Timestamp("2026-03-01"), richieste_90g=3, importo_totale_90g=120.0,
                giorni_ultima_richiesta=20.0, categoria_preferita="Pulizie", piano="Plus")
    base.update(kw)
    return base


class TestValidation(unittest.TestCase):
    def test_valid_record_has_no_errors(self):
        self.assertEqual(N.validate_new_client(rec())[0], [])

    def test_incoherent_records_are_rejected(self):
        bad = [rec(richieste_90g=0, importo_totale_90g=50.0, giorni_ultima_richiesta=120.0),   # importo senza richieste
               rec(richieste_90g=2, importo_totale_90g=0.0),                                    # richieste senza importo
               rec(richieste_90g=0, importo_totale_90g=0.0, giorni_ultima_richiesta=30.0),      # 0 richieste ma recente
               rec(richieste_90g=2, giorni_ultima_richiesta=150.0),                             # richieste ma vecchie
               rec(data_iscrizione=pd.Timestamp("2026-09-20"), giorni_ultima_richiesta=40.0),   # ultima prima dell'iscrizione
               rec(data_iscrizione=pd.Timestamp("2026-12-01"))]                                 # iscrizione nel futuro
        for r in bad:
            self.assertTrue(N.validate_new_client(r)[0], r)

    def test_zero_requests_with_old_last_request_is_valid(self):
        r = rec(richieste_90g=0, importo_totale_90g=0.0, giorni_ultima_richiesta=120.0)
        errors, warns = N.validate_new_client(r)
        self.assertEqual(errors, [])
        self.assertTrue(warns)  # ma il segmento è solo indicativo

    def test_single_request_warns_but_is_saved(self):
        errors, warns = N.validate_new_client(rec(richieste_90g=1))
        self.assertEqual(errors, [])
        self.assertTrue(warns)


class TestStorageAndRetrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = S.load_data()
        cls.df, _ = S.clean_data(cls.base)
        cls.seg = S.run_segmentation(cls.df, k=5)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path = self.tmp / "nuovi.csv"
        self.state = self.tmp / "stato.json"

    def test_append_assigns_incremental_ids_and_keeps_originals(self):
        before = self.base.copy()
        r1 = N.append_new(rec(), "Clienti regolari", self.path)
        r2 = N.append_new(rec(zona="Sud"), "Clienti occasionali", self.path)
        self.assertEqual((r1["customer_id"], r2["customer_id"]), ("N0001", "N0002"))
        saved = N.load_new(self.path)
        self.assertEqual(list(saved["customer_id"]), ["N0001", "N0002"])
        self.assertTrue(set(saved["customer_id"]).isdisjoint(self.base["customer_id"]))
        pd.testing.assert_frame_equal(before, S.load_data())  # il file originale non cambia

    def test_missing_state_means_original_model(self):
        self.assertEqual(N.get_active_count(self.state), 0)
        N.set_active_count(3, self.state)
        self.assertEqual(N.get_active_count(self.state), 3)

    def test_training_data_includes_only_the_first_n_new_clients(self):
        for z in ("Sud", "Nord-Est", "Isole"):
            N.append_new(rec(zona=z), "x", self.path)
        n0 = len(N.training_data(0, self.base, self.path)[0])
        n2 = len(N.training_data(2, self.base, self.path)[0])
        self.assertEqual(n2, n0 + 2)

    def test_retrain_comparison_is_consistent(self):
        for z in ("Sud", "Nord-Est", "Isole", "Centro"):
            N.append_new(rec(zona=z), "x", self.path)
        df_all, _ = N.training_data(4, self.base, self.path)
        res = N.retrain_comparison(df_all, self.seg, "kmeans", 5)
        self.assertEqual(res["clienti_aggiunti"], 4)
        self.assertEqual(len(res["seg_new"].labels), len(df_all))
        self.assertEqual(int(res["tabella"]["clienti_dopo"].sum()), len(df_all))
        self.assertEqual(int(res["tabella"]["clienti_prima"].sum()), len(self.df))
        self.assertGreater(res["ari"], 0.8)  # pochi clienti in più -> i gruppi restano sostanzialmente gli stessi
        self.assertTrue(0 <= res["quota_cambiata_pct"] <= 100)

    def test_explain_returns_assigned_and_second_segment(self):
        e = self.seg.explain(pd.DataFrame([rec()]))
        self.assertIn(e["segmento"], self.seg.names.values())
        self.assertNotEqual(e["segmento"], e["secondo"])
        self.assertGreater(e["rapporto"], 0)


if __name__ == "__main__":
    unittest.main()
