"""Verifiche delle correzioni di revisione: esportazione intera, allineamento nel riaddestramento, parametri in config"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config as C
import generate_data as G
import newclients as N
import segmentation as S


class TestCorrezioni(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, _ = S.clean_data(G.generate())
        cls.seg = S.run_segmentation(cls.df, k=4)

    def test_export_intero_senza_decimali(self):
        out = S.for_export(self.df)
        self.assertEqual(str(out["richieste_90g"].dtype), "Int64")
        self.assertTrue((out["richieste_90g"].astype(float) == self.df["richieste_90g"]).all())

    def test_export_non_altera_dati_non_interi(self):
        d = self.df.copy()
        d.loc[d.index[0], "giorni_ultima_richiesta"] = 10.5
        self.assertEqual(str(S.for_export(d)["giorni_ultima_richiesta"].dtype), "float64")

    def test_export_tabella_vuota_o_oggetto(self):
        import pandas as pd
        vuota = pd.DataFrame({"richieste_90g": pd.Series([], dtype=object),
                              "giorni_ultima_richiesta": pd.Series([], dtype=object)})
        S.for_export(vuota)  # non deve sollevare errori

    def test_riaddestramento_rifiuta_ordine_diverso(self):
        scambiato = self.df.iloc[::-1].reset_index(drop=True)
        with self.assertRaises(ValueError):
            N.retrain_comparison(scambiato, self.seg, "kmeans", 4)

    def test_parametri_in_config(self):
        self.assertEqual(C.ASSIGN_MARGIN, 0.8)
        self.assertEqual(C.MIN_VIEW, 10)
        self.assertIn("Clienti occasionali", C.CAMPAIGN_ASSUMPTIONS)
        self.assertNotIn("Clienti regolari", C.CAMPAIGN_ASSUMPTIONS)


if __name__ == "__main__":
    unittest.main()
