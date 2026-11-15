"""Test del grafico Categoria preferita per zona (plots.py)"""

import sys, unittest, pathlib
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C
import plots as P


class TestZoneCategoria(unittest.TestCase):
    def test_una_barra_per_zona_e_percentuali_sommano_100(self):
        df = pd.read_csv(ROOT / "data" / "clienti.csv")
        fig = P.zone_category_fig(df)
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        self.assertEqual(len(labels), len(C.ZONES))
        totals = {z: int((df["zona"] == z).sum()) for z in C.ZONES}
        for z in C.ZONES:
            self.assertTrue(any(lbl.startswith(f"{z} (n={totals[z]})") for lbl in labels))
        # somma delle larghezze deve essere 100 x ogni zona
        for i, z in enumerate(C.ZONES):
            if totals[z] == 0:
                continue
            larghezza = sum(c.patches[i].get_width() for c in ax.containers)
            self.assertAlmostEqual(larghezza, 100, places=6)
        plt.close(fig)

    def test_vista_vuota_non_crasha(self):
        df = pd.read_csv(ROOT / "data" / "clienti.csv").head(0)
        fig = P.zone_category_fig(df)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
