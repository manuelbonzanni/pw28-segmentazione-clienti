"""Smoke test della dashboard con Streamlit (AppTest). Viene saltato se Streamlit non è installato"""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "src" / "app.py"

try:
    from streamlit.testing.v1 import AppTest
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


@unittest.skipUnless(HAS_STREAMLIT, "streamlit non installato: smoke test della dashboard saltato")
class TestApp(unittest.TestCase):
    def test_app_runs(self):
        at = AppTest.from_file(str(APP), default_timeout=60).run()
        self.assertFalse(at.exception)


if __name__ == "__main__":
    unittest.main()
