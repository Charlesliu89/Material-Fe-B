import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import enthalpy_batch_cli as bep
    from enthalpy_plot import build_binary_figure
except Exception as exc:  # pragma: no cover - skip if optional deps are missing
    bep = None
    build_binary_figure = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class BatchEnthalpyPlotsTests(unittest.TestCase):
    @unittest.skipIf(IMPORT_ERROR is not None, f"Import failed: {IMPORT_ERROR}")
    def test_softmax_sums_to_one(self) -> None:
        values = [0.2, -0.1, 0.5]
        result = bep._softmax(values)
        self.assertAlmostEqual(sum(result), 1.0, places=6)
        self.assertTrue(all(value > 0 for value in result))

    @unittest.skipIf(IMPORT_ERROR is not None, f"Import failed: {IMPORT_ERROR}")
    def test_fractions_respect_minimum(self) -> None:
        fractions = bep._fractions_from_logits([1.0, 0.0, -1.0], min_fraction=0.05)
        self.assertAlmostEqual(sum(fractions), 1.0, places=6)
        self.assertTrue(all(value >= 0.05 for value in fractions))

    @unittest.skipIf(IMPORT_ERROR is not None, f"Import failed: {IMPORT_ERROR}")
    def test_write_or_html_creates_output(self) -> None:
        fig = build_binary_figure(["Fe", "Ni"], [0.0, 0.5, 1.0], [0.0, -1.2, 0.0])
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "test_plot.png"
            bep.write_or_html(fig, target)
            png_exists = target.exists()
            html_exists = target.with_suffix(".html").exists()
            self.assertTrue(png_exists or html_exists)
