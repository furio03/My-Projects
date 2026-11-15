import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from portfolio_analysis.service import Settings, analyze


class ServiceTests(unittest.TestCase):
    def test_imports_do_not_open_a_window(self):
        with patch('tkinter.Tk', side_effect=AssertionError('Import opened a window')):
            for name in ('portfolio_analysis', 'portfolio_analysis.gui', 'portfolio_analysis.run_guy'):
                importlib.reload(importlib.import_module(name))

    def test_launcher_runs_from_outside_the_repository(self):
        launcher = Path(__file__).resolve().parents[1] / 'portfolio_analysis' / 'run_guy.py'
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(launcher), '--help'], cwd=directory,
                                    capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--demo', result.stdout)

    def test_internal_gui_file_points_to_the_launcher(self):
        script = Path(__file__).resolve().parents[1] / 'portfolio_analysis' / 'gui.py'
        result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('run_guy.py', result.stderr)
        self.assertNotIn('Traceback', result.stderr)

    def test_settings_reject_invalid_weights_and_work_sizes(self):
        for values in [dict(weights={'A': 0.9}), dict(weights={'A': -1}),
                       dict(weights={'A': 1}, observations=100),
                       dict(weights={'A': 1}, scenarios=10),
                       dict(weights={'A': 1}, value=np.nan)]:
            with self.assertRaises(ValueError):
                Settings(**values)

    def test_offline_service_produces_report_and_keeps_evt_failure_visible(self):
        settings = Settings({'AAA': 0.6, 'BBB': 0.4}, observations=1200, scenarios=2000)
        progress = []
        with patch('portfolio_analysis.service.download_prices', side_effect=AssertionError('Demo download')):
            result = analyze(settings, demo=True, progress=progress.append)
        self.assertTrue(result['demo'])
        self.assertEqual(len(result['evt']), 7)
        self.assertEqual(len(result['daily']), 6)
        self.assertGreater(len(progress), 2)
        self.assertTrue(np.isfinite(result['tail'].shape))
        short = analyze(Settings({'AAA': 1}, observations=252, scenarios=1000), demo=True)
        self.assertIsNone(short['evt'])
        self.assertIn('50 threshold', short['evt_error'])
        self.assertEqual(len(short['daily']), 6)


if __name__ == '__main__':
    unittest.main()
