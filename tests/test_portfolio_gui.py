"""Opt-in desktop checks: RUN_PORTFOLIO_GUI_TESTS=1 requires a working display."""

import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('RUN_PORTFOLIO_GUI_TESTS') == '1', 'Requires a desktop display')
class DesktopTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        from portfolio_analysis.gui import PortfolioApp
        self.root = tk.Tk()
        self.app = PortfolioApp(self.root)

    def tearDown(self):
        self.app.close()

    def test_holdings_and_validation(self):
        app = self.app
        app.add_row('GOOG', '10')
        self.assertEqual(len(app.rows), 3)
        with self.assertRaisesRegex(ValueError, 'totaling 100'):
            app.settings()
        app.remove_row(app.rows[-1])
        self.assertEqual(app.settings().weights, {'AAPL': 0.6, 'MSFT': 0.4})
        app.rows[1][1].delete(0, 'end')
        app.rows[1][1].insert(0, 'AAPL')
        with patch('portfolio_analysis.gui.messagebox.showerror') as error:
            app.start()
        error.assert_called_once()
        self.assertFalse(app.running)

    def test_background_analysis_chart_and_export(self):
        app = self.app
        for name, value in [('Daily observations', '1200'), ('Simulation scenarios', '2000')]:
            app.fields[name].delete(0, 'end')
            app.fields[name].insert(0, value)
        with patch('portfolio_analysis.gui.messagebox.showerror') as errors:
            app.start(demo=True)
            self.assertTrue(app.running)
            deadline = time.monotonic() + 30
            while app.running and time.monotonic() < deadline:
                self.root.update()
                time.sleep(0.02)
            self.root.update()
        errors.assert_not_called()
        self.assertFalse(app.running)
        self.assertIsNotNone(app.result['evt'])
        app.notebook.select(1)
        self.root.update()
        self.assertGreater(app.canvases[0].get_tk_widget().winfo_width(), 600)
        app.notebook.select(2)
        self.root.update()
        self.assertGreater(app.canvases[1].get_tk_widget().winfo_width(), 600)
        with tempfile.TemporaryDirectory() as folder:
            with patch('portfolio_analysis.gui.filedialog.askdirectory', return_value=folder):
                app.export()
            self.assertTrue((Path(folder) / 'monthly_evt_probabilities.csv').exists())
            self.assertTrue((Path(folder) / 'analysis_metadata.json').exists())
            self.assertTrue((Path(folder) / 'efficient_frontier.csv').exists())
            self.assertGreater((Path(folder) / 'efficient_frontier.png').stat().st_size, 1000)

    def test_failed_download_can_be_retried(self):
        app = self.app
        with patch('portfolio_analysis.gui.analyze', side_effect=ValueError('Offline')):
            with patch('portfolio_analysis.gui.messagebox.showerror') as error:
                app.start()
                deadline = time.monotonic() + 5
                while app.running and time.monotonic() < deadline:
                    self.root.update()
                    time.sleep(0.02)
        error.assert_called_once()
        self.assertFalse(app.running)
        self.assertEqual(str(app.run_button['state']), 'normal')
        self.assertIsNone(app.result)


if __name__ == '__main__':
    unittest.main()
