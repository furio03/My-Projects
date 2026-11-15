"""Launch the portfolio desktop application. No other module opens a window."""


from pathlib import Path
import sys

# Direct execution must resolve the package from its parent directory.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true', help='Start with synthetic data, without a download.')
    args = parser.parse_args()

    import tkinter as tk
    from portfolio_analysis.gui import PortfolioApp

    root = tk.Tk()
    app = PortfolioApp(root)
    if args.demo:
        root.after(150, lambda: app.start(demo=True))
    root.mainloop()


if __name__ == '__main__':
    main()
