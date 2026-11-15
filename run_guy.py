"""Launch the portfolio desktop application. No other module opens a window."""


def main():
    import argparse
    import tkinter as tk
    from portfolio_analysis.gui import PortfolioApp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true', help='Start with synthetic data, without a download.')
    args = parser.parse_args()
    root = tk.Tk()
    app = PortfolioApp(root)
    if args.demo:
        root.after(150, lambda: app.start(demo=True))
    root.mainloop()


if __name__ == '__main__':
    main()
