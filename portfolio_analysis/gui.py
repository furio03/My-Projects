"""Desktop presentation. Window creation belongs exclusively to run_guy.py."""

import json
from pathlib import Path

if __name__ == '__main__':
    launcher = Path(__file__).with_name('run_guy.py')
    raise SystemExit(f'To open Portfolio Lab, run the launcher instead:\npython3 "{launcher}"')

from queue import Empty, Queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .service import Settings, analyze


BG = '#F3F5F8'
INK = '#17283C'
MUTED = '#607086'
NAVY = '#122033'
TEAL = '#008C82'


class PortfolioApp:
    def __init__(self, root):
        self.root = root
        self.result = None
        self.running = False
        self.closed = False
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.queue = Queue()
        self.rows = []
        self.canvases = []
        root.title('Portfolio Lab')
        root.geometry('1320x880')
        root.minsize(1100, 780)
        root.configure(bg=BG)
        self.style()
        self.sidebar()
        self.content = tk.Frame(root, bg=BG)
        self.content.pack(side='left', fill='both', expand=True, padx=28, pady=24)
        tk.Label(self.content, text='Know your downside.', bg=BG, fg=INK,
                 font=('Helvetica Neue', 28, 'bold')).pack(anchor='w')
        self.context = tk.Label(self.content, text='Daily risk. Monthly tail losses. One clear view.',
                                bg=BG, fg=MUTED, font=('Helvetica Neue', 12))
        self.context.pack(anchor='w', pady=(6, 20))
        self.cards = tk.Frame(self.content, bg=BG)
        self.cards.pack(fill='x', pady=(0, 18))
        self.notebook = ttk.Notebook(self.content)
        self.notebook.pack(fill='both', expand=True)
        self.pages = []
        self.overview = self.page('Portfolio & daily risk')
        self.monthly = self.page('Monthly EVT losses')
        self.frontier_page = self.page('Efficient frontier')
        root.bind('<MouseWheel>', self.scroll_page)
        ttk.Label(self.frontier_page, text='Explore the return–risk trade-off', font=('Helvetica Neue', 20, 'bold')).pack(anchor='w', pady=18)
        ttk.Label(self.frontier_page, text='Compare your allocation with long-only portfolios built from the same assets.', wraplength=700).pack(anchor='w')
        self.empty_state()
        self.poll_id = self.root.after(100, self.poll)

    def close(self):
        self.closed = True
        if hasattr(self, 'poll_id'):
            self.root.after_cancel(self.poll_id)
        self.root.destroy()

    def page(self, title):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=f'  {title}  ')
        scrollbar = ttk.Scrollbar(tab, orient='vertical')
        scrollbar.pack(side='right', fill='y')
        canvas = tk.Canvas(tab, bg='white', highlightthickness=0, yscrollcommand=scrollbar.set)
        canvas.pack(fill='both', expand=True)
        scrollbar.configure(command=canvas.yview)
        content = ttk.Frame(canvas, padding=16)
        window = canvas.create_window((0, 0), window=content, anchor='nw')
        content.bind('<Configure>', lambda event: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda event: canvas.itemconfigure(window, width=event.width))
        self.pages.append(canvas)
        return content

    def scroll_page(self, event):
        if self.notebook.tabs() and event.delta:
            self.pages[self.notebook.index('current')].yview_scroll(-1 if event.delta > 0 else 1, 'units')

    def style(self):
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('.', font=('Helvetica Neue', 11))
        style.configure('TFrame', background='white')
        style.configure('TLabel', background='white', foreground=INK)
        style.configure('TNotebook', background=BG, borderwidth=0)
        style.configure('TNotebook.Tab', padding=(12, 10), background='#E5EAF0')
        style.map('TNotebook.Tab', background=[('selected', 'white')])
        style.configure('Treeview', background='white', fieldbackground='white', foreground=INK,
                        rowheight=32, borderwidth=0)
        style.configure('Treeview.Heading', background='#EAF0F5', foreground=INK,
                        font=('Helvetica Neue', 10, 'bold'), padding=8)
        style.map('Treeview', background=[('selected', '#D7EEEB')], foreground=[('selected', INK)])
        style.configure('Accent.TButton', background=TEAL, foreground='white', padding=12)
        style.map('Accent.TButton', background=[('active', '#00776F'), ('disabled', '#637D7B')])

    def sidebar(self):
        panel = tk.Frame(self.root, bg=NAVY, width=300, padx=20, pady=26)
        panel.pack(side='left', fill='y')
        panel.pack_propagate(False)
        tk.Label(panel, text='PORTFOLIO / LAB', bg=NAVY, fg='#62D6C5',
                 font=('Helvetica Neue', 15, 'bold')).pack(anchor='w')
        tk.Label(panel, text='Build your portfolio', bg=NAVY, fg='white',
                 font=('Helvetica Neue', 19, 'bold')).pack(anchor='w', pady=(25, 8))
        tk.Label(panel, text='Allocations must total 100%.', bg=NAVY, fg='#A9B8CA',
                 font=('Helvetica Neue', 10)).pack(anchor='w')
        tk.Label(panel, text='TICKER                         WEIGHT %', bg=NAVY, fg='#A9B8CA',
                 font=('Helvetica Neue', 9, 'bold')).pack(anchor='w', pady=(20, 8))
        scroll = tk.Frame(panel, bg=NAVY)
        scroll.pack(fill='x')
        canvas = tk.Canvas(scroll, height=135, bg=NAVY, highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        self.holdings = tk.Frame(canvas, bg=NAVY)
        canvas.create_window((0, 0), window=self.holdings, anchor='nw')
        self.holdings.bind('<Configure>', lambda event: canvas.configure(scrollregion=canvas.bbox('all')))
        self.add_row('AAPL', '60')
        self.add_row('MSFT', '40')
        tk.Button(panel, text='+ Add holding', command=self.add_row, bg=NAVY, fg='#62D6C5',
                  relief='flat', highlightthickness=0, font=('Helvetica Neue', 11)).pack(anchor='w', pady=(4, 12))
        self.fields = {}
        settings = tk.Frame(panel, bg=NAVY)
        settings.pack(fill='x', pady=(5, 15))
        for row, (label, value) in enumerate([
            ('Portfolio value', '100000'), ('Daily observations', '3780'),
            ('Simulation scenarios', '100000'), ('EVT threshold (%)', '95'),
        ]):
            tk.Label(settings, text=label, bg=NAVY, fg='#D9E3EE', font=('Helvetica Neue', 10)).grid(
                row=row, column=0, sticky='w', pady=7)
            entry = self.entry(settings, value, 10)
            entry.grid(row=row, column=1, padx=(9, 0))
            self.fields[label] = entry
        self.run_button = ttk.Button(panel, text='Analyze portfolio', style='Accent.TButton', command=self.start)
        self.run_button.pack(fill='x', pady=(4, 8))
        self.demo_button = ttk.Button(panel, text='Explore with synthetic data', command=lambda: self.start(demo=True))
        self.demo_button.pack(fill='x', pady=(0, 8))
        self.export_button = ttk.Button(panel, text='Export results', command=self.export, state='disabled')
        self.export_button.pack(fill='x')
        self.progress = ttk.Progressbar(panel, mode='indeterminate')
        self.progress.pack(fill='x', pady=(20, 8))
        self.status = tk.Label(panel, text='Ready when you are.', wraplength=255, justify='left',
                               bg=NAVY, fg='#A9B8CA', font=('Helvetica Neue', 10))
        self.status.pack(anchor='w')
        tk.Label(panel, text='21 trading sessions · Seed 42\nUse prices in the same currency.\nDaily rebalancing; no FX conversion.',
                 wraplength=255, justify='left', bg=NAVY, fg='#A9B8CA',
                 font=('Helvetica Neue', 9)).pack(side='bottom', anchor='w', pady=(10, 0))

    def entry(self, parent, value, width):
        entry = tk.Entry(parent, width=width, bg='#24354B', fg='white', insertbackground='white',
                         relief='flat', font=('Helvetica Neue', 12))
        entry.insert(0, value)
        return entry

    def add_row(self, ticker='', weight=''):
        frame = tk.Frame(self.holdings, bg=NAVY)
        frame.pack(anchor='w', pady=5)
        symbol = self.entry(frame, ticker, 11)
        allocation = self.entry(frame, weight, 7)
        symbol.pack(side='left', ipady=6)
        allocation.pack(side='left', padx=8, ipady=6)
        record = (frame, symbol, allocation)
        self.rows.append(record)
        tk.Button(frame, text='×', command=lambda: self.remove_row(record), bg=NAVY,
                  fg='#A9B8CA', relief='flat', highlightthickness=0).pack(side='left')

    def remove_row(self, row):
        self.rows.remove(row)
        row[0].destroy()

    def settings(self):
        weights = {}
        for _, ticker, allocation in self.rows:
            symbol = ticker.get().strip().upper()
            if not symbol or symbol in weights:
                raise ValueError('Every holding needs a unique, nonempty ticker.')
            try:
                weights[symbol] = float(allocation.get().replace(',', '.')) / 100
            except ValueError as error:
                raise ValueError(f'Allocation for {symbol} must be a number, such as 25.') from error
        values = {}
        for label, entry in self.fields.items():
            try:
                parser = int if label in ('Daily observations', 'Simulation scenarios') else float
                values[label] = parser(entry.get().replace(',', '.'))
            except ValueError as error:
                raise ValueError(f'{label} must be a valid number.') from error
        return Settings(
            weights=weights, value=values['Portfolio value'],
            observations=values['Daily observations'], scenarios=values['Simulation scenarios'],
            quantile=values['EVT threshold (%)'] / 100,
        )

    def start(self, demo=False):
        if self.running:
            return
        try:
            settings = self.settings()
        except ValueError as error:
            messagebox.showerror('Check portfolio inputs', str(error), parent=self.root)
            return
        self.running = True
        self.run_button.configure(state='disabled')
        self.demo_button.configure(state='disabled')
        self.export_button.configure(state='disabled')
        self.progress.start(12)
        self.status.configure(text='Starting analysis…')
        threading.Thread(target=self.work, args=(settings, demo), daemon=True).start()

    def work(self, settings, demo):
        try:
            result = analyze(settings, demo, progress=lambda text: self.queue.put(('progress', text)))
            self.queue.put(('result', result))
        except Exception as error:
            self.queue.put(('error', str(error)))

    def poll(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == 'progress':
                    self.status.configure(text=payload)
                else:
                    self.running = False
                    self.progress.stop()
                    self.run_button.configure(state='normal')
                    self.demo_button.configure(state='normal')
                    if kind == 'result':
                        self.result = payload
                        self.render()
                        self.export_button.configure(state='normal')
                        self.status.configure(text='Analysis ready. Results use the allocations submitted.')
                    else:
                        self.status.configure(text='Analysis failed. Check inputs or connection and retry.')
                        messagebox.showerror('Analysis could not finish', payload, parent=self.root)
                        if self.result is not None:
                            self.export_button.configure(state='normal')
        except Empty:
            pass
        self.poll_id = self.root.after(100, self.poll)

    def empty_state(self):
        ttk.Label(self.overview, text='Your portfolio, in perspective', font=('Helvetica Neue', 20, 'bold')).pack(anchor='w', pady=18)
        ttk.Label(self.overview, text='Enter your holdings and analyze, or explore the app with synthetic data.',
                  wraplength=700).pack(anchor='w')
        ttk.Label(self.monthly, text='What could a difficult month look like?', font=('Helvetica Neue', 20, 'bold')).pack(anchor='w', pady=18)
        ttk.Label(self.monthly, text='EVT estimates end-of-month loss probabilities at seven levels, from 1% to 25%.',
                  wraplength=700).pack(anchor='w')

    def clear(self, parent):
        for child in parent.winfo_children():
            child.destroy()

    def table(self, parent, frame, height):
        area = ttk.Frame(parent)
        area.pack(fill='x', pady=(8, 14))
        headings = ['Metric'] + list(frame.columns)
        tree = ttk.Treeview(area, columns=list(range(len(headings))), show='headings', height=height)
        for i, label in enumerate(headings):
            tree.heading(i, text=str(label))
            tree.column(i, width=145 if i == 0 else 125, minwidth=75, anchor='w' if i == 0 else 'e')
        for index, row in frame.iterrows():
            label = ' / '.join(map(str, index)) if isinstance(index, tuple) else str(index)
            tree.insert('', 'end', values=[label] + [f'{value:,.3f}' if isinstance(value, float) else str(value) for value in row])
        vertical = ttk.Scrollbar(area, orient='vertical', command=tree.yview)
        vertical.pack(side='right', fill='y')
        tree.configure(yscrollcommand=vertical.set)
        bar = ttk.Scrollbar(area, orient='horizontal', command=tree.xview)
        tree.configure(xscrollcommand=bar.set)
        tree.pack(fill='x')
        bar.pack(fill='x')
        return tree

    def card(self, title, value):
        panel = tk.Frame(self.cards, bg='white', padx=16, pady=14)
        panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        tk.Label(panel, text=title, bg='white', fg=MUTED, font=('Helvetica Neue', 10)).pack(anchor='w')
        tk.Label(panel, text=value, bg='white', fg=INK, font=('Helvetica Neue', 22, 'bold')).pack(anchor='w', pady=(6, 0))

    def render(self):
        result = self.result
        self.canvases.clear()
        for page in self.pages:
            page.yview_moveto(0)
        for parent in (self.cards, self.overview, self.monthly, self.frontier_page):
            self.clear(parent)
        mode = 'DEMO · Synthetic data' if result['demo'] else 'Yahoo Finance · Adjusted prices'
        self.context.configure(text=f"{mode}  |  {result['start']:%d %b %Y} – {result['end']:%d %b %Y}  |  {result['settings'].observations:,} returns")
        summary = result['summary'].iloc[-1]
        self.card('Annual mean · arithmetic', f"{summary.iloc[0]:.2f}%")
        self.card('Annual volatility', f"{summary.iloc[1]:.2f}%")
        self.card('Daily log-loss tail index ξ', f"{result['tail'].shape:.3f}" if result['evt'] is not None else 'Unavailable')
        probability = result['evt'].loc[5., 'Probability (%)'] if result['evt'] is not None else None
        self.card('P(month loss ≥ 5%)', f'{probability:.3f}%' if probability is not None else 'Unavailable')
        ttk.Label(self.overview, text='Annualized portfolio statistics', font=('Helvetica Neue', 14, 'bold')).pack(anchor='w')
        self.table(self.overview, result['summary'], min(5, len(result['summary'])))
        ttk.Label(self.overview, text='One-day losses · historical and copula models', font=('Helvetica Neue', 14, 'bold')).pack(anchor='w')
        self.table(self.overview, result['daily'], 6)
        ttk.Label(self.overview, text='Positive VaR and ES mean losses. Annual means are arithmetic estimates, not CAGR.\nCopula margins stay within historical asset extremes; estimates are static and in-sample.',
                  foreground=MUTED, wraplength=800).pack(anchor='w', pady=5)
        self.render_monthly(result)
        self.render_frontier(result)

    def render_monthly(self, result):
        if result['evt'] is None:
            ttk.Label(self.monthly, text='More tail data is needed', font=('Helvetica Neue', 18, 'bold')).pack(anchor='w', pady=15)
            ttk.Label(self.monthly, text=result['evt_error'], wraplength=750).pack(anchor='w')
            return
        tail = result['tail']
        lower, upper = result['xi_interval']
        ttk.Label(self.monthly, text='Loss relative to today, after 21 sessions', font=('Helvetica Neue', 16, 'bold')).pack(anchor='w')
        ttk.Label(self.monthly, text=f'ξ = {tail.shape:.3f}  |  Bootstrap 95%: [{lower:.3f}, {upper:.3f}]  |  {len(tail.excesses)} exceedances above the {tail.quantile:.1%} threshold',
                  foreground=MUTED, wraplength=820).pack(anchor='w', pady=(6, 0))
        table = result['evt'].copy()
        table.index = [f'≥ {level:.0f}%' for level in table.index]
        table.index.name = 'Loss'
        self.table(self.monthly, table, 7)
        figure = Figure(figsize=(7, 2.2), dpi=100, facecolor='white')
        axis = figure.add_subplot(111)
        table = result['evt']
        values = table['Probability (%)'].to_numpy()
        axis.bar(range(7), values, color=TEAL, width=0.6)
        axis.errorbar(range(7), values, yerr=[values - table['MC lower 95% (%)'], table['MC upper 95% (%)'] - values],
                      fmt='none', ecolor=INK, capsize=3)
        axis.set_xticks(range(7), [f'{level:.0f}%' for level in table.index])
        axis.set_ylabel('Probability (%)')
        axis.set_xlabel('End-of-month loss at least')
        axis.spines[['top', 'right']].set_visible(False)
        figure.tight_layout()
        canvas = FigureCanvasTkAgg(figure, master=self.monthly)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='x')
        self.canvases.append(canvas)
        ttk.Label(self.monthly, text='Intervals in the table are Monte Carlo counting error only. Zero events do not mean zero risk.\nEmpirical body + GPD tail; independent days and daily rebalancing. Parameter and model uncertainty remain.',
                  foreground=MUTED, wraplength=820).pack(anchor='w', pady=(10, 0))
        ttk.Label(self.monthly, text='Tail-index sensitivity to the threshold', font=('Helvetica Neue', 13, 'bold')).pack(anchor='w', pady=(16, 0))
        self.table(self.monthly, result['stability'].set_index('Threshold quantile'), len(result['stability']))

    def render_frontier(self, result):
        model = result['frontier_model']
        frontier = result['frontier']
        cloud = result['opportunity_set']
        current = model.point(result['settings'].weights)
        minimum = model.point(model.minimum_weights)
        ttk.Label(self.frontier_page, text='Efficient frontier · long-only, fully invested', font=('Helvetica Neue', 16, 'bold')).pack(anchor='w')
        ttk.Label(self.frontier_page, text='Historical arithmetic returns and sample covariance, annualized over 252 sessions.',
                  foreground=MUTED, wraplength=800).pack(anchor='w', pady=(6, 10))
        figure = Figure(figsize=(7.5, 4.2), dpi=100, facecolor='white')
        axis = figure.add_subplot(111)
        axis.scatter(cloud['Annual volatility (%)'], cloud['Annual return (%)'],
                     s=8, c='#C2D2DF', alpha=0.4, label='Feasible allocations')
        axis.plot(frontier['Annual volatility (%)'], frontier['Annual return (%)'],
                  color=TEAL, linewidth=2.5, marker='o' if len(frontier) == 1 else None,
                  label='Efficient frontier')
        axis.scatter(current['Annual volatility (%)'], current['Annual return (%)'],
                     color='#E89935', marker='*', s=190, zorder=5, label='Your portfolio')
        axis.scatter(minimum['Annual volatility (%)'], minimum['Annual return (%)'],
                     color=INK, marker='D', s=55, zorder=5, label='Minimum variance')
        for i, asset in enumerate(model.assets):
            point = model.point([float(j == i) for j in range(model.dimension)])
            axis.scatter(point['Annual volatility (%)'], point['Annual return (%)'], s=30, color=MUTED)
            axis.annotate(asset, (point['Annual volatility (%)'], point['Annual return (%)']),
                          xytext=(5, 5), textcoords='offset points', fontsize=9, color=MUTED)
        axis.set_xlabel('Annual volatility (%)')
        axis.set_ylabel('Annual expected return (%)')
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(alpha=0.15)
        axis.legend(loc='best', fontsize=9)
        figure.tight_layout()
        canvas = FigureCanvasTkAgg(figure, master=self.frontier_page)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='x')
        self.frontier_figure = figure
        self.canvases.append(canvas)
        comparison = result['summary'].iloc[[-1]].copy()
        comparison.index = ['Your portfolio']
        comparison.loc['Minimum variance'] = [minimum['Annual return (%)'], minimum['Annual volatility (%)']]
        self.table(self.frontier_page, comparison, 2)
        weights = {asset: [result['settings'].weights[asset] * 100, weight * 100]
                   for asset, weight in zip(model.assets, model.minimum_weights)}
        self.table(self.frontier_page, pd.DataFrame(weights, index=['Your weights (%)', 'Minimum variance (%)']), 2)
        ttk.Label(self.frontier_page, text=(
            'The curve starts at the global minimum-variance portfolio. Short sales and leverage are excluded.\n'
            'Estimated expected returns are uncertain; the frontier is not a forecast or a recommendation. '
            'EVT loss probabilities are a separate model.'),
                  foreground=MUTED, wraplength=820).pack(anchor='w', pady=10)

    def export(self):
        if self.result is None:
            return
        folder = filedialog.askdirectory(title='Choose a folder for the result CSVs', parent=self.root)
        if not folder:
            return
        try:
            result = self.result
            path = Path(folder)
            result['summary'].to_csv(path / 'portfolio_summary.csv')
            result['daily'].to_csv(path / 'daily_risk.csv')
            result['frontier'].to_csv(path / 'efficient_frontier.csv', index=False)
            self.frontier_figure.savefig(path / 'efficient_frontier.png', dpi=180, bbox_inches='tight')
            if result['evt'] is not None:
                result['evt'].to_csv(path / 'monthly_evt_probabilities.csv')
                result['stability'].to_csv(path / 'evt_threshold_stability.csv', index=False)
            metadata = {'demo': result['demo'], 'weights': result['settings'].weights,
                        'start': str(result['start']), 'end': str(result['end']),
                        'sessions': 21, 'scenarios': result['settings'].scenarios,
                        'seed': result['settings'].seed, 'value': result['settings'].value,
                        'observations': result['settings'].observations,
                        'threshold_quantile': result['settings'].quantile,
                        'evt_error': result['evt_error'],
                        'assumptions': 'Independent daily returns; daily rebalanced weights; same currency.'}
            if result['evt'] is not None:
                metadata.update({'xi': result['tail'].shape, 'xi_bootstrap_95': list(result['xi_interval']),
                                 'gpd_scale': result['tail'].scale, 'log_loss_threshold': result['tail'].threshold,
                                 'exceedances': len(result['tail'].excesses)})
            (path / 'analysis_metadata.json').write_text(json.dumps(metadata, indent=2))
            self.status.configure(text='Results exported successfully.')
        except OSError as error:
            messagebox.showerror('Could not export results', str(error), parent=self.root)
