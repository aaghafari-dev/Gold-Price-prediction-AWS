"""Gold Price Prediction V18 desktop control center.

A non-Streamlit, Windows-friendly Tkinter GUI for data ingestion, preparation,
training, candidate approval, API startup, and review of training artifacts.
"""
from __future__ import annotations
import os, sys, subprocess, threading, queue, json, csv, webbrowser, shutil
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

class GoldApp(tk.Tk):
    """Create the main application window and wire workflow actions to scripts."""
    def __init__(self):
        """Initialize theme, controls, event queue, and dashboard layout."""
        super().__init__()
        self.title("Gold Price Intelligence")
        self.geometry("1180x790")
        self.minsize(980, 680)
        self.configure(bg="#0b1220")
        self.events = queue.Queue()
        self.process = None
        self._style()
        self._build()
        self.after(100, self._poll_events)
        self._refresh_metrics()

    def _style(self):
        """Configure a high-contrast navy, teal, and gold ttk visual theme."""
        style = ttk.Style(self)
        try: style.theme_use("clam")
        except tk.TclError: pass
        style.configure("TFrame", background="#111c2f")
        style.configure("Card.TFrame", background="#17243a")
        style.configure("TLabel", background="#111c2f", foreground="#e6edf7", font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background="#111c2f", foreground="#9fb0c9", font=("Segoe UI", 9))
        style.configure("Hero.TLabel", background="#0b1220", foreground="#f5c451", font=("Segoe UI Semibold", 22))
        style.configure("SubHero.TLabel", background="#0b1220", foreground="#b6c5dc", font=("Segoe UI", 10))
        style.configure("TLabelframe", background="#111c2f", foreground="#f5c451")
        style.configure("TLabelframe.Label", background="#111c2f", foreground="#f5c451", font=("Segoe UI Semibold", 10))
        style.configure("TButton", font=("Segoe UI Semibold", 10), padding=(12, 8), background="#245b73", foreground="#ffffff")
        style.map("TButton", background=[("active", "#2f7890"), ("disabled", "#354154")])
        style.configure("Gold.TButton", background="#d69e2e", foreground="#111827")
        style.map("Gold.TButton", background=[("active", "#f5c451")])
        style.configure("Treeview", background="#0f1a2b", fieldbackground="#0f1a2b", foreground="#e6edf7", rowheight=27, borderwidth=0)
        style.configure("Treeview.Heading", background="#24344e", foreground="#f5c451", font=("Segoe UI Semibold", 9))
        style.map("Treeview", background=[("selected", "#236477")])
        style.configure("TNotebook", background="#0b1220", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(15, 9), background="#17243a", foreground="#cbd5e1")
        style.map("TNotebook.Tab", background=[("selected", "#245b73")], foreground=[("selected", "#ffffff")])

    def _build(self):
        """Build header, workflow controls, results table, and live console."""
        header = tk.Frame(self, bg="#0b1220", padx=24, pady=18)
        header.pack(fill="x")
        ttk.Label(header, text="GOLD PRICE PREDICTION", style="Hero.TLabel").pack(anchor="w")
        ttk.Label(header, text="GOLD PRICE INTELLIGENCE  /  RESEARCH WORKBENCH ", style="SubHero.TLabel").pack(anchor="w", pady=(3,0))
        self.status = tk.StringVar(value="READY  •  Local mode  •  No cloud resources launched")
        tk.Label(header, textvariable=self.status, bg="#102b32", fg="#7de2c5", padx=12, pady=7, font=("Segoe UI", 9, "bold")).pack(anchor="e", side="right", pady=(0,5))
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=18, pady=(0,12))
        self.workflow_tab = ttk.Frame(notebook, padding=16)
        self.results_tab = ttk.Frame(notebook, padding=16)
        self.about_tab = ttk.Frame(notebook, padding=16)
        self.forecast_tab = ttk.Frame(notebook, padding=16)
        self.aws_tab = ttk.Frame(notebook, padding=16)
        notebook.add(self.workflow_tab, text="  WORKFLOW  ")
        notebook.add(self.results_tab, text="  MODEL RESULTS  ")
        notebook.add(self.forecast_tab, text="  FORECAST CENTER  ")
        notebook.add(self.aws_tab, text="  AWS CONTROL CENTER  ")
        notebook.add(self.about_tab, text="  METHOD & HELP  ")
        self._build_workflow(); self._build_results(); self._build_about(); self._build_forecast_tab(); self._build_aws_tab()
        footer = tk.Frame(self, bg="#0b1220", padx=18, pady=8)
        footer.pack(fill="x")
        tk.Label(footer, text="RESEARCH PROTOTYPE  •  Validate data coverage and out-of-sample results before any financial use", bg="#0b1220", fg="#8294ae", font=("Segoe UI", 8)).pack(side="left")

    def _build_workflow(self):
        """Create data-source settings and buttons for the end-to-end workflow."""
        pane = self.workflow_tab
        settings = ttk.LabelFrame(pane, text="01  DATA SETTINGS", padding=12)
        settings.pack(fill="x")
        self.use_google_rss = tk.BooleanVar(value=True)
        self.use_newsapi = tk.BooleanVar(value=False)
        self.use_alphavantage = tk.BooleanVar(value=False)
        self.use_gdelt = tk.BooleanVar(value=False)
        self.use_yfinance_news = tk.BooleanVar(value=False)
        self.news_query = tk.StringVar(value='"gold" OR bullion OR "precious metals" OR "Federal Reserve" OR inflation OR geopolitics')
        # One shared lookback prevents the news and market windows drifting apart.
        self.analysis_window = tk.StringVar(value="5y")
        self.timespan = self.analysis_window
        self.years = self.analysis_window  # Backward-compatible alias for older GUI callbacks.
        self.coverage = tk.StringVar(value="0.80")
        self.test_fraction = tk.StringVar(value="0.20")
        self.symbol = tk.StringVar(value="GC=F")
        self.allow_unverified_news = tk.BooleanVar(value=False)
        self.use_news_features = tk.BooleanVar(value=False)
        self.enable_tuning = tk.BooleanVar(value=True)
        self.compute_mode = tk.StringVar(value="CPU (recommended)")
        self.robust_seeds = tk.BooleanVar(value=False)
        self.newsapi_key = tk.StringVar(value=os.getenv("NEWSAPI_API_KEY", ""))
        self.alphavantage_key = tk.StringVar(value=os.getenv("ALPHAVANTAGE_API_KEY", ""))
        self.sentiment_method = tk.StringVar(value="lexicon")
        grid = tk.Frame(settings, bg="#111c2f"); grid.pack(fill="x")
        def field(label, var, col, width=22):
            """Add a labeled setting field to the source-configuration grid."""
            box = tk.Frame(grid, bg="#111c2f"); box.grid(row=0, column=col, sticky="ew", padx=7, pady=4)
            ttk.Label(box, text=label, style="Muted.TLabel").pack(anchor="w")
            ttk.Entry(box, textvariable=var, width=width).pack(fill="x", pady=(4,0))
        field("News search query", self.news_query, 0, 46)
        lookbacks = ["1d", "3d", "7d", "14d", "30d", "90d", "180d", "6mo", "1y", "2y", "5y", "10y", "20y", "30y"]
        ttk.Label(grid, text="Shared analysis window — news + market", style="Muted.TLabel").grid(row=0, column=1, columnspan=2, sticky="w", padx=7, pady=4)
        ttk.Combobox(grid, textvariable=self.analysis_window, values=lookbacks, state="readonly", width=18).grid(row=1, column=1, sticky="ew", padx=7, pady=(0,4))
        ttk.Label(grid, text="Same window is used for both sources", style="Muted.TLabel").grid(row=1, column=2, sticky="w", padx=7, pady=(0,4))
        field("Minimum logged news-window overlap (0–1)", self.coverage, 3, 20)
        field("Final chronological test fraction (0.10–0.40)", self.test_fraction, 4, 22)
        source_row = tk.Frame(settings, bg="#111c2f"); source_row.pack(fill="x", pady=(9,0))
        ttk.Label(source_row, text="News providers (key-free RSS is the default)", style="Muted.TLabel").pack(side="left", padx=(7,10))
        ttk.Checkbutton(source_row, text="Google News RSS", variable=self.use_google_rss).pack(side="left", padx=5)
        ttk.Checkbutton(source_row, text="NewsAPI (API key)", variable=self.use_newsapi).pack(side="left", padx=5)
        ttk.Checkbutton(source_row, text="Alpha Vantage (API key)", variable=self.use_alphavantage).pack(side="left", padx=5)
        ttk.Checkbutton(source_row, text="Legacy GDELT", variable=self.use_gdelt).pack(side="left", padx=5)
        ttk.Checkbutton(source_row, text="Legacy Yahoo", variable=self.use_yfinance_news).pack(side="left", padx=5)
        key_row = tk.Frame(settings, bg="#111c2f"); key_row.pack(fill="x", pady=(7,0))
        ttk.Label(key_row, text="NewsAPI key", style="Muted.TLabel").pack(side="left", padx=(7,4))
        ttk.Entry(key_row, textvariable=self.newsapi_key, show="•", width=28).pack(side="left", padx=4)
        ttk.Label(key_row, text="Alpha Vantage key", style="Muted.TLabel").pack(side="left", padx=(12,4))
        ttk.Entry(key_row, textvariable=self.alphavantage_key, show="•", width=28).pack(side="left", padx=4)
        ttk.Button(key_row, text="Save API keys to .env", command=self.save_news_keys).pack(side="left", padx=8)
        options_row = tk.Frame(settings, bg="#111c2f"); options_row.pack(fill="x", pady=(8,0))
        ttk.Label(options_row, text="Sentiment file", style="Muted.TLabel").pack(side="left", padx=(7,8))
        ttk.Combobox(options_row, textvariable=self.sentiment_method, values=["lexicon", "finbert"], state="readonly", width=12).pack(side="left")
        ttk.Button(options_row, text="Score News", command=self.score_news).pack(side="left", padx=8)
        ttk.Checkbutton(options_row, text="Use news features in ML experiment", variable=self.use_news_features).pack(side="left", padx=10)
        ttk.Checkbutton(options_row, text="Exploratory unverified news (no promotion)", variable=self.allow_unverified_news).pack(side="left", padx=10)
        ttk.Label(options_row, text="Long historical news may be unavailable from RSS/API providers.", style="Muted.TLabel").pack(side="left", padx=8)
        robust_row = tk.Frame(settings, bg="#111c2f"); robust_row.pack(fill="x", pady=(4,0))
        ttk.Checkbutton(robust_row, text="Nested time-series tuning + feature selection (slower)", variable=self.enable_tuning).pack(side="left", padx=7)
        ttk.Checkbutton(robust_row, text="3-seed sensitivity check (slower)", variable=self.robust_seeds).pack(side="left", padx=7)
        ttk.Label(robust_row, text="Compute:", style="Muted.TLabel").pack(side="left", padx=(12,4))
        ttk.Combobox(robust_row, textvariable=self.compute_mode, values=["CPU (recommended)", "GPU + CPU (XGBoost GPU only)", "Auto (CPU fallback)"], state="readonly", width=22).pack(side="left")
        actions = ttk.LabelFrame(pane, text="02  RUN PIPELINE", padding=12); actions.pack(fill="x", pady=12)
        row1 = tk.Frame(actions, bg="#111c2f"); row1.pack(fill="x")
        ttk.Button(row1, text="Fetch News", command=self.fetch_news).pack(side="left", padx=5, pady=4)
        ttk.Button(row1, text="Prepare Market + Features", command=self.prepare).pack(side="left", padx=5, pady=4)
        ttk.Button(row1, text="Train & Evaluate", style="Gold.TButton", command=self.train).pack(side="left", padx=5, pady=4)
        ttk.Button(row1, text="Approve Candidate", command=self.approve).pack(side="left", padx=5, pady=4)
        row2 = tk.Frame(actions, bg="#111c2f"); row2.pack(fill="x", pady=(6,0))
        ttk.Button(row2, text="Refresh Results", command=self._refresh_metrics).pack(side="left", padx=5)
        ttk.Button(row2, text="Audit News Archive", command=self.audit_news).pack(side="left", padx=5)
        ttk.Button(row2, text="Start Live News Monitor", command=self.live_news_monitor).pack(side="left", padx=5)
        ttk.Button(row2, text="Run Tests", command=self.run_tests).pack(side="left", padx=5)
        ttk.Button(row2, text="Plot Market Data", command=self.plot_market_data).pack(side="left", padx=5)
        row3 = tk.Frame(actions, bg="#111c2f"); row3.pack(fill="x", pady=(5,0))
        ttk.Button(row3, text="Open Project Folder", command=lambda: os.startfile(ROOT)).pack(side="left", padx=5)
        ttk.Button(row3, text="Stop Current Task", command=self.stop_process).pack(side="left", padx=5)
        ttk.Label(row3, text="Cloud commands and deployment are managed in the AWS CONTROL CENTER tab.", style="Muted.TLabel").pack(side="left", padx=12)
        console_frame = ttk.LabelFrame(pane, text="03  LIVE RUN CONSOLE", padding=8); console_frame.pack(fill="both", expand=True)
        self.console = tk.Text(console_frame, height=13, bg="#07101d", fg="#c7d7eb", insertbackground="white", relief="flat", font=("Cascadia Mono", 9), wrap="word")
        sb = ttk.Scrollbar(console_frame, orient="vertical", command=self.console.yview); self.console.configure(yscrollcommand=sb.set)
        self.console.pack(side="left", fill="both", expand=True); sb.pack(side="right", fill="y")
        self._log("Ready. Shared analysis window: the same lookback is passed to market and news ingestion. Very short windows are for ingestion/plot/data-quality checks; ML evaluation needs enough historical trading rows after rolling-feature warm-up.")

    def _build_results(self):
        """Create the model-comparison table and run-summary cards."""
        pane = self.results_tab
        self.summary = tk.StringVar(value="No training result loaded yet.")
        history_bar = ttk.Frame(pane); history_bar.pack(fill="x", pady=(0, 8))
        ttk.Label(history_bar, text="Saved run:").pack(side="left", padx=(0, 8))
        self.run_selector = ttk.Combobox(history_bar, state="readonly", width=48)
        self.run_selector.pack(side="left", padx=(0, 8))
        ttk.Button(history_bar, text="Load selected run", command=self.load_selected_run).pack(side="left")
        ttk.Label(history_bar, text="Run metadata is saved under artifacts/runs/.", style="Muted.TLabel").pack(side="left", padx=10)
        tk.Label(pane, textvariable=self.summary, bg="#111c2f", fg="#f5c451", justify="left", anchor="w", padx=12, pady=12, font=("Segoe UI Semibold", 11)).pack(fill="x", pady=(0,12))
        cols = ("model", "feature_set", "rmse", "rmse_ci95_low", "rmse_ci95_high", "holdout_rmse", "holdout_mae", "holdout_persistence_rmse", "baseline_rmse", "seasonal_naive_rmse", "improvement_pct", "mae", "directional_accuracy", "beats_persistence", "beats_seasonal_naive", "beats_baseline")
        table_frame = ttk.Frame(pane)
        table_frame.pack(fill="both", expand=True)
        self.table = ttk.Treeview(table_frame, columns=cols, show="headings", height=13)
        yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        xscroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.table.xview)
        self.table.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        table_frame.grid_rowconfigure(0, weight=1); table_frame.grid_columnconfigure(0, weight=1)
        self.table.grid(row=0, column=0, sticky="nsew"); yscroll.grid(row=0, column=1, sticky="ns"); xscroll.grid(row=1, column=0, sticky="ew")
        labels = {"model":"Model", "feature_set":"Features", "rmse":"RMSE ↓", "rmse_ci95_low":"RMSE CI low", "rmse_ci95_high":"RMSE CI high", "holdout_rmse":"Holdout RMSE", "holdout_mae":"Holdout MAE", "holdout_persistence_rmse":"Holdout persistence RMSE", "baseline_rmse":"Persistence RMSE", "seasonal_naive_rmse":"Seasonal RMSE", "improvement_pct":"vs strongest %", "mae":"MAE ↓", "directional_accuracy":"Directional Acc.", "beats_persistence":"Beats persistence", "beats_seasonal_naive":"Beats seasonal", "beats_baseline":"Beats both baselines"}
        widths = {"model":145,"feature_set":130,"rmse":95,"rmse_ci95_low":105,"rmse_ci95_high":105,"holdout_rmse":105,"holdout_mae":105,"holdout_persistence_rmse":155,"baseline_rmse":120,"seasonal_naive_rmse":115,"improvement_pct":115,"mae":95,"directional_accuracy":115,"beats_persistence":125,"beats_seasonal_naive":125,"beats_baseline":135}
        for c in cols: self.table.heading(c, text=labels[c]); self.table.column(c, width=widths[c], anchor="center")
        self.table.tag_configure("good", background="#153a36", foreground="#8ef0c9")
        self.table.tag_configure("weak", background="#3a2630", foreground="#ffd0d7")
        self.table.tag_configure("best", background="#7f1d35", foreground="#fff1f2")
        ttk.Button(pane, text="Open artifacts folder", command=lambda: os.startfile(ROOT / "artifacts")).pack(anchor="e", pady=10)


    def plot_market_data(self):
        """Open a data-inspection window for the cached market CSV."""
        path = ROOT / "data" / "raw" / "market_daily.csv"
        if not path.exists():
            messagebox.showinfo("Market data not found", "Run Prepare Market + Features first. Expected file: data/raw/market_daily.csv")
            return
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.dates as mdates
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            df = pd.read_csv(path)
            date_col = "Date" if "Date" in df.columns else ("date" if "date" in df.columns else None)
            if not date_col:
                raise ValueError("Could not find a Date/date column in market_daily.csv")
            gold_col = next((c for c in ("GC=F", "gold_close", "Close", "close") if c in df.columns), None)
            if not gold_col:
                # yfinance normalization often stores the target as a ticker-named column.
                candidates = [c for c in df.columns if "GC=F" in c or "gold" in c.lower()]
                gold_col = candidates[0] if candidates else None
            if not gold_col:
                raise ValueError("Could not identify the gold-price column. Columns: " + ", ".join(df.columns))
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df[gold_col] = pd.to_numeric(df[gold_col], errors="coerce")
            df = df.dropna(subset=[date_col, gold_col]).sort_values(date_col)
            win = tk.Toplevel(self); win.title("Market Data Inspection"); win.geometry("1050x650")
            fig = Figure(figsize=(10, 5.5), dpi=100)
            ax = fig.add_subplot(111)
            x = df[date_col].to_numpy()
            y = df[gold_col].to_numpy(dtype=float)
            line, = ax.plot(df[date_col], df[gold_col], label=f"Gold proxy ({gold_col})", linewidth=1.3, picker=5)
            # A single reusable highlight marker and tooltip keep mouse movement responsive
            # even when the archive contains several thousand daily observations.
            highlight, = ax.plot([], [], marker="o", markersize=8, markerfacecolor="#f5c451",
                                 markeredgecolor="#111827", linestyle="None", zorder=5)
            tooltip = ax.annotate("", xy=(0, 0), xytext=(14, 14), textcoords="offset points",
                                  bbox=dict(boxstyle="round,pad=0.45", fc="#10233a", ec="#f5c451", alpha=0.96),
                                  color="#ffffff", fontsize=9, arrowprops=dict(arrowstyle="->", color="#f5c451"))
            tooltip.set_visible(False)
            ax.set_title("Gold proxy price over downloaded market history")
            ax.set_xlabel("Date"); ax.set_ylabel("Price (provider units)")
            ax.grid(True, alpha=.25); ax.legend(); fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=win); canvas.draw()
            def on_hover(event):
                """Update the plot highlight and tooltip for the nearest visible observation."""
                if event.inaxes != ax or event.xdata is None or event.ydata is None:
                    if tooltip.get_visible():
                        tooltip.set_visible(False); highlight.set_data([], []); canvas.draw_idle()
                    return
                # Use screen-space distance to the rendered line; only show a point when
                # the cursor is reasonably close, avoiding misleading distant selections.
                points = ax.transData.transform(np.column_stack((mdates.date2num(df[date_col]), y)))
                cursor = (event.x, event.y)
                distances = ((points[:, 0] - cursor[0]) ** 2 + (points[:, 1] - cursor[1]) ** 2) ** 0.5
                idx = int(distances.argmin())
                if distances[idx] > 18:
                    if tooltip.get_visible():
                        tooltip.set_visible(False); highlight.set_data([], []); canvas.draw_idle()
                    return
                highlight.set_data([df[date_col].iloc[idx]], [y[idx]])
                tooltip.xy = (df[date_col].iloc[idx], y[idx])
                tooltip.set_text(f"Date: {df[date_col].iloc[idx]:%Y-%m-%d}\nPrice: {y[idx]:,.2f}")
                tooltip.set_visible(True)
                canvas.draw_idle()
            canvas.mpl_connect("motion_notify_event", on_hover)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            NavigationToolbar2Tk(canvas, win).update()
            ttk.Label(win, text=f"{len(df):,} valid rows | {df[date_col].min().date()} to {df[date_col].max().date()} | Source: {path.name}").pack(anchor="w", padx=10, pady=5)
        except Exception as exc:
            messagebox.showerror("Plot failed", f"{type(exc).__name__}: {exc}")

    def _aws_environment(self):
        """Build a child-process environment from the current environment and project .env.

        Credential values are never printed. Explicit .env values are passed only to the
        AWS subprocess launched by this GUI; they are not written to the global environment.
        """
        env = os.environ.copy()
        env_file = ROOT / ".env"
        if env_file.exists():
            try:
                from dotenv import dotenv_values
                values = dotenv_values(env_file)
                for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_DEFAULT_REGION", "AWS_PROFILE"):
                    value = values.get(key)
                    if value and str(value).strip():
                        env[key] = str(value).strip()
            except Exception as exc:
                self._log(f"Warning: could not parse .env for AWS settings: {type(exc).__name__}: {exc}")
        region_var = getattr(self, "aws_region_var", None)
        profile_var = getattr(self, "aws_profile_var", None)
        if region_var and region_var.get().strip():
            env["AWS_DEFAULT_REGION"] = region_var.get().strip()
        else:
            env.setdefault("AWS_DEFAULT_REGION", "eu-central-1")
        if profile_var and profile_var.get().strip():
            profile = profile_var.get().strip()
            if profile.lower() == "default": env.pop("AWS_PROFILE", None)
            else: env["AWS_PROFILE"] = profile
        return env

    def _resolve_cli(self, command):
        """Find an AWS CLI executable even when the GUI inherited a stale PATH.

        Checks the current process PATH, Windows where.exe, PowerShell command
        discovery, and common per-user installation directories. Returns an
        absolute executable path when found, otherwise None.
        """
        found = shutil.which(command) or shutil.which(command + ".exe")
        if found:
            return found
        candidates = []
        where_exe = shutil.which("where.exe") or shutil.which("where")
        if where_exe:
            try:
                result = subprocess.run([where_exe, command], capture_output=True, text=True, timeout=5)
                candidates.extend(line.strip() for line in result.stdout.splitlines() if line.strip())
            except Exception:
                pass
        for shell in (shutil.which("powershell.exe"), shutil.which("powershell"), shutil.which("pwsh.exe"), shutil.which("pwsh")):
            if shell:
                try:
                    result = subprocess.run([shell, "-NoProfile", "-Command", f"(Get-Command {command} -ErrorAction SilentlyContinue).Source"], capture_output=True, text=True, timeout=8)
                    candidates.extend(line.strip() for line in result.stdout.splitlines() if line.strip())
                except Exception:
                    pass
        local = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        if command == "sam":
            candidates.extend([str(local / "Amazon" / "AWSSAMCLI" / "bin" / "sam.exe"), str(program_files / "Amazon" / "AWSSAMCLI" / "bin" / "sam.exe")])
        elif command == "aws":
            candidates.extend([str(program_files / "Amazon" / "AWSCLIV2" / "aws.exe"), str(local / "Programs" / "Amazon" / "AWSCLIV2" / "aws.exe")])
        for candidate in candidates:
            path = Path(candidate.strip('"'))
            if path.is_file():
                return str(path.resolve())
        return None

    def aws_preflight(self):
        """Run non-deploying AWS tool and identity checks with project settings loaded."""
        script = ROOT / "scripts" / "aws_preflight.ps1"
        if not shutil.which("powershell") and not shutil.which("pwsh"):
            messagebox.showerror("PowerShell not found", "Install/enable Windows PowerShell, or run AWS CLI checks in the VS Code PowerShell terminal.")
            return
        shell = shutil.which("powershell") or shutil.which("pwsh")
        self._run([shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)], "AWS preflight (no deployment)", extra_env=self._aws_environment())

    def aws_check_identity(self):
        """Verify AWS identity; provide specific guidance when the token is invalid."""
        aws_exe = self._resolve_cli("aws")
        if not aws_exe:
            messagebox.showerror("AWS CLI not found", "The GUI could not resolve aws.exe. Your terminal may have a newer PATH than this already-running GUI. Close all VS Code windows, reopen VS Code, confirm `aws --version`, then retry. You can also open the project from a fresh terminal. See docs/V18_AWS_BEGINNER_ROADMAP.md.")
            self._log("AWS CLI executable not resolved by PATH, where.exe, PowerShell, or common install locations. No AWS request was made.")
            return
        self._log(f"AWS CLI resolved to: {aws_exe}")
        self._log("AWS identity check: credentials are passed to the child process securely; secret values are never logged.")
        self._run([aws_exe, "sts", "get-caller-identity"], "AWS caller identity check", extra_env=self._aws_environment())

    def sam_validate(self):
        """Validate the SAM template without creating AWS resources."""
        sam_exe = self._resolve_cli("sam")
        if not sam_exe:
            messagebox.showerror("SAM CLI not resolved", "Your terminal reports SAM CLI installed, but this GUI cannot currently resolve sam.exe. Close all VS Code windows and reopen them from a fresh terminal. If it still fails, add the folder containing sam.exe to the Windows user PATH and restart. See docs/V18_AWS_BEGINNER_ROADMAP.md.")
            self._log("SAM Validate: sam.exe was not resolved; checked PATH, where.exe, PowerShell and common install folders. No validation ran.")
            return
        self._log(f"SAM CLI resolved to: {sam_exe}")
        self._run([sam_exe, "validate", "--lint", "--template-file", str(ROOT / "template.yaml")], "AWS SAM template validation", extra_env=self._aws_environment())

    def sam_build(self):
        """Build SAM resources locally; does not deploy them."""
        sam_exe = self._resolve_cli("sam")
        if not sam_exe:
            messagebox.showerror("SAM CLI not resolved", "Your terminal reports SAM CLI installed, but this GUI cannot currently resolve sam.exe. Close all VS Code windows and reopen them from a fresh terminal. If it still fails, add the folder containing sam.exe to the Windows user PATH and restart. See docs/V18_AWS_BEGINNER_ROADMAP.md.")
            self._log("SAM Build: sam.exe was not resolved; checked PATH, where.exe, PowerShell and common install folders. No build ran.")
            return
        self._log(f"SAM CLI resolved to: {sam_exe}")
        self._run([sam_exe, "build", "--template-file", str(ROOT / "template.yaml")], "AWS SAM build (local only)", extra_env=self._aws_environment())

    def _build_forecast_tab(self):
        """Build a GUI-first forecast workflow without requiring users to operate FastAPI."""
        pane = self.forecast_tab
        intro = ttk.LabelFrame(pane, text="RESEARCH FORECAST — NO LOCAL API SERVER REQUIRED", padding=12)
        intro.pack(fill="x", pady=(0, 10))
        ttk.Label(intro, text=("Create a next-available-trading-session estimate from the latest market data and the locally approved champion model. "
            "This runs the project's forecast script in the background and shows its output here; it does not start a web server or deploy anything."),
            wraplength=1000, justify="left").pack(anchor="w")
        controls = ttk.Frame(intro); controls.pack(fill="x", pady=(10, 2))
        ttk.Label(controls, text="Market history to refresh:").pack(side="left", padx=(0, 8))
        self.forecast_period = tk.StringVar(value="2y")
        ttk.Combobox(controls, textvariable=self.forecast_period, values=["6mo", "1y", "2y", "5y", "10y"], state="readonly", width=10).pack(side="left")
        ttk.Button(controls, text="Generate Next-Session Forecast", style="Gold.TButton", command=self.run_forecast).pack(side="left", padx=10)
        ttk.Button(controls, text="Open Forecast JSON", command=self.open_forecast_file).pack(side="left", padx=5)
        ttk.Button(controls, text="Open Forecast Folder", command=lambda: os.startfile(ROOT / "artifacts")).pack(side="left", padx=5)
        ttk.Label(pane, text="LATEST FORECAST OUTPUT", style="Muted.TLabel").pack(anchor="w", pady=(3, 5))
        self.forecast_output = tk.Text(pane, height=20, bg="#07101d", fg="#c7d7eb", insertbackground="white", relief="flat", font=("Cascadia Mono", 10), wrap="word")
        self.forecast_output.pack(fill="both", expand=True)
        self.forecast_output.insert("1.0", "No forecast has been generated in this session.\n\nPrerequisite: models/champion_model.joblib and models/champion_manifest.json must exist. Train, review the chronological holdout against persistence, and approve a model only when justified.\n\nImportant: the forecast is a research estimate for the next available trading session, not necessarily tomorrow's calendar date. GC=F is a gold-futures proxy, not an official spot-price quote.")
        self.forecast_output.configure(state="disabled")
        self._load_forecast_output()

    def _load_forecast_output(self):
        """Display the latest saved forecast JSON when it exists."""
        path = ROOT / "artifacts" / "latest_forecast.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            text = json.dumps(data, indent=2, ensure_ascii=False)
            self.forecast_output.configure(state="normal"); self.forecast_output.delete("1.0", "end"); self.forecast_output.insert("1.0", text); self.forecast_output.configure(state="disabled")
        except Exception as exc:
            self._log(f"Could not load saved forecast: {exc}")

    def run_forecast(self):
        """Run the forecast script as a managed subprocess and refresh its output afterwards."""
        model = ROOT / "models" / "champion_model.joblib"
        manifest = ROOT / "models" / "champion_manifest.json"
        if not model.exists() or not manifest.exists():
            messagebox.showwarning("Approved model required", "No approved champion model was found. Train a candidate, inspect the out-of-sample holdout and baseline comparisons, and approve only if the evidence supports it.")
            return
        self._run([PYTHON, "scripts/predict_next_session.py", "--period", self.forecast_period.get()], "Generate next-session forecast")

    def open_forecast_file(self):
        """Open the forecast JSON with the Windows default application."""
        path = ROOT / "artifacts" / "latest_forecast.json"
        if path.exists():
            os.startfile(path)
        else:
            messagebox.showinfo("Forecast not available", "Generate a forecast first. The result will be written to artifacts/latest_forecast.json.")

    def _build_aws_tab(self):
        """Build a guided AWS control center with explicit, guarded cloud actions."""
        pane = self.aws_tab
        intro = ttk.LabelFrame(pane, text="AWS CONTROL CENTER — REVIEW BEFORE ANY CLOUD-COST ACTION", padding=12)
        intro.pack(fill="x", pady=(0, 10))
        ttk.Label(intro, text=("This tab runs AWS CLI, SAM and project scripts from the desktop app. It uses your existing Windows AWS CLI credentials/profile; "
            "it does not create resources until you explicitly confirm a deployment or pipeline action. Build and validation are local checks."),
            wraplength=1020, justify="left").pack(anchor="w")
        modes = ttk.LabelFrame(pane, text="CHOOSE THE WORKFLOW DIRECTION", padding=10); modes.pack(fill="x", pady=(0, 10))
        self.aws_mode = tk.StringVar(value="local_to_cloud")
        ttk.Radiobutton(modes, text="A. Local data + local ML → AWS storage / deployment", variable=self.aws_mode, value="local_to_cloud", command=self._update_aws_mode_help).pack(anchor="w", pady=2)
        ttk.Radiobutton(modes, text="B. AWS data + AWS processing / ML pipeline", variable=self.aws_mode, value="cloud_native", command=self._update_aws_mode_help).pack(anchor="w", pady=2)
        self.aws_mode_help = ttk.Label(modes, text="Direction A: keep the current Windows data science workflow; upload selected prepared data/artifacts to S3 and deploy the scheduled collector only when ready.", wraplength=1020, justify="left")
        self.aws_mode_help.pack(anchor="w", padx=(22, 0), pady=(4, 0))
        config = ttk.LabelFrame(pane, text="AWS PROJECT SETTINGS", padding=10); config.pack(fill="x", pady=(0, 10))
        self.aws_region_var = tk.StringVar(value=os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", "eu-central-1")))
        self.aws_profile_var = tk.StringVar(value=os.getenv("AWS_PROFILE", "default"))
        self.aws_bucket_var = tk.StringVar(value=os.getenv("S3_BUCKET", ""))
        self.aws_role_var = tk.StringVar(value=os.getenv("SAGEMAKER_ROLE_ARN", ""))
        self.aws_stack_var = tk.StringVar(value="gold-price-prediction")
        fields = [("Region", self.aws_region_var, 18), ("AWS profile (optional)", self.aws_profile_var, 20), ("S3 bucket name (after deploy)", self.aws_bucket_var, 30), ("SageMaker execution role ARN", self.aws_role_var, 42), ("SAM stack name", self.aws_stack_var, 24)]
        grid = tk.Frame(config, bg="#111c2f"); grid.pack(fill="x")
        for i, (label, var, width) in enumerate(fields):
            cell = tk.Frame(grid, bg="#111c2f"); cell.grid(row=i//3, column=i%3, sticky="ew", padx=5, pady=4)
            ttk.Label(cell, text=label, style="Muted.TLabel").pack(anchor="w")
            ttk.Entry(cell, textvariable=var, width=width).pack(anchor="w", fill="x", pady=(3, 0))
        for i in range(3): grid.columnconfigure(i, weight=1)
        ttk.Button(config, text="Save non-secret AWS settings to .env", command=self.save_aws_settings).pack(anchor="w", padx=5, pady=(5, 0))
        actions = ttk.LabelFrame(pane, text="SAFE CHECKS / LOCAL BUILD", padding=10); actions.pack(fill="x", pady=(0, 8))
        row = tk.Frame(actions, bg="#111c2f"); row.pack(fill="x")
        for label, fn in [("Check AWS Identity", self.aws_check_identity), ("AWS Preflight", self.aws_preflight), ("SAM Validate", self.sam_validate), ("SAM Build (local)", self.sam_build), ("List S3 Buckets", self.aws_list_buckets), ("Open AWS Roadmap", self.open_aws_roadmap)]:
            ttk.Button(row, text=label, command=fn).pack(side="left", padx=4, pady=4)
        cloud = ttk.LabelFrame(pane, text="CLOUD ACTIONS — THESE CAN CREATE RESOURCES OR COST MONEY", padding=10); cloud.pack(fill="x", pady=(0, 8))
        row2 = tk.Frame(cloud, bg="#111c2f"); row2.pack(fill="x")
        ttk.Button(row2, text="Deploy SAM Stack…", style="Gold.TButton", command=self.aws_deploy).pack(side="left", padx=4, pady=4)
        ttk.Button(row2, text="Prepare + Upload Market History", command=self.aws_prepare_upload).pack(side="left", padx=4, pady=4)
        ttk.Button(row2, text="Start SageMaker Pipeline…", command=self.aws_start_sagemaker_pipeline).pack(side="left", padx=4, pady=4)
        ttk.Button(row2, text="Open CloudFormation Outputs", command=self.aws_stack_outputs).pack(side="left", padx=4, pady=4)
        ttk.Button(row2, text="Stop Current Task", command=self.stop_process).pack(side="left", padx=4, pady=4)
        ttk.Label(pane, text="All command output is streamed to the existing Live Run Console on the Workflow tab. Use AWS Console to independently verify every created resource.", style="Muted.TLabel", wraplength=1020).pack(anchor="w", pady=(4, 0))

    def _update_aws_mode_help(self):
        """Explain the currently selected local-versus-cloud architecture in plain language."""
        if self.aws_mode.get() == "local_to_cloud":
            msg = "Direction A: data download, feature engineering, model comparison and evaluation remain on this PC. AWS is used for S3 storage and the scheduled Lambda collector; SageMaker can be an optional later deployment. Lower complexity and easier debugging."
        else:
            msg = "Direction B: market snapshots land in S3; SageMaker Processing prepares features, SageMaker Training fits a model, Processing evaluates against persistence, and the pipeline can register a candidate for manual review. This needs an execution role, a valid S3 bucket/raw CSV, SageMaker permissions, and billable managed jobs. News ingestion/feature integration is not yet fully cloud-native in this version."
        self.aws_mode_help.configure(text=msg)

    def save_aws_settings(self):
        """Save only non-secret AWS configuration values to .env."""
        try:
            from dotenv import set_key
            env_path = ROOT / ".env"
            if not env_path.exists():
                example = ROOT / ".env.example"
                env_path.write_text(example.read_text(encoding="utf-8") if example.exists() else "", encoding="utf-8")
            for key, value in [("AWS_DEFAULT_REGION", self.aws_region_var.get().strip()), ("AWS_REGION", self.aws_region_var.get().strip()), ("AWS_PROFILE", self.aws_profile_var.get().strip()), ("S3_BUCKET", self.aws_bucket_var.get().strip()), ("SAGEMAKER_ROLE_ARN", self.aws_role_var.get().strip())]:
                if value:
                    set_key(str(env_path), key, value)
            self._log("Saved AWS region/profile/bucket/role settings to .env. No secret keys were requested or logged.")
            messagebox.showinfo("Settings saved", "Non-secret AWS settings saved. Keep .env private and out of version control.")
        except Exception as exc:
            messagebox.showerror("Could not save settings", str(exc))

    def open_aws_roadmap(self):
        """Open the plain-language AWS roadmap."""
        path = ROOT / "docs" / "V18_AWS_BEGINNER_ROADMAP.md"
        if path.exists(): webbrowser.open(path.as_uri())

    def _aws_cli(self):
        """Resolve the AWS CLI executable for this Windows GUI process."""
        exe = self._resolve_cli("aws")
        if not exe:
            messagebox.showerror("AWS CLI not found", "Install AWS CLI v2, reopen VS Code/the app, and verify `aws --version` in a new terminal.")
        return exe

    def _aws_cli_args(self, *args):
        """Add region and optional profile to an AWS CLI argument list."""
        base = [self._aws_cli()]
        if not base[0]: return None
        profile = self.aws_profile_var.get().strip()
        region = self.aws_region_var.get().strip()
        base += list(args)
        if profile and profile.lower() != "default": base += ["--profile", profile]
        if region and "--region" not in base: base += ["--region", region]
        return base

    def aws_list_buckets(self):
        """List S3 buckets for the selected account and region context."""
        args = self._aws_cli_args("s3", "ls")
        if args: self._run(args, "List S3 buckets", extra_env=self._aws_environment())

    def aws_deploy(self):
        """Deploy the SAM stack only after explicit confirmation and existing guided config."""
        sam = self._resolve_cli("sam")
        if not sam:
            messagebox.showerror("SAM CLI not found", "Install AWS SAM CLI and verify `sam --version` in a new terminal."); return
        config_file = ROOT / "samconfig.toml"
        stack = self.aws_stack_var.get().strip() or "gold-price-prediction"
        first_deploy = not config_file.exists()
        if first_deploy:
            detail = (f"First deployment of stack '{stack}' to region '{self.aws_region_var.get()}'.\n\n"
                      "SAM will use --resolve-s3 to create/use its managed deployment bucket and --capabilities CAPABILITY_IAM. "
                      "This creates a private project S3 bucket, an IAM role, a Lambda collector and a weekday EventBridge schedule. "
                      "The command is non-interactive and will confirm the CloudFormation change set automatically after this confirmation. "
                      "AWS charges may apply. Check AWS identity and run SAM Validate/Build first.")
        else:
            detail = (f"Deploy/update SAM stack '{stack}' in region '{self.aws_region_var.get()}'? This creates/updates a private S3 bucket, IAM role, Lambda function, and scheduled EventBridge rule. AWS charges may apply. Confirm your AWS identity first.")
        if not messagebox.askyesno("Confirm AWS deployment", detail):
            return
        args = [sam, "deploy", "--template-file", str(ROOT / "template.yaml"), "--stack-name", stack,
                "--no-confirm-changeset", "--no-fail-on-empty-changeset", "--region", self.aws_region_var.get().strip()]
        if first_deploy:
            args += ["--resolve-s3", "--capabilities", "CAPABILITY_IAM"]
        profile = self.aws_profile_var.get().strip()
        if profile and profile.lower() != "default": args += ["--profile", profile]
        self._run(args, "Deploy SAM cloud stack", extra_env=self._aws_environment())

    def aws_stack_outputs(self):
        """Read CloudFormation outputs for the selected stack."""
        stack = self.aws_stack_var.get().strip() or "gold-price-prediction"
        args = self._aws_cli_args("cloudformation", "describe-stacks", "--stack-name", stack, "--query", "Stacks[0].Outputs", "--output", "table")
        if args: self._run(args, "Read CloudFormation stack outputs", extra_env=self._aws_environment())

    def aws_prepare_upload(self):
        """Prepare a normalized historical market file and upload it to the configured private S3 bucket."""
        bucket = self.aws_bucket_var.get().strip()
        if not bucket:
            messagebox.showwarning("S3 bucket required", "Enter the bucket name shown in the SAM/CloudFormation output first, then save settings."); return
        if not (ROOT / "data/raw/market_daily.csv").exists():
            messagebox.showwarning("Market data missing", "Run Prepare Market + Features on the Workflow tab first."); return
        if not messagebox.askyesno("Upload market history", f"Upload prepared market history to s3://{bucket}/gold-price-prediction/raw/market/bootstrap_market_history.csv? Confirm the bucket belongs to your intended account."):
            return
        self._run([PYTHON, "scripts/prepare_sagemaker_data.py", "--input", "data/raw/market_daily.csv", "--output-dir", "data/processed/sagemaker"], "Prepare market data for cloud pipeline")
        # The chained upload runs only after preparation succeeds in _poll_events.
        self._pending_s3_upload = bucket

    def aws_start_sagemaker_pipeline(self):
        """Start a managed SageMaker pipeline only with explicit cost/permission confirmation."""
        if self.aws_mode.get() != "cloud_native":
            if not messagebox.askyesno("Cloud-native mode not selected", "The pipeline runs Processing and Training jobs in AWS and can incur charges. Switch to Direction B and continue?"):
                return
            self.aws_mode.set("cloud_native"); self._update_aws_mode_help()
        bucket = self.aws_bucket_var.get().strip()
        role = self.aws_role_var.get().strip()
        if not bucket or not role:
            messagebox.showwarning("Cloud settings incomplete", "Enter and save both the S3 bucket name and SageMaker execution role ARN. The role must trust SageMaker and have least-privilege access to this project's S3 prefix."); return
        if not messagebox.askyesno("Start SageMaker pipeline", "This registers/updates a SageMaker Pipeline and starts a run with managed Processing and Training jobs. It may incur charges. Before continuing, verify the S3 bootstrap CSV exists at gold-price-prediction/raw/market/bootstrap_market_history.csv, the SageMaker execution role is correct, and the AWS identity/region are correct."):
            return
        self._run([PYTHON, "scripts/start_pipeline.py"], "Start SageMaker cloud pipeline", extra_env=self._aws_environment())

    def _build_about(self):
        """Explain source semantics, sentiment method, target, and caveats in the GUI."""
        pane = self.about_tab  # V5 fix: bind the method-card parent to the About tab.
        text = ("V18 METHOD CARD\n\n"
          "MARKET DATA\nGC=F is Yahoo Finance's symbol for a continuous gold-futures proxy (not spot gold and not a guaranteed official COMEX settlement series). The GUI uses it as a convenient educational market proxy, plus configured context tickers via yfinance. The shared analysis-window control requests the same lookback for market and news ingestion. Daily market bars exist only on trading sessions, so weekends and exchange holidays are expected calendar gaps. Features include the current observed gold close (available at forecast time), lagged returns, rolling volatility/momentum and related-market values.\n\n"
          "NEWS SOURCE OPTIONS\n• Google News RSS is the default key-free headline source; NewsAPI and Alpha Vantage keys can be entered in the GUI and saved to .env. Legacy GDELT/Yahoo remain available. The same selected lookback is passed to both market ingestion and the requested news query window. However, this aligns the requested time span, not the actual article archive: RSS feeds and free APIs may only expose recent headlines, may ignore long historical ranges, and do not guarantee completeness. Provider failures and requested query windows are recorded in news_ingestion_log.csv. One-day windows are useful for ingestion/plot/data-quality checks, but cannot support meaningful supervised ML evaluation.\n• Start Live News Monitor polls RSS every five minutes and records FinBERT/lexicon sentiment snapshots. It is a polling monitor, not a push stream or a validated intraday price prediction.\n\n"
          "SENTIMENT\nFetch News must happen before Score News. For the default lexicon, training can calculate the lexicon features itself; Score News creates a visible scored CSV. For FinBERT, score the latest fetched archive first and then run Train & Evaluate. FinBERT is optional and may require downloading model weights.\nDeterministic English keyword lexicon. Article title + summary; lowercase tokenization; unique tokens; score=(positive matches−negative matches)/max(1,total matches). The positive and negative vocabularies live in src/features/news_features.py. No negation, contextual model, FinBERT or LLM by default.\n\n"
          "EVALUATION\nThe GUI includes an explicit Use news features in ML experiment checkbox; market-only is the default control experiment. A configurable final chronological holdout (GUI test fraction, default 20%, range 10–40%) is reserved from the newest observations; development data is evaluated using five chronological TimeSeriesSplit folds by default; gap equals target horizon (one market session). Nested inner chronological grid search and SelectKBest feature selection run inside each outer training fold; outer validation rows do not tune that fold. The GUI intentionally starts this reproducible protocol rather than exposing unsafe random-split controls. Compare against persistence and a 5-session seasonal-naive benchmark. Report pooled RMSE/MAE, directional accuracy, fold metrics, and a moving-block-bootstrap 95% RMSE interval. ‘best=null’ means no model passed the benchmark gate. In Model Results, Beats persistence and Beats seasonal identify which benchmark was beaten; Beats both baselines is True only when both comparisons pass. False is a legitimate research result, not automatically a software failure.\n\n"
          "RESEARCH SAFETY\nNews coverage windows mean a query was attempted, not that all relevant stories were found. Never call recent-feed results a historical news backtest. Use an explicit forecast cutoff, verified point-in-time archives, ablation tests and independent out-of-sample evaluation before making claims.")
        box = tk.Text(pane, bg="#0b1626", fg="#dce7f5", insertbackground="white", relief="flat", wrap="word", font=("Segoe UI", 10), padx=18, pady=18)
        box.pack(fill="both", expand=True); box.insert("1.0", text); box.configure(state="disabled")

    def _log(self, message):
        """Append a timestamped line to the live console without blocking UI."""
        import datetime
        self.console.insert("end", f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}\n")
        self.console.see("end")

    def _run(self, args, label, extra_env=None):
        """Run a project command asynchronously and stream output to the GUI."""
        if self.process and self.process.poll() is None:
            messagebox.showwarning("Task running", "Wait for or stop the current task before starting another."); return
        self.status.set(f"RUNNING  •  {label}")
        self._log(f"START: {label}\n$ {' '.join(map(str,args))}")
        def worker():
            """Run a subprocess in a background thread and forward stdout/exit events to the Tkinter event queue."""
            env=os.environ.copy(); env.setdefault("LOKY_MAX_CPU_COUNT", "4"); env["PYTHONUNBUFFERED"] = "1"
            if extra_env:
                env.update(extra_env)
            try:
                proc=subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env, bufsize=1)
                self.process=proc
                for line in proc.stdout: self.events.put(("line", line.rstrip()))
                code=proc.wait(); self.events.put(("done", label, code))
            except FileNotFoundError as exc:
                if args and str(args[0]).lower() in ("sam", "sam.exe"):
                    detail = ("AWS SAM CLI executable was not found. Install AWS SAM CLI for Windows, restart VS Code, "
                              "then confirm `sam --version` in a new terminal. Identity checks can work without SAM; "
                              "template validation/build cannot. See docs/V18_AWS_BEGINNER_ROADMAP.md.")
                elif args and str(args[0]).lower() in ("aws", "aws.exe"):
                    detail = "AWS CLI executable was not found. Install AWS CLI v2 for Windows and restart VS Code."
                else:
                    detail = f"Executable not found: {args[0] if args else '(empty command)'}. Check the interpreter/tool installation and PATH."
                self.events.put(("error", label, detail))
            except Exception as exc: self.events.put(("error", label, str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def _poll_events(self):
        """Drain subprocess output events on Tk's main thread."""
        try:
            while True:
                item=self.events.get_nowait()
                if item[0]=="line": self._log(item[1])
                elif item[0]=="done":
                    _, label, code=item; self.status.set(("SUCCESS" if code==0 else "CHECK LOG")+f"  •  {label}  •  exit={code}"); self._log(f"Finished: {label}; exit code={code}"); self._refresh_metrics()
                    if label == "Generate next-session forecast" and code == 0: self._load_forecast_output()
                    if label == "Prepare market data for cloud pipeline":
                        bucket = getattr(self, "_pending_s3_upload", None)
                        if code == 0 and bucket:
                            self._pending_s3_upload = None
                            aws_args = self._aws_cli_args("s3", "cp", str(ROOT / "data/processed/sagemaker/bootstrap_market_history.csv"), f"s3://{bucket}/gold-price-prediction/raw/market/bootstrap_market_history.csv")
                            if aws_args: self._run(aws_args, "Upload bootstrap market history to S3", extra_env=self._aws_environment())
                        else:
                            self._pending_s3_upload = None
                elif item[0]=="error": self.status.set("ERROR  •  see console"); self._log(f"ERROR {item[1]}: {item[2]}")
        except queue.Empty: pass
        self.after(100, self._poll_events)

    def save_news_keys(self):
        """Persist optional provider API keys to the project .env without logging their values."""
        from dotenv import set_key
        env_path = ROOT / ".env"
        if not env_path.exists():
            example = ROOT / ".env.example"
            env_path.write_text(example.read_text(encoding="utf-8") if example.exists() else "", encoding="utf-8")
        try:
            for key, value in (("NEWSAPI_API_KEY", self.newsapi_key.get().strip()), ("ALPHAVANTAGE_API_KEY", self.alphavantage_key.get().strip())):
                if value:
                    set_key(str(env_path), key, value)
            messagebox.showinfo("API keys saved", "Optional news API keys were saved to the project .env file. Values are not written to the console. Keep .env private and never commit it to Git.")
            self._log("Optional news provider credentials saved to .env (values hidden).")
        except Exception as exc:
            messagebox.showerror("Could not save keys", f"{type(exc).__name__}: {exc}")

    def fetch_news(self):
        """Fetch from the selected news providers independently; preserve successful provider results."""
        selected = [name for name, enabled in (("google_news_rss", self.use_google_rss.get()), ("newsapi", self.use_newsapi.get()), ("alphavantage", self.use_alphavantage.get()), ("gdelt", self.use_gdelt.get()), ("yfinance", self.use_yfinance_news.get())) if enabled]
        if not selected:
            messagebox.showwarning("Select a news provider", "Select at least one news provider.")
            return
        if self.use_newsapi.get() and not self.newsapi_key.get().strip():
            messagebox.showwarning("NewsAPI key required", "NewsAPI is selected but its key is blank. Enter the key above, then click Save API keys to .env.")
            return
        if self.use_alphavantage.get() and not self.alphavantage_key.get().strip():
            messagebox.showwarning("Alpha Vantage key required", "Alpha Vantage is selected but its key is blank. Enter the key above, then click Save API keys to .env.")
            return
        period = self.analysis_window.get().strip()
        args=[PYTHON,"scripts/fetch_news.py","--source",",".join(selected),
              "--query",self.news_query.get(),"--timespan",period,"--symbol",self.symbol.get()]
        self._run(args,f"Fetch news ({period} requested): " + "+".join(selected))

    def prepare(self):
        """Fetch market data for the shared analysis window and build audited features."""
        period = self.analysis_window.get().strip()
        if not period:
            messagebox.showerror("Invalid input", "Choose a shared analysis window.")
            return
        self._run([PYTHON, "scripts/prepare_training_data.py", "--refresh-market", "--period", period],
                  f"Prepare market + features ({period})")

    def train(self):
        """Run chronological model comparison after validating lookback and evaluation settings.

        Very short lookbacks are useful for ingestion and data-quality checks, but they
        cannot provide enough supervised trading rows after rolling-feature warm-up,
        target alignment, walk-forward folds, and the final chronological holdout.
        """
        period = self.analysis_window.get().strip().lower()
        # Fail early in the GUI rather than launching a training subprocess that is
        # guaranteed to fail for the short diagnostic windows.
        minimum_training_periods = {"1d", "3d", "7d", "14d", "30d", "90d"}
        if period in minimum_training_periods:
            proceed = messagebox.askyesno(
                "Insufficient history for ML evaluation",
                f"The selected shared window is {period}. This is appropriate for market/news ingestion, plotting, and data-quality checks, but normally leaves too few supervised trading rows for walk-forward validation plus a final holdout.\n\n"
                "Recommended: select 6mo or longer for training (preferably 1y–5y), click Prepare Market + Features, inspect the data-quality report, and then click Train & Evaluate.\n\n"
                "Do you still want to attempt training with this short window?",
            )
            if not proceed:
                self._log(f"Training not started: {period} is a short diagnostic window; no training subprocess launched.")
                return
        try: coverage=float(self.coverage.get())
        except ValueError: messagebox.showerror("Invalid input","Coverage must be between 0 and 1."); return
        if not 0 <= coverage <= 1: messagebox.showerror("Invalid input","Coverage must be between 0 and 1."); return
        try: test_fraction=float(self.test_fraction.get())
        except ValueError: messagebox.showerror("Invalid input","Test fraction must be between 0.10 and 0.40."); return
        if not 0.10 <= test_fraction <= 0.40: messagebox.showerror("Invalid input","Test fraction must be between 0.10 and 0.40."); return
        args=[PYTHON,"scripts/train_local.py","--period",self.analysis_window.get(),"--min-news-coverage",str(coverage),"--test-fraction",str(test_fraction)]
        if self.allow_unverified_news.get(): args.append("--allow-unverified-news-experiment")
        if self.use_news_features.get(): args.append("--use-news-features")
        if not self.enable_tuning.get(): args.append("--disable-tuning")
        if self.compute_mode.get() == "GPU + CPU (XGBoost GPU only)": args += ["--compute-mode", "gpu"]
        elif self.compute_mode.get() == "Auto (CPU fallback)": args += ["--compute-mode", "auto"]
        else: args += ["--compute-mode", "cpu"]
        if self.robust_seeds.get(): args.append("--robust-seeds")
        scored = ROOT/"data"/"processed"/f"news_{self.sentiment_method.get()}_scored.csv"
        raw_news = ROOT/"data"/"raw"/"news.csv"
        scored_is_current = scored.exists() and (not raw_news.exists() or scored.stat().st_mtime >= raw_news.stat().st_mtime)
        if self.sentiment_method.get()=="finbert" and not scored_is_current:
            if messagebox.askyesno("FinBERT scores missing or stale", "Run the optional FinBERT scoring step now? It may download model weights and requires requirements-nlp.txt."):
                self.score_news()
            return
        if self.sentiment_method.get()=="lexicon" and scored_is_current:
            args += ["--news",str(scored)]
        self._run(args,"Train and evaluate")

    def live_news_monitor(self):
        """Start a five-minute near-live headline/sentiment polling process; stop it from the GUI when needed."""
        if self.use_newsapi.get() and not self.newsapi_key.get().strip():
            messagebox.showwarning("NewsAPI key required", "Enter and save the NewsAPI key first."); return
        if self.use_alphavantage.get() and not self.alphavantage_key.get().strip():
            messagebox.showwarning("Alpha Vantage key required", "Enter and save the Alpha Vantage key first."); return
        method = self.sentiment_method.get()
        provider = "newsapi" if self.use_newsapi.get() else ("alphavantage" if self.use_alphavantage.get() else "google_news_rss")
        args = [PYTHON, "scripts/live_news_monitor.py", "--provider", provider, "--poll-seconds", "300"]
        if method == "finbert": args.append("--finbert")
        self._run(args, "Live news sentiment monitor (5-minute polling)")

    def audit_news(self):
        """Audit archive metadata and gaps between logged query windows."""
        self._run([PYTHON,"scripts/audit_news_archive.py"],"Audit news archive")

    def run_tests(self):
        """Run the automated project test suite with the selected interpreter."""
        self._run([PYTHON,"-m","pytest","-q"],"Run automated tests")

    def score_news(self):
        """Create a scored article archive for lexicon or optional FinBERT comparison."""
        method=self.sentiment_method.get()
        output=ROOT/"data"/"processed"/f"news_{method}_scored.csv"
        args=[PYTHON,"scripts/compare_sentiment.py","--news","data/raw/news.csv","--out",str(output)]
        if method=="finbert": args.append("--finbert")
        self._run(args,f"Score news with {method}")

    def approve(self):
        """Ask for confirmation before promoting a reviewed candidate model."""
        if not messagebox.askyesno("Human approval", "Have you reviewed artifacts/model_comparison.csv and models/candidate_manifest.json? Approve this candidate for local serving?"):
            return
        self._run([PYTHON,"scripts/promote_candidate.py","--approve"],"Approve candidate")

    def start_api(self):
        """Start the local FastAPI server in a background subprocess."""
        self._run([PYTHON,"-m","uvicorn","src.serving.api:app","--host","127.0.0.1","--port","8000"],"Local prediction API")

    def stop_process(self):
        """Terminate the currently running workflow process when possible."""
        if self.process and self.process.poll() is None:
            self.process.terminate(); self._log("Termination requested for current task."); self.status.set("STOP REQUESTED")
        else: self._log("No running task to stop.")

    def _refresh_metrics(self):
        """Load latest summary/metrics and refresh the saved-run selector."""
        self._refresh_run_selector()
        result_path=ROOT/"artifacts"/"training_result.json"
        best_key=None
        if result_path.exists():
            try:
                data=json.loads(result_path.read_text(encoding="utf-8"))
                best=data.get("best")
                if best: best_key=(best.get("model"),best.get("feature_set"))
                msg=f"Run: {data.get('run_id','—')}   |   rows: {data.get('training_rows','—')}   |   news rows: {data.get('news_rows','—')}   |   status: {data.get('selection_status','—')}\n{data.get('selection_reason','')}\nNews: {data.get('news_experiment_status','not reported')} | query-window span: {_fmt(data.get('news_query_window_coverage_ratio'))} | latest market date: {data.get('market_latest_date','—')}"
                self.summary.set(msg)
            except Exception as exc: self.summary.set(f"Could not read training result: {exc}")
        for item in self.table.get_children(): self.table.delete(item)
        csv_path=ROOT/"artifacts"/"model_comparison.csv"
        if csv_path.exists():
            try:
                with csv_path.open(newline="",encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        good=str(row.get("beats_baseline", "")).lower() in ("true","1")
                        key=(row.get("model"),row.get("feature_set"))
                        values=(row.get("model",""),row.get("feature_set",""),_fmt(row.get("rmse")),_fmt(row.get("rmse_ci95_low")),_fmt(row.get("rmse_ci95_high")),_fmt(row.get("holdout_rmse")),_fmt(row.get("holdout_mae")),_fmt(row.get("holdout_persistence_rmse")),_fmt(row.get("baseline_rmse")),_fmt(row.get("seasonal_naive_rmse")),_fmt(row.get("improvement_pct")),_fmt(row.get("mae")),_fmt(row.get("directional_accuracy")),row.get("beats_persistence",""),row.get("beats_seasonal_naive",""),row.get("beats_baseline",""))
                        tag="best" if best_key and key==best_key else ("good" if good else "weak")
                        self.table.insert("","end",values=values,tags=(tag,))
            except Exception as exc: self._log(f"Metrics load warning: {exc}")

    def _refresh_run_selector(self):
        """Populate the run selector from immutable per-run training audit files."""
        run_dir = ROOT / "artifacts"
        files = sorted(run_dir.glob("training_audit_*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if run_dir.exists() else []
        self.run_files = {f"{p.stem.replace('training_audit_', '')} | {p.name}": p for p in files}
        values = list(self.run_files)
        self.run_selector["values"] = values
        if values and (self.run_selector.get() not in self.run_files):
            self.run_selector.set(values[0])

    def load_selected_run(self):
        """Display metrics and provenance summary for the selected historical run."""
        key = self.run_selector.get()
        path = getattr(self, "run_files", {}).get(key)
        if not path or not path.exists():
            messagebox.showinfo("Run history", "No saved run selected. Train a model first.")
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.summary.set(f"Run: {data.get('run_id', path.parent.name)} | rows: {data.get('training_rows', '—')} | news rows: {data.get('news_rows', '—')} | status: {data.get('selection_status', '—')}\n{data.get('selection_reason', '')}\nCalendar verified: {data.get('gc_session_calendar_audit', {}).get('verified', False)} | archive: {data.get('news_experiment_status', 'not reported')} | Python: {data.get('python_version', '—')}")
            for item in self.table.get_children(): self.table.delete(item)
            rows = data.get("results", [])
            best = data.get("best") or {}
            best_key = (best.get("model"), best.get("feature_set"))
            for row in rows:
                key_tuple = (row.get("model"), row.get("feature_set"))
                good = bool(row.get("beats_baseline", False))
                values = tuple(row.get(c, "") if c in ("model", "feature_set", "beats_persistence", "beats_seasonal_naive", "beats_baseline") else _fmt(row.get(c)) for c in ("model", "feature_set", "rmse", "rmse_ci95_low", "rmse_ci95_high", "holdout_rmse", "holdout_mae", "holdout_persistence_rmse", "baseline_rmse", "seasonal_naive_rmse", "improvement_pct", "mae", "directional_accuracy", "beats_persistence", "beats_seasonal_naive", "beats_baseline"))
                tag = "best" if best_key != (None, None) and key_tuple == best_key else ("good" if good else "weak")
                self.table.insert("", "end", values=values, tags=(tag,))
            self._log(f"Loaded saved run {data.get('run_id', path.parent.name)} from {path}")
        except Exception as exc:
            messagebox.showerror("Run history", f"Could not load selected run: {exc}")

def _fmt(value):
    """Format numeric metric cells consistently, retaining unavailable values."""
    try: return f"{float(value):.4f}"
    except (ValueError,TypeError): return str(value or "—")

if __name__ == "__main__":
    """Launch the V10 desktop workbench."""
    GoldApp().mainloop()
