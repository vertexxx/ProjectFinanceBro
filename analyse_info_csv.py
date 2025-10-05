#!/usr/bin/env python
"""
Analyze and visualize results CSV produced by download_csv_finance_infos.py.

Features:
- Graceful handling of missing columns / values
- Summary stats for annual returns, avg 5Y return, volatility
- Rankings (top/bottom by avg return, lowest volatility)
- Plots saved to disk (no GUI required):
  * Heatmap of annual returns (instrument vs year)
  * Bar chart of Avg5YReturn
  * Bar chart of VolatilityAnnualized
  * Scatter plot: Avg5YReturn vs Volatility (risk-return)
  * Correlation heatmap between instruments (based on overlapping yearly returns)
Hints (not hard errors) are printed if data is missing or dependencies absent.
"""

import argparse
import os
import sys
import math
import warnings
from pathlib import Path

# Force non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pandas as pd
except ImportError:
    print("Need pandas installed. pip install pandas", file=sys.stderr)
    sys.exit(1)

# Optional seaborn
try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False

warnings.filterwarnings("ignore", category=RuntimeWarning)


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        print(f"Input file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Could not read CSV (hint: encoding/format?): {e}", file=sys.stderr)
        sys.exit(1)
    return df


def detect_year_columns(df: pd.DataFrame):
    year_cols = []
    for c in df.columns:
        if c.startswith("YearReturn_"):
            year_part = c.replace("YearReturn_", "")
            if year_part.isdigit():
                year_cols.append(c)
    # Sort by numeric year asc
    year_cols.sort(key=lambda x: int(x.split("_")[1]))
    return year_cols


def to_num(series):
    return pd.to_numeric(series, errors="coerce")


def safe_float(x):
    if x is None:
        return math.nan
    try:
        if isinstance(x, str) and not x.strip():
            return math.nan
        return float(x)
    except Exception:
        return math.nan


def summarize(df: pd.DataFrame, year_cols, avg_col, vol_col):
    print("=== Summary ===")
    if not year_cols:
        print("No YearReturn_<YYYY> columns (hint: run the data fetch script first).")
    else:
        yr_stack = []
        for c in year_cols:
            col_vals = to_num(df[c])
            yr_stack.append(col_vals.rename(c))
        if yr_stack:
            all_vals = pd.concat(yr_stack, axis=0)
            all_vals = all_vals.dropna()
            if all_vals.empty:
                print("Year returns present but all empty / missing.")
            else:
                print(f"Yearly returns count={len(all_vals)} mean={all_vals.mean():.4f} median={all_vals.median():.4f} std={all_vals.std():.4f}")
                # Per year
                for c in year_cols:
                    col = to_num(df[c]).dropna()
                    if col.empty:
                        print(f"  {c}: (missing)")
                    else:
                        print(f"  {c}: mean={col.mean():.4f} median={col.median():.4f} n={len(col)}")
    if avg_col in df.columns:
        avg_vals = to_num(df[avg_col]).dropna()
        if avg_vals.empty:
            print("Avg5YReturn: (all missing)")
        else:
            print(f"Avg5YReturn: mean={avg_vals.mean():.4f} median={avg_vals.median():.4f} n={len(avg_vals)}")
    else:
        print("Avg5YReturn column missing (hint: expected).")

    if vol_col in df.columns:
        vol_vals = to_num(df[vol_col]).dropna()
        if vol_vals.empty:
            print("VolatilityAnnualized: (all missing)")
        else:
            print(f"VolatilityAnnualized: mean={vol_vals.mean():.4f} median={vol_vals.median():.4f} n={len(vol_vals)}")
    else:
        print("VolatilityAnnualized column missing.")


def rankings(df: pd.DataFrame, avg_col: str, vol_col: str):
    print("\n=== Rankings ===")
    name_col = "Name" if "Name" in df.columns else (df.columns[0] if len(df.columns) else None)
    if not name_col:
        print("No identifiable name column for rankings.")
        return
    if avg_col in df.columns:
        sub = df[[name_col, avg_col]].copy()
        sub[avg_col] = to_num(sub[avg_col])
        sub = sub.dropna(subset=[avg_col])
        if sub.empty:
            print("No data for Avg5YReturn ranking.")
        else:
            top = sub.sort_values(avg_col, ascending=False).head(5)
            print("Top Avg5YReturn:")
            for _, r in top.iterrows():
                print(f"  {r[name_col]}: {r[avg_col]:.4f}")
            worst = sub.sort_values(avg_col, ascending=True).head(5)
            print("Bottom Avg5YReturn:")
            for _, r in worst.iterrows():
                print(f"  {r[name_col]}: {r[avg_col]:.4f}")
    else:
        print("Avg5YReturn column missing (ranking skipped).")

    if vol_col in df.columns:
        subv = df[[name_col, vol_col]].copy()
        subv[vol_col] = to_num(subv[vol_col])
        subv = subv.dropna(subset=[vol_col])
        if subv.empty:
            print("No volatility data for ranking.")
        else:
            low = subv.sort_values(vol_col, ascending=True).head(5)
            print("Lowest VolatilityAnnualized:")
            for _, r in low.iterrows():
                print(f"  {r[name_col]}: {r[vol_col]:.4f}")
    else:
        print("Volatility column missing (ranking skipped).")


def plot_avg_bar(df, avg_col, name_col, outdir):
    if avg_col not in df.columns or name_col not in df.columns:
        return
    dat = df[[name_col, avg_col]].copy()
    dat[avg_col] = to_num(dat[avg_col])
    if dat[avg_col].notna().sum() == 0:
        print("Hint: Avg5YReturn all missing (avg bar plot skipped).")
        return
    dat = dat.sort_values(avg_col, ascending=False)
    plt.figure(figsize=(10, 5))
    if _HAS_SEABORN:
        sns.barplot(data=dat, x=avg_col, y=name_col, palette="viridis")
    else:
        plt.barh(dat[name_col], dat[avg_col])
        plt.gca().invert_yaxis()
    plt.title("Average 5Y Return")
    plt.xlabel("Avg5YReturn")
    plt.ylabel("")
    for i, v in enumerate(dat[avg_col]):
        if not math.isnan(v):
            plt.text(v, i, f"{v:.2%}", va='center', ha='left', fontsize=8)
    plt.tight_layout()
    path = os.path.join(outdir, "plot_avg5y_return_bar.png")
    plt.savefig(path, dpi=140)
    plt.close()


def plot_vol_bar(df, vol_col, name_col, outdir):
    if vol_col not in df.columns or name_col not in df.columns:
        return
    dat = df[[name_col, vol_col]].copy()
    dat[vol_col] = to_num(dat[vol_col])
    dat = dat.dropna(subset=[vol_col])
    if dat.empty:
        print("Hint: Volatility data missing (vol bar plot skipped).")
        return
    dat = dat.sort_values(vol_col, ascending=False)
    plt.figure(figsize=(10, 5))
    if _HAS_SEABORN:
        sns.barplot(data=dat, x=vol_col, y=name_col, palette="magma")
    else:
        plt.barh(dat[name_col], dat[vol_col])
        plt.gca().invert_yaxis()
    plt.title("Annualized Volatility")
    plt.xlabel("VolatilityAnnualized")
    plt.ylabel("")
    for i, v in enumerate(dat[vol_col]):
        plt.text(v, i, f"{v:.2%}", va='center', ha='left', fontsize=8)
    plt.tight_layout()
    path = os.path.join(outdir, "plot_volatility_bar.png")
    plt.savefig(path, dpi=140)
    plt.close()


def plot_heatmap_returns(df, year_cols, name_col, outdir):
    if not year_cols or name_col not in df.columns:
        return
    mat = df[[name_col] + year_cols].copy()
    for c in year_cols:
        mat[c] = to_num(mat[c])
    # Optional: shorten extremely long names for display but keep mapping
    display_names = []
    for n in mat[name_col].astype(str).tolist():
        display_names.append(n if len(n) <= 40 else (n[:37] + "..."))
    mat[name_col] = display_names
    mat.set_index(name_col, inplace=True)

    if mat.dropna(how="all").empty:
        print("Hint: All yearly returns missing (heatmap skipped).")
        return

    n_years = len(year_cols)
    n_instr = len(mat.index)

    # Dynamic figure size
    fig_w = max(3, min(0.95 * n_years + 2.0, 30))
    fig_h = max(3, min(0.45 * n_instr + 1.5, 40))

    # Dynamic font sizes
    base_size = 11
    cell_scale = max(n_years, n_instr)
    annot_fs = max(6, min(10, 260 / (cell_scale + 10)))
    tick_fs = max(6, min(11, 200 / (cell_scale + 5)))

    plt.figure(figsize=(fig_w, fig_h))
    if _HAS_SEABORN:
        sns.heatmap(
            mat,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=0,
            cbar=True,
            linewidths=0.4,
            linecolor='lightgray',
            annot_kws={'fontsize': annot_fs},
            cbar_kws={'shrink': 0.6, 'pad': 0.02}
        )
        plt.xticks(rotation=45, ha='right', fontsize=tick_fs)
        plt.yticks(fontsize=tick_fs)
    else:
        data = mat.values
        plt.imshow(data, aspect='auto', cmap='RdYlGn', vmin=-0.5, vmax=0.5)
        plt.colorbar(shrink=0.6, pad=0.02)
        plt.yticks(range(len(mat.index)), mat.index, fontsize=tick_fs)
        plt.xticks(range(len(year_cols)), year_cols, rotation=45, ha='right', fontsize=tick_fs)
        # Manual annotations
        for i in range(len(mat.index)):
            for j in range(len(year_cols)):
                val = data[i, j]
                if not (isinstance(val, float) and math.isnan(val)):
                    plt.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=annot_fs)

    plt.title("Annual Returns Heatmap", fontsize=base_size + 1)
    plt.tight_layout()
    path = os.path.join(outdir, "plot_annual_returns_heatmap.png")
    plt.savefig(path, dpi=160)
    plt.close()


def plot_risk_return(df, avg_col, vol_col, name_col, outdir):
    if any(c not in df.columns for c in (avg_col, vol_col, name_col)):
        return
    dat = df[[name_col, avg_col, vol_col]].copy()
    dat[avg_col] = to_num(dat[avg_col])
    dat[vol_col] = to_num(dat[vol_col])
    dat = dat.dropna(subset=[avg_col, vol_col])
    if dat.empty:
        print("Hint: Need both Avg5YReturn and Volatility for scatter (skipped).")
        return
    plt.figure(figsize=(8, 6))
    if _HAS_SEABORN:
        sns.scatterplot(data=dat, x=vol_col, y=avg_col)
    else:
        plt.scatter(dat[vol_col], dat[avg_col])
    for _, r in dat.iterrows():
        plt.text(r[vol_col], r[avg_col], r[name_col], fontsize=7, ha='left', va='bottom')
    plt.xlabel("VolatilityAnnualized")
    plt.ylabel("Avg5YReturn")
    plt.title("Risk vs Return")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, "plot_risk_return_scatter.png")
    plt.savefig(path, dpi=140)
    plt.close()


def plot_correlation(df, year_cols, name_col, outdir):
    if len(year_cols) < 2 or name_col not in df.columns:
        return
    # Build matrix instrument x year
    pivot = df[[name_col] + year_cols].copy()
    for c in year_cols:
        pivot[c] = to_num(pivot[c])
    # Require at least 2 non-NaN rows for correlation to have meaning
    if pivot[year_cols].dropna(how="all").empty:
        print("Hint: Not enough data for correlation heatmap (skipped).")
        return
    # Correlation across instruments: treat each instrument's vector of year returns
    data = pivot.set_index(name_col)
    # Drop columns (years) that are all NaN
    data = data.dropna(axis=1, how="all")
    if data.shape[1] < 2:
        print("Hint: Need at least 2 year columns with data for correlation heatmap (skipped).")
        return
    corr = data.T.corr(min_periods=1)
    if corr.isna().all().all():
        print("Hint: Correlation matrix all NaN (skipped).")
        return
    plt.figure(figsize=(0.6 * len(corr) + 3, 0.6 * len(corr) + 3))
    if _HAS_SEABORN:
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
                    cbar=True, linewidths=0.5, linecolor='white')
    else:
        plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.index)), corr.index)
        # annotate
        for i in range(len(corr)):
            for j in range(len(corr)):
                val = corr.iloc[i, j]
                if not math.isnan(val):
                    plt.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=7)
    plt.title("Instrument Return Correlation")
    plt.tight_layout()
    path = os.path.join(outdir, "plot_correlation_heatmap.png")
    plt.savefig(path, dpi=140)
    plt.close()


def select_input_file() -> str:
    """
    Open a file dialog to select the input CSV if none supplied.
    Falls back with a hint if GUI not available.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        print("No input file provided and GUI file dialog unavailable. Pass a path argument.", file=sys.stderr)
        sys.exit(1)
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select analysis CSV ( *_out.csv )",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initialdir="."
    )
    root.destroy()
    if not path:
        print("No file selected. Exiting.", file=sys.stderr)
        sys.exit(1)
    return path


def main():
    ap = argparse.ArgumentParser(description="Analyze and plot instrument performance CSV.")
    ap.add_argument("input_csv", nargs="?", help="Path to *_out.csv produced earlier")
    ap.add_argument("--outdir", help="(Optional) Override output directory (default: <inputstem>_analysis)")
    ap.add_argument("--prefix", default="", help="Optional filename prefix for plots")
    args = ap.parse_args()

    if not args.input_csv:
        print("No input file specified. Opening file dialog...")
        args.input_csv = select_input_file()

    # Derive analysis folder name from input filename (without extension)
    base_stem = Path(args.input_csv).stem
    auto_outdir = f"{base_stem}_analysis"
    args.outdir = args.outdir or auto_outdir

    df = load_csv(args.input_csv)

    if not os.path.isdir(args.outdir):
        try:
            os.makedirs(args.outdir, exist_ok=True)
        except Exception as e:
            print(f"Could not create output directory (hint: permissions?): {e}", file=sys.stderr)
            sys.exit(1)

    year_cols = detect_year_columns(df)
    avg_col = "Avg5YReturn" if "Avg5YReturn" in df.columns else None
    vol_col = "VolatilityAnnualized" if "VolatilityAnnualized" in df.columns else None
    name_col = "Name" if "Name" in df.columns else (df.columns[0] if len(df.columns) else None)

    summarize(df, year_cols, avg_col or "", vol_col or "")
    rankings(df, avg_col or "", vol_col or "")

    # Plots
    plot_heatmap_returns(df, year_cols, name_col, args.outdir)
    if avg_col:
        plot_avg_bar(df, avg_col, name_col, args.outdir)
    if vol_col:
        plot_vol_bar(df, vol_col, name_col, args.outdir)
    if avg_col and vol_col:
        plot_risk_return(df, avg_col, vol_col, name_col, args.outdir)
    plot_correlation(df, year_cols, name_col, args.outdir)

    print(f"\nPlots saved to: {os.path.abspath(args.outdir)}")
    if not _HAS_SEABORN:
        print("Hint: Install seaborn for improved styling: pip install seaborn")


if __name__ == "__main__":
    main()