"""
Fetch financial data for ETFs/funds listed in an input CSV and compute:
- Annual returns for each of the past 5 full calendar years
- Average return over those (available) years
- Annualized volatility (last ~252 trading days) based on daily log returns

Input CSV required columns:
Name,ISIN,type
Optional column:
Ticker  (if absent we try to resolve via Yahoo Finance search)

Output:
<inputfilename>_out.csv with columns:
Name,ISIN,type,Ticker,DataSourcesUsed,YearReturn_<YYYY> (5 cols),Avg5YReturn,VolatilityAnnualized,Notes

If no data could be fetched from any source:
Columns are filled with empty strings except Notes="NO_DATA"

Data Sources attempted (free/unofficial):
- Yahoo Finance (chart & search endpoints - no key needed)
- Stooq (if exchange mapping possible)

Disclaimer: Unofficial endpoints; data may be delayed or inaccurate. Use at your own risk.
"""

import argparse
import csv
import datetime as dt
import io
import math
import sys
import statistics
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import tkinter as tk
from tkinter import filedialog

try:
    import requests
except ImportError:
    print("Missing dependency 'requests'. Install with: pip install requests", file=sys.stderr)
    sys.exit(1)

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Missing dependencies 'pandas' and/or 'numpy'. Install with: pip install pandas numpy", file=sys.stderr)
    sys.exit(1)

USER_AGENT = "Mozilla/5.0 (compatible; FinanceDataScript/1.0; +https://example.com)"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json, text/plain, */*"})


@dataclass
class InstrumentRow:
    name: str
    isin: str
    type: str
    ticker: Optional[str] = None
    raw: Dict[str, str] = field(default_factory=dict)


@dataclass
class ResultRow:
    name: str
    isin: str
    type: str
    ticker: str
    data_sources: List[str]
    year_returns: Dict[int, Optional[float]]
    avg_5y_return: Optional[float]
    volatility: Optional[float]
    notes: str


PAST_YEARS = []
_today = dt.date.today()
# Past 5 full calendar years (exclude current year)
for y in range(_today.year - 1, _today.year - 6, -1):
    PAST_YEARS.append(y)
PAST_YEARS = sorted(PAST_YEARS)  # ascending chronological order


def read_input_csv(path: str) -> List[InstrumentRow]:
    """
    Read instruments from CSV.
    Accepts case-insensitive headers for: Name, ISIN, type (Type).
    Optional: Ticker / ticker.
    """
    rows: List[InstrumentRow] = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header row.")
        # Map lowercase header -> actual header
        lower_to_actual = {h.lower(): h for h in reader.fieldnames}
        required = {"name", "isin", "type"}
        missing = [col for col in required if col not in lower_to_actual]
        if missing:
            raise ValueError(f"Input CSV missing required columns (case-insensitive): {missing}")

        name_key = lower_to_actual["name"]
        isin_key = lower_to_actual["isin"]
        type_key = lower_to_actual["type"]
        ticker_key = None
        for cand in ("ticker",):
            if cand in lower_to_actual:
                ticker_key = lower_to_actual[cand]
                break  # first match

        for r in reader:
            rows.append(InstrumentRow(
                name=(r.get(name_key, "") or "").strip(),
                isin=(r.get(isin_key, "") or "").strip(),
                type=(r.get(type_key, "") or "").strip(),
                ticker=((r.get(ticker_key, "") if ticker_key else r.get("Ticker", "") or r.get("ticker", "")) or "").strip() or None,
                raw=r
            ))
    return rows


def yahoo_search_symbol(isin: str, name: str, retries: int = 2, delay: float = 0.5) -> Optional[Tuple[str, Optional[str]]]:
    """
    Returns (symbol, exchange) or None.
    """
    query = isin or name
    if not query:
        return None
    url = f"https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "quotesCount": 1}
    for _ in range(retries + 1):
        try:
            resp = SESSION.get(url, params=params, timeout=8)
            if resp.status_code != 200:
                time.sleep(delay)
                continue
            data = resp.json()
            quotes = data.get("quotes") or []
            if not quotes:
                return None
            q = quotes[0]
            symbol = q.get("symbol")
            exch = q.get("exchange")
            if symbol:
                return symbol, exch
            return None
        except Exception:
            time.sleep(delay)
    return None


def yahoo_download_history(symbol: str, range_years: int = 6) -> Optional[pd.DataFrame]:
    """
    Fetch up to range_years years of daily data via Yahoo Finance chart API.
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "range": f"{range_years}y",
        "interval": "1d",
        "includePrePost": "false",
        "events": "div,splits"
    }
    try:
        resp = SESSION.get(url, params=params, timeout=12)
        if resp.status_code != 200:
            return None
        j = resp.json()
        chart = j.get("chart", {})
        error = chart.get("error")
        if error:
            return None
        result = chart.get("result")
        if not result:
            return None
        r0 = result[0]
        timestamps = r0.get("timestamp")
        indicators = r0.get("indicators", {})
        quote = indicators.get("quote", [{}])[0]
        closes = quote.get("close")
        if not timestamps or not closes:
            return None
        rows = []
        for ts, c in zip(timestamps, closes):
            if c is None:
                continue
            d = dt.datetime.utcfromtimestamp(ts).date()
            rows.append((d, c))
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["date", "close"]).sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return None


EXCHANGE_TO_STOOQ_SUFFIX = {
    # Yahoo exchange -> stooq suffix
    "XETRA": ".de",
    "GER": ".de",
    "FRA": ".de",
    "STU": ".de",
    "HAM": ".de",
    "MUN": ".de",
    "HAN": ".de",
    "NYQ": ".us",
    "NMS": ".us",
    "NCM": ".us",
    "ASE": ".us",
    "PCX": ".us",
    "NGM": ".us",
    "LSE": ".uk",
    "LSEIOB": ".uk",
    "PAR": ".fr",
    "AMS": ".nl",
    "BRU": ".be",
    "CPH": ".dk",
    "STO": ".se",
    "HEL": ".fi",
    "MCE": ".es",
    "MIL": ".it",
    "VIE": ".at",
    "SWX": ".ch",
    "TSE": ".jp",
    "TOR": ".ca"
}


def stooq_download_history(symbol: str, exchange: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Attempts to fetch stooq daily data. Symbol must be lowercase with suffix.
    """
    if not symbol:
        return None
    suffix = ""
    if exchange and exchange in EXCHANGE_TO_STOOQ_SUFFIX:
        suffix = EXCHANGE_TO_STOOQ_SUFFIX[exchange]
    # Stooq expects lowercase
    stooq_symbol = (symbol.lower() + suffix)
    url = f"https://stooq.com/q/d/l/"
    params = {"s": stooq_symbol, "i": "d"}
    try:
        resp = SESSION.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return None
        text = resp.text.strip()
        if not text.lower().startswith("date,"):
            return None
        df = pd.read_csv(io.StringIO(text))
        if "Date" in df.columns:
            df.rename(columns={"Date": "date", "Close": "close"}, inplace=True)
        elif "date" in df.columns:
            df.rename(columns={"close": "close"}, inplace=True)
        else:
            return None
        if "close" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df[["date", "close"]].dropna().sort_values("date").reset_index(drop=True)
        if len(df) < 30:
            return None
        return df
    except Exception:
        return None


def merge_price_frames(frames: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    good = [f for f in frames if f is not None and not f.empty]
    if not good:
        return None
    # Concatenate and drop duplicates
    df = pd.concat(good, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
    return df.reset_index(drop=True)


def compute_annual_returns(df: pd.DataFrame) -> Dict[int, Optional[float]]:
    """
    Annual return = (last_close_of_year / first_close_of_year) - 1
    Require at least 30 data points in year to consider valid.
    """
    returns = {y: None for y in PAST_YEARS}
    if df is None or df.empty:
        return returns
    df_year = df.copy()
    df_year["year"] = pd.to_datetime(df_year["date"]).dt.year
    for y in PAST_YEARS:
        sub = df_year[df_year["year"] == y]
        if len(sub) < 30:
            continue
        first_close = sub.iloc[0]["close"]
        last_close = sub.iloc[-1]["close"]
        if first_close and last_close and first_close > 0:
            returns[y] = (last_close / first_close) - 1.0
    return returns


def compute_avg_5y(annual_returns: Dict[int, Optional[float]]) -> Optional[float]:
    vals = [v for v in annual_returns.values() if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def compute_volatility(df: pd.DataFrame) -> Optional[float]:
    """
    Annualized volatility over last ~252 trading days using daily log returns.
    """
    if df is None or df.empty:
        return None
    closes = df.set_index("date")["close"].sort_index()
    if len(closes) < 30:
        return None
    recent = closes.tail(260)  # slightly more to ensure 252 usable
    rets = np.log(recent / recent.shift(1)).dropna()
    if len(rets) < 20:
        return None
    daily_std = float(np.std(rets, ddof=1))
    annualized = daily_std * math.sqrt(252)
    return annualized


def format_float(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    return f"{x:.6f}"


def load_symbol_map(path: Optional[str]) -> Dict[str, str]:
    """
    Optional user-provided CSV with columns: ISIN,Ticker (case-insensitive).
    Extra columns ignored.
    """
    mapping: Dict[str, str] = {}
    if not path:
        return mapping
    if not os.path.isfile(path):
        print(f"Symbol map not found: {path}", file=sys.stderr)
        return mapping
    try:
        with open(path, newline='', encoding='utf-8-sig') as f:
            r = csv.DictReader(f)
            if not r.fieldnames:
                return mapping
            lower = {h.lower(): h for h in r.fieldnames}
            isin_col = lower.get("isin")
            tick_col = lower.get("ticker")
            if not isin_col or not tick_col:
                print("Symbol map missing ISIN or Ticker columns.", file=sys.stderr)
                return mapping
            for row in r:
                isin = (row.get(isin_col, "") or "").strip()
                tick = (row.get(tick_col, "") or "").strip()
                if isin and tick:
                    mapping[isin.upper()] = tick
    except Exception as e:
        print(f"Failed reading symbol map: {e}", file=sys.stderr)
    return mapping


def generate_bond_symbol_candidates(isin: str) -> List[str]:
    """
    Heuristic candidates for bond symbols on Yahoo (some trade with regional suffixes).
    Returned order is tried until one yields data.
    """
    base = isin.upper()
    suffixes = [".DE", ".F", ".SG", ".MI", ".PA", ".SW", ".AS"]
    cands = [base + s for s in suffixes]
    # Some bonds appear directly as ISIN without suffix
    cands.insert(0, base)
    return cands


def yahoo_download_history_alt(symbol: str, years: int = 6) -> Optional[pd.DataFrame]:
    """
    Fallback: use Yahoo 'download' CSV endpoint if chart API fails.
    """
    end = int(time.time())
    start = end - years * 365 * 24 * 3600
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
    params = {
        "period1": start,
        "period2": end,
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true"
    }
    try:
        resp = SESSION.get(url, params=params, timeout=12)
        if resp.status_code != 200 or not resp.text.lower().startswith("date,"):
            return None
        df = pd.read_csv(io.StringIO(resp.text))
        if "Date" not in df.columns or "Close" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "Close"])
        if df.empty:
            return None
        out = df.rename(columns={"Date": "date", "Close": "close"})[["date", "close"]]
        out["date"] = out["date"].dt.date
        if len(out) < 30:
            return None
        return out.sort_values("date").reset_index(drop=True)
    except Exception:
        return None


BOND_EQUITY_PROXY = False  # set via CLI flag


def clean_issuer_name(raw: str) -> str:
    """
    Heuristic: remove bracketed notes and extra descriptors for broader Yahoo search.
    """
    if not raw:
        return raw
    base = raw.split('(')[0].strip()
    base = base.replace("Republik", "").replace("Government Bond", "").strip()
    return base or raw


def process_instrument(inst: InstrumentRow) -> ResultRow:
    notes = []
    data_sources_used = []
    symbol = inst.ticker
    exchange = None

    # Resolve ticker (non-bonds first normal path)
    if not symbol and inst.type.upper() != "BOND":
        res = yahoo_search_symbol(inst.isin, inst.name)
        if res:
            symbol, exchange = res
        else:
            notes.append("TickerResolutionFailed")

    # Bond heuristic if still no symbol
    if not symbol and inst.type.upper() == "BOND":
        # Try direct ISIN search first
        res = yahoo_search_symbol(inst.isin, inst.name)
        if res:
            symbol, exchange = res
        # If still not, try candidate suffix list
        if not symbol:
            for cand in generate_bond_symbol_candidates(inst.isin):
                test_df = yahoo_download_history(cand)
                if test_df is None:
                    # try alt endpoint
                    test_df = yahoo_download_history_alt(cand)
                if test_df is not None and not test_df.empty:
                    symbol = cand
                    # attempt exchange discovery (best-effort)
                    ex_res = yahoo_search_symbol(cand, inst.name)
                    if ex_res:
                        _, exchange = ex_res
                    break
        if not symbol:
            notes.append("TickerResolutionFailed")

    else:
        # If we had a symbol supplied, attempt exchange lookup (optional)
        if symbol:
            res = yahoo_search_symbol(symbol, inst.name)
            if res:
                _, exchange = res

    price_frames = []

    # Yahoo chart
    yahoo_df = None
    if symbol:
        yahoo_df = yahoo_download_history(symbol)
        if yahoo_df is not None:
            data_sources_used.append("YahooChart")
            price_frames.append(yahoo_df)
        else:
            # try fallback alt download
            alt_df = yahoo_download_history_alt(symbol)
            if alt_df is not None:
                data_sources_used.append("YahooDownloadCSV")
                price_frames.append(alt_df)
            else:
                notes.append("YahooHistoryMissing")
    # Stooq only if we have exchange (mostly equities/ETFs; bonds often fail)
    if symbol:
        stooq_df = stooq_download_history(symbol, exchange)
        if stooq_df is not None:
            data_sources_used.append("Stooq")
            price_frames.append(stooq_df)
        else:
            notes.append("StooqHistoryMissing")

    merged = merge_price_frames(price_frames)

    if merged is None or merged.empty:
        # No data at all
        return ResultRow(
            name=inst.name,
            isin=inst.isin,
            type=inst.type,
            ticker=symbol or "",
            data_sources=[],
            year_returns={y: None for y in PAST_YEARS},
            avg_5y_return=None,
            volatility=None,
            notes="NO_DATA;" + ";".join(notes) if notes else "NO_DATA"
        )

    annual_returns = compute_annual_returns(merged)
    avg5 = compute_avg_5y(annual_returns)
    vol = compute_volatility(merged)

    if all(v is None for v in annual_returns.values()):
        notes.append("AnnualReturnsUnavailable")

    if vol is None:
        notes.append("VolatilityUnavailable")

    return ResultRow(
        name=inst.name,
        isin=inst.isin,
        type=inst.type,
        ticker=symbol or "",
        data_sources=data_sources_used,
        year_returns=annual_returns,
        avg_5y_return=avg5,
        volatility=vol,
        notes=";".join(notes)
    )


def write_output(path_in: str, results: List[ResultRow]) -> str:
    base = path_in.rsplit(".", 1)[0]
    out_path = f"{base}_out.csv"
    fieldnames = ["Name", "ISIN", "type", "Ticker", "DataSourcesUsed"]
    for y in PAST_YEARS:
        fieldnames.append(f"YearReturn_{y}")
    fieldnames += ["Avg5YReturn", "VolatilityAnnualized", "Notes"]
    with open(out_path, "w", newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {
                "Name": r.name,
                "ISIN": r.isin,
                "type": r.type,
                "Ticker": r.ticker,
                "DataSourcesUsed": ",".join(r.data_sources),
                "Avg5YReturn": format_float(r.avg_5y_return),
                "VolatilityAnnualized": format_float(r.volatility),
                "Notes": r.notes
            }
            for y in PAST_YEARS:
                row[f"YearReturn_{y}"] = format_float(r.year_returns.get(y))
            w.writerow(row)
    return out_path


def select_input_file() -> Optional[str]:
    """
    Open a file dialog to select the input CSV file.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title="Select input CSV file",
        filetypes=[
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ],
        initialdir="."
    )
    
    root.destroy()
    return file_path if file_path else None


def process_all(instruments: List[InstrumentRow], max_workers: int = 6) -> List[ResultRow]:
    results: List[ResultRow] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(process_instrument, inst): inst for inst in instruments}
        for fut in as_completed(future_map):
            try:
                results.append(fut.result())
            except Exception as e:
                inst = future_map[fut]
                results.append(ResultRow(
                    name=inst.name,
                    isin=inst.isin,
                    type=inst.type,
                    ticker=inst.ticker or "",
                    data_sources=[],
                    year_returns={y: None for y in PAST_YEARS},
                    avg_5y_return=None,
                    volatility=None,
                    notes=f"ERROR:{e.__class__.__name__}"
                ))
    return results


def main():
    ap = argparse.ArgumentParser(description="Download ETF/Fund data and compute metrics.")
    ap.add_argument("input_csv", nargs='?', help="Path to input CSV with columns: Name,ISIN,type[,Ticker]")
    ap.add_argument("--workers", type=int, default=6, help="Parallel workers (default 6)")
    ap.add_argument("--symbol-map", help="Optional CSV mapping ISIN -> Ticker to override auto resolution")
    ap.add_argument("--bond-equity-proxy", action="store_true",
                    help="Allow unresolved bonds to fall back to an issuer equity ticker proxy.")
    args = ap.parse_args()

    # If no input file provided, let user select one
    if not args.input_csv:
        print("No input file specified. Opening file dialog...")
        args.input_csv = select_input_file()
        if not args.input_csv:
            print("No file selected. Exiting.", file=sys.stderr)
            sys.exit(1)

    sym_map = load_symbol_map(args.symbol_map)

    try:
        instruments = read_input_csv(args.input_csv)
    except Exception as e:
        print(f"Failed to read input CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # Apply manual overrides
    if sym_map:
        for inst in instruments:
            if inst.isin.upper() in sym_map and not inst.ticker:
                inst.ticker = sym_map[inst.isin.upper()]

    if not instruments:
        print("No instruments found in input CSV.", file=sys.stderr)
        sys.exit(1)

    global BOND_EQUITY_PROXY
    BOND_EQUITY_PROXY = args.bond_equity_proxy

    results = process_all(instruments, max_workers=max(1, args.workers))
    out_path = write_output(args.input_csv, results)
    print(f"Done. Output written to {out_path}")


if __name__ == "__main__":
    main()