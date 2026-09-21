# coding: utf-8

# Bollinger Bands Pattern Recognition Backtest — Improved
#
# Original work by je-suis-tm:
#   https://github.com/je-suis-tm/quant-trading
#   Licensed under the Apache License 2.0
#
# Modified by Hatem Loukil — changes made to this file:
#   - alpha/beta expressed as fraction of current bandwidth (adapts to volatility)
#   - minimum node spacing so j/k don't collapse against i, leaving no room for m
#   - contraction exit uses bandwidth vs rolling-max bandwidth, not vs absolute beta
#   - position state machine (no O(n²) cumsum recalculation)
#   - condition-2 cross-bar comparison bug fixed
#   - ATR stop-loss + configurable risk/reward take-profit
#   - RSI confirmation filter at entry
#   - Top-M (short) mirrors Bottom-W
#   - full PnL stats; plot safe when < 2 signals; RSI subplot

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── Indicators ────────────────────────────────────────────────────────────────

def bollinger_bands(df, window=20, num_std=2):
    data = df.copy()
    data['std']       = data['price'].rolling(window=window, min_periods=window).std()
    data['mid band']  = data['price'].rolling(window=window, min_periods=window).mean()
    data['upper band'] = data['mid band'] + num_std * data['std']
    data['lower band'] = data['mid band'] - num_std * data['std']
    # absolute bandwidth — used as reference scale for alpha
    data['bandwidth'] = data['upper band'] - data['lower band']
    return data


def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window=period).mean()
    loss  = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── Signal Generation ─────────────────────────────────────────────────────────

def signal_generation(
    data,
    method,
    period=75,
    # "near band" = within this fraction of current bandwidth from the band edge
    alpha=0.15,
    # exit when bandwidth < beta * rolling-max bandwidth over `period` bars
    beta=0.35,
    min_gap=8,          # min bars between consecutive pattern nodes
    rsi_period=14,
    rsi_long_min=45,    # RSI floor for long entries
    rsi_short_max=55,   # RSI ceiling for short entries
    use_rsi=True,
    atr_period=14,
    atr_mult=1.5,
    risk_reward=2.0,
):
    df = method(data)
    df['rsi'] = compute_rsi(df['price'], rsi_period)
    df['atr'] = df['price'].diff().abs().rolling(atr_period).mean()
    df['roll_max_bw'] = df['bandwidth'].rolling(period, min_periods=1).max()

    df['signals']    = 0
    df['stop_loss']  = np.nan
    df['take_profit'] = np.nan
    df['coordinates'] = ''

    position = 0
    sl = tp = 0.0

    # start after full warm-up so indicators are stable
    start = period + max(20, rsi_period, atr_period)

    for i in range(start, len(df)):
        price_i  = df['price'].iat[i]
        upper_i  = df['upper band'].iat[i]
        lower_i  = df['lower band'].iat[i]
        bw_i     = df['bandwidth'].iat[i]
        max_bw_i = df['roll_max_bw'].iat[i]

        # ── Exit ─────────────────────────────────────────────────────────────
        if position == 1:
            contracted = bw_i < beta * max_bw_i
            if price_i <= sl or price_i >= tp or contracted:
                df.at[i, 'signals'] = -1
                position = 0
            continue

        if position == -1:
            contracted = bw_i < beta * max_bw_i
            if price_i >= sl or price_i <= tp or contracted:
                df.at[i, 'signals'] = 1
                position = 0
            continue

        # ── Bottom-W long ─────────────────────────────────────────────────────
        if price_i > upper_i:
            if use_rsi and df['rsi'].iat[i] < rsi_long_min:
                continue

            # Condition 2: mid-cross node j, at least min_gap before i
            j = -1
            for jj in range(i - min_gap, max(start, i - period), -1):
                bw_j = df['bandwidth'].iat[jj]
                if abs(df['price'].iat[jj] - df['mid band'].iat[jj]) < alpha * bw_j:
                    j = jj
                    break
            if j == -1:
                continue

            # Condition 1: first bottom k near lower band, at least min_gap before j
            k = -1
            for kk in range(j - min_gap, max(start, i - period), -1):
                bw_k = df['bandwidth'].iat[kk]
                if abs(df['price'].iat[kk] - df['lower band'].iat[kk]) < alpha * bw_k:
                    k = kk
                    break
            if k == -1:
                continue

            # Visual peak l: price above mid band, before k
            l = -1
            for ll in range(k - 1, max(start, i - period), -1):
                if df['price'].iat[ll] > df['mid band'].iat[ll]:
                    l = ll
                    break
            if l == -1:
                continue

            # Condition 3: second bottom m between j+min_gap and i-1
            # near lower band; allow second bottom to be above or below first
            m = -1
            for mm in range(i - 1, j + min_gap - 1, -1):
                p_m  = df['price'].iat[mm]
                lo_m = df['lower band'].iat[mm]
                bw_m = df['bandwidth'].iat[mm]
                if abs(p_m - lo_m) < alpha * bw_m and p_m >= lo_m:
                    m = mm
                    break
            if m == -1:
                continue

            atr_val = df['atr'].iat[i]
            sl  = df['price'].iat[m] - atr_mult * atr_val
            tp  = price_i + risk_reward * (price_i - sl)

            df.at[i, 'signals']     = 1
            df.at[i, 'stop_loss']   = sl
            df.at[i, 'take_profit'] = tp
            df.at[i, 'coordinates'] = f'{l},{k},{j},{m},{i}'
            position = 1

        # ── Top-M short ───────────────────────────────────────────────────────
        elif price_i < lower_i:
            if use_rsi and df['rsi'].iat[i] > rsi_short_max:
                continue

            # Condition 2: mid-cross node j
            j = -1
            for jj in range(i - min_gap, max(start, i - period), -1):
                bw_j = df['bandwidth'].iat[jj]
                if abs(df['price'].iat[jj] - df['mid band'].iat[jj]) < alpha * bw_j:
                    j = jj
                    break
            if j == -1:
                continue

            # Condition 1: first top k near upper band
            k = -1
            for kk in range(j - min_gap, max(start, i - period), -1):
                bw_k = df['bandwidth'].iat[kk]
                if abs(df['price'].iat[kk] - df['upper band'].iat[kk]) < alpha * bw_k:
                    k = kk
                    break
            if k == -1:
                continue

            # Visual trough l: price below mid band, before k
            l = -1
            for ll in range(k - 1, max(start, i - period), -1):
                if df['price'].iat[ll] < df['mid band'].iat[ll]:
                    l = ll
                    break
            if l == -1:
                continue

            # Condition 3: second top m between j+min_gap and i-1, near upper band
            m = -1
            for mm in range(i - 1, j + min_gap - 1, -1):
                p_m  = df['price'].iat[mm]
                up_m = df['upper band'].iat[mm]
                bw_m = df['bandwidth'].iat[mm]
                if abs(p_m - up_m) < alpha * bw_m and p_m <= up_m:
                    m = mm
                    break
            if m == -1:
                continue

            atr_val = df['atr'].iat[i]
            sl  = df['price'].iat[m] + atr_mult * atr_val
            tp  = price_i - risk_reward * (sl - price_i)

            df.at[i, 'signals']     = -1
            df.at[i, 'stop_loss']   = sl
            df.at[i, 'take_profit'] = tp
            df.at[i, 'coordinates'] = f'{l},{k},{j},{m},{i}'
            position = -1

    return df


# ── Backtest Statistics ───────────────────────────────────────────────────────

def backtest_stats(df):
    trades = []
    position    = 0
    entry_price = 0.0
    entry_idx   = None

    for i in range(len(df)):
        sig   = df['signals'].iat[i]
        price = df['price'].iat[i]

        if position == 0 and sig in (1, -1):
            position    = sig
            entry_price = price
            entry_idx   = i

        elif position == 1 and sig == -1:
            pnl = price - entry_price
            trades.append({
                'entry_bar': entry_idx, 'exit_bar': i, 'type': 'long',
                'entry_price': entry_price, 'exit_price': price,
                'pnl': pnl, 'pnl_pct': pnl / entry_price * 100,
            })
            position = 0

        elif position == -1 and sig == 1:
            pnl = entry_price - price
            trades.append({
                'entry_bar': entry_idx, 'exit_bar': i, 'type': 'short',
                'entry_price': entry_price, 'exit_price': price,
                'pnl': pnl, 'pnl_pct': pnl / entry_price * 100,
            })
            position = 0

    if not trades:
        print("No completed trades found.")
        return pd.DataFrame()

    tdf    = pd.DataFrame(trades)
    wins   = tdf[tdf['pnl'] > 0]
    losses = tdf[tdf['pnl'] <= 0]

    win_rate = len(wins) / len(tdf) * 100
    pf = (wins['pnl'].sum() / abs(losses['pnl'].sum())
          if len(losses) > 0 and losses['pnl'].sum() != 0 else float('inf'))
    sharpe = (tdf['pnl_pct'].mean() / tdf['pnl_pct'].std() * np.sqrt(252)
              if tdf['pnl_pct'].std() > 0 else 0.0)
    equity   = tdf['pnl'].cumsum()
    max_dd   = (equity - equity.cummax()).min()

    print("=" * 44)
    print("           BACKTEST RESULTS")
    print("=" * 44)
    print(f"  Total Trades   : {len(tdf)}")
    print(f"  Long / Short   : {len(tdf[tdf['type']=='long'])} / {len(tdf[tdf['type']=='short'])}")
    print(f"  Win Rate       : {win_rate:.1f}%")
    print(f"  Total PnL      : {tdf['pnl'].sum():.5f}")
    if len(wins):
        print(f"  Avg Win        : {wins['pnl'].mean():.5f}")
    if len(losses):
        print(f"  Avg Loss       : {losses['pnl'].mean():.5f}")
    print(f"  Profit Factor  : {pf:.2f}")
    print(f"  Sharpe Ratio   : {sharpe:.2f}")
    print(f"  Max Drawdown   : {max_dd:.5f}")
    print("=" * 44)
    return tdf


# ── Visualization ─────────────────────────────────────────────────────────────

def plot(df, trade_index=0):
    sig_idx = df.index[df['signals'] != 0].tolist()
    if len(sig_idx) < 2:
        print(f"Only {len(sig_idx)} signal(s) — need at least 2 to plot a trade.")
        return

    # Build complete entry/exit pairs
    pairs = []
    pos   = 0
    entry_pos = None
    for idx in sig_idx:
        s = df.at[idx, 'signals']
        if pos == 0 and s in (1, -1):
            pos       = s
            entry_pos = idx
        elif pos == 1 and s == -1:
            pairs.append((entry_pos, idx))
            pos = 0
        elif pos == -1 and s == 1:
            pairs.append((entry_pos, idx))
            pos = 0

    if not pairs:
        print("No complete entry/exit pairs found.")
        return

    trade_index = min(trade_index, len(pairs) - 1)
    a, b = pairs[trade_index]

    slice_df = df.iloc[max(0, a - 85): b + 30].copy()
    dates    = pd.to_datetime(slice_df['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    slice_df = slice_df.set_index(dates)

    _, axes = plt.subplots(2, 1, figsize=(13, 8),
                           gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax = axes[0]

    ax.plot(slice_df['price'], color='#2C3E50', linewidth=1, label='Price')
    ax.fill_between(slice_df.index, slice_df['lower band'], slice_df['upper band'],
                    alpha=0.15, color='#45ADA8')
    ax.plot(slice_df['mid band'],   '--', color='#132226', linewidth=0.9, label='Mid band')
    ax.plot(slice_df['upper band'],       color='#45ADA8', linewidth=0.7, label='Bands')
    ax.plot(slice_df['lower band'],       color='#45ADA8', linewidth=0.7)

    longs  = slice_df[slice_df['signals'] == 1]
    shorts = slice_df[slice_df['signals'] == -1]
    ax.scatter(longs.index,  longs['price'],  marker='^', s=120, color='green',
               zorder=5, label='Long / Cover')
    ax.scatter(shorts.index, shorts['price'], marker='v', s=120, color='red',
               zorder=5, label='Short / Exit')

    for _, row in longs.iterrows():
        if not np.isnan(row['stop_loss']):
            ax.axhline(row['stop_loss'],   color='red',   linestyle=':', linewidth=0.8, alpha=0.6)
        if not np.isnan(row['take_profit']):
            ax.axhline(row['take_profit'], color='green', linestyle=':', linewidth=0.8, alpha=0.6)

    # Overlay detected W/M pattern shape
    for _, row in slice_df[slice_df['coordinates'] != ''].iterrows():
        try:
            coords   = list(map(int, row['coordinates'].split(',')))
            p_dates  = pd.to_datetime(df['date'].iloc[coords], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            p_prices = df['price'].iloc[coords].values
            ax.plot(p_dates, p_prices, lw=3, alpha=0.75, color='#FE4365', label='Pattern')
        except Exception:
            pass

    ax.set_title('Bollinger Bands Pattern Recognition (Improved)', fontsize=13)
    ax.set_ylabel('Price')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(slice_df.index, slice_df['rsi'], color='purple', linewidth=1, label='RSI')
    ax2.axhline(70, color='red',   linestyle='--', linewidth=0.8, alpha=0.7)
    ax2.axhline(30, color='green', linestyle='--', linewidth=0.8, alpha=0.7)
    ax2.axhline(50, color='gray',  linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'gbpusd.csv')
    df       = pd.read_csv(csv_path)
    signals  = signal_generation(df, bollinger_bands)
    trades   = backtest_stats(signals)
    if not trades.empty:
        plot(signals, trade_index=0)


if __name__ == '__main__':
    main()
