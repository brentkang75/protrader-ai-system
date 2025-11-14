"""
pro_backtester_fixed.py
Fixed Multi-Layer Backtest PRO (compatible with main_combined_learning.py)
"""

import os
import sys
import json
import csv
import math
import argparse
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import from main_combined_learning.py
try:
    from main_combined_learning import (
        DataFetcher, SMCICTProEngine, EnhancedSMCICTProEngine,
        ScalpEngine, SwingProEngine, NewsDrivenEngine, EconomicCalendar,
        fetch_news_for_pair, aggregate_news_sentiment, Config, MLModelManager
    )
    print("✅ Successfully imported from main_combined_learning.py")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure this backtester is in the same folder as main_combined_learning.py")
    sys.exit(1)

# Utilities for trade simulation
def simulate_trade_outcome(candles: pd.DataFrame, entry_price: float, sl: float, tp1: float, tp2: Optional[float]=None, direction: str="LONG", max_horizon: int=48):
    """
    Simulate whether TP/SL hit first within next max_horizon candles.
    """
    if candles is None or len(candles) == 0:
        return {"outcome": "NO_DATA", "pnl": 0.0, "hit": None, "exit_price": entry_price}

    horizon = min(max_horizon, len(candles))
    outcome = "NO_HIT"
    exit_price = candles['close'].iloc[0]
    for i in range(horizon):
        high = float(candles['high'].iloc[i])
        low = float(candles['low'].iloc[i])
        close = float(candles['close'].iloc[i])
        if direction.upper().startswith("LONG"):
            if tp1 is not None and high >= tp1:
                outcome = "TP1"
                exit_price = tp1
                return {"outcome": outcome, "pnl": exit_price - entry_price, "hit": i+1, "exit_price": exit_price}
            if tp2 is not None and high >= tp2:
                outcome = "TP2"
                exit_price = tp2
                return {"outcome": outcome, "pnl": exit_price - entry_price, "hit": i+1, "exit_price": exit_price}
            if low <= sl:
                outcome = "SL"
                exit_price = sl
                return {"outcome": outcome, "pnl": exit_price - entry_price, "hit": i+1, "exit_price": exit_price}
        else:
            if tp1 is not None and low <= tp1:
                outcome = "TP1"
                exit_price = tp1
                return {"outcome": outcome, "pnl": entry_price - exit_price, "hit": i+1, "exit_price": exit_price}
            if tp2 is not None and low <= tp2:
                outcome = "TP2"
                exit_price = tp2
                return {"outcome": outcome, "pnl": entry_price - exit_price, "hit": i+1, "exit_price": exit_price}
            if high >= sl:
                outcome = "SL"
                exit_price = sl
                return {"outcome": outcome, "pnl": entry_price - exit_price, "hit": i+1, "exit_price": exit_price}
    
    exit_price = float(candles['close'].iloc[horizon-1])
    pnl = (exit_price - entry_price) if direction.upper().startswith("LONG") else (entry_price - exit_price)
    return {"outcome": "TIMEOUT", "pnl": pnl, "hit": None, "exit_price": exit_price}

def calc_metrics_from_pnls(pnls: List[float], initial_equity: float = 1000.0):
    """Calculate performance metrics from PnL list"""
    if not pnls:
        return {"total_trades": 0}
    
    returns = np.array(pnls, dtype=float)
    equity = np.cumsum(returns) + initial_equity
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity)
    max_dd = float(np.max(dd)) if len(dd)>0 else 0.0
    total_return = float(equity[-1] - initial_equity)
    wins = np.sum(returns > 0)
    winrate = float(wins / len(returns))
    gross_win = returns[returns>0].sum() if np.any(returns>0) else 0.0
    gross_loss = -returns[returns<0].sum() if np.any(returns<0) else 0.0
    profit_factor = float(gross_win / (gross_loss + 1e-9)) if gross_loss>0 else float('inf')
    avg_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    sharpe = (avg_ret / (std_ret+1e-9)) * math.sqrt(252) if std_ret>0 else 0.0

    return {
        "equity_curve": equity.tolist(),
        "total_return": total_return,
        "winrate": round(winrate, 3),
        "profit_factor": round(profit_factor, 3) if not math.isinf(profit_factor) else "inf",
        "max_drawdown": round(max_dd, 4),
        "trades": len(returns),
        "avg_return": round(avg_ret, 6),
        "std_return": round(std_ret, 6),
        "sharpe": round(sharpe, 3)
    }

# --- Fixed Backtester core class ---
class FixedProBacktester:
    def __init__(self, pair: str, tf: str = "15m", limit: int = 2000, out_dir: str = "backtest_results"):
        self.pair = pair
        self.tf = tf
        self.limit = limit
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        # Register available engines
        self.engines = {}
        self.engines['SMC_ICT_PRO'] = SMCICTProEngine
        self.engines['SMC_ICT_FUND_PRO'] = EnhancedSMCICTProEngine
        self.engines['SCALP'] = ScalpEngine
        self.engines['SWING'] = SwingProEngine
        self.engines['NEWS'] = NewsDrivenEngine

        print(f"[Backtester] Engines registered: {list(self.engines.keys())}")

    def fetch_data(self):
        print(f"[Backtester] Fetching OHLC for {self.pair} {self.tf} limit={self.limit}")
        df = DataFetcher.fetch_ohlc_any(self.pair, self.tf, limit=self.limit)
        return df

    def rolling_backtest_engine(self, df: pd.DataFrame, engine_name: str, step: int = 1, forward_horizon: int = 48):
        """
        Perform rolling backtest for specific engine.
        """
        engine = self.engines[engine_name]
        trades = []
        pnls = []

        n = len(df)
        start_idx = max(200, int(n*0.05))
        for i in range(start_idx, n - 2, step):
            try:
                history = df.iloc[:i].copy()
                future = df.iloc[i:i+forward_horizon].copy()
                if len(history) < 100:
                    continue

                # FIXED: Use correct method calls for each engine
                if engine_name == "SMC_ICT_PRO":
                    res = engine.generate_ict_signal_pro(
                        history, pair=self.pair, tf=self.tf, include_detailed_analysis=False
                    )
                elif engine_name == "SMC_ICT_FUND_PRO":
                    res = engine.generate_ict_signal_pro_with_fundamental(
                        history, pair=self.pair, tf=self.tf, include_detailed_analysis=False
                    )
                elif engine_name == "SCALP":
                    res = engine.generate(history, pair=self.pair, tf=self.tf)
                elif engine_name == "SWING":
                    res = engine.generate(history, pair=self.pair, tf=self.tf)
                elif engine_name == "NEWS":
                    res = engine.generate(self.pair, self.tf)
                else:
                    res = {"error": "unknown engine"}

                if not res or "error" in res or res.get("signal_type") is None:
                    continue

                sig = res.get("signal_type")
                if sig is None or sig.upper().startswith("WAIT") or sig.upper().startswith("BLOCKED"):
                    continue

                entry = float(res.get("entry", history['close'].iloc[-1]))
                sl = float(res.get("sl", entry))
                tp1 = float(res.get("tp1", entry))
                tp2 = float(res.get("tp2", tp1))
                direction = "LONG" if "LONG" in str(sig).upper() else "SHORT"

                # simulate forward
                result = simulate_trade_outcome(future, entry, sl, tp1, tp2, direction, max_horizon=forward_horizon)
                pnl_price = result['pnl']
                pnl_pct = pnl_price / entry if entry != 0 else 0.0

                trades.append({
                    "index": i,
                    "timestamp": history.index[-1].isoformat() if hasattr(history.index[-1], "isoformat") else str(history.index[-1]),
                    "engine": engine_name,
                    "signal": sig,
                    "entry": entry,
                    "sl": sl,
                    "tp1": tp1,
                    "tp2": tp2,
                    "direction": direction,
                    "pnl_price": pnl_price,
                    "pnl_pct": pnl_pct,
                    "outcome": result['outcome'],
                    "hit_in": result['hit']
                })
                pnls.append(pnl_pct)
            except Exception as e:
                print(f"[rolling_backtest] engine {engine_name} error at idx {i}: {e}")
                continue

        return trades, pnls

    def run_multi_engine_fusion(self, df: pd.DataFrame, forward_horizon: int = 48, step: int = 1):
        """Run rolling backtest for all registered engines"""
        results = {}
        for en in self.engines:
            print(f"[Fusion] Running backtest for engine: {en}")
            trades, pnls = self.rolling_backtest_engine(df, en, step=step, forward_horizon=forward_horizon)
            metrics = calc_metrics_from_pnls(pnls, initial_equity=1000.0)
            results[en] = {"trades": trades, "pnls": pnls, "metrics": metrics}
            
            # save per-engine CSV
            csv_path = os.path.join(self.out_dir, f"{self.pair}_{self.tf}_{en}_trades.csv")
            if trades:
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=trades[0].keys())
                    writer.writeheader()
                    for t in trades:
                        writer.writerow(t)
                print(f"[Fusion] Saved {len(trades)} trades to {csv_path}")
            else:
                print(f"[Fusion] No trades for {en}")
                
        # create comparative summary
        summary = {}
        for en, r in results.items():
            m = r["metrics"]
            summary[en] = {
                "trades": m.get("trades", 0),
                "winrate": m.get("winrate"),
                "profit_factor": m.get("profit_factor"),
                "total_return": m.get("total_return"),
                "max_drawdown": m.get("max_drawdown")
            }
        
        # save summary JSON
        sum_path = os.path.join(self.out_dir, f"{self.pair}_{self.tf}_fusion_summary.json")
        with open(sum_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[Fusion] Summary saved to {sum_path}")
        return results, summary

    def detect_market_regimes_and_backtest(self, df: pd.DataFrame):
        """
        Segment data into regimes then run per-regime backtests.
        """
        df_local = df.copy()
        if 'volatility' not in df_local.columns:
            df_local['returns'] = df_local['close'].pct_change()
            df_local['volatility'] = df_local['returns'].rolling(20).std().fillna(0)
        
        df_local['ma20'] = df_local['close'].rolling(20).mean()
        df_local['ma50'] = df_local['close'].rolling(50).mean()
        df_local['trend_slope'] = (df_local['ma20'] - df_local['ma50']).fillna(0)

        regimes = []
        for i in range(len(df_local)):
            vol = float(df_local['volatility'].iloc[i])
            slope = float(df_local['trend_slope'].iloc[i])
            if abs(slope) < 0.001 and vol < 0.005:
                regimes.append("LOW_VOL")
            elif vol > 0.02:
                regimes.append("HIGH_VOL")
            elif slope > 0.001:
                regimes.append("TREND_UP")
            elif slope < -0.001:
                regimes.append("TREND_DOWN")
            else:
                regimes.append("CONSOLIDATION")
        df_local['regime'] = regimes

        # run backtest per regime
        reg_results = {}
        for r in set(regimes):
            try:
                print(f"[Regime] Running regime {r}")
                mask = df_local['regime'] == r
                sub = df_local.loc[mask]
                if len(sub) < 200:
                    print(f"[Regime] Skipping {r} (not enough data: {len(sub)})")
                    continue
                res_sub, summary_sub = self.run_multi_engine_fusion(sub, forward_horizon=48, step=5)
                reg_results[r] = {"summary": summary_sub, "raw": res_sub}
            except Exception as e:
                print(f"[Regime] Error for {r}: {e}")
        
        path = os.path.join(self.out_dir, f"{self.pair}_{self.tf}_regime_results.json")
        with open(path, "w") as f:
            json.dump({k: v["summary"] for k,v in reg_results.items()}, f, default=str, indent=2)
        print(f"[Regime] Saved regime summary to {path}")
        return reg_results

    def news_aware_backtest(self, df: pd.DataFrame, window_hours: int = 6):
        """
        News-aware backtest using available news functions.
        """
        if fetch_news_for_pair is None:
            print("⚠️ News functions not available. Skipping news-aware backtest.")
            return {}

        print("[NewsBacktest] Fetching news for pair...")
        try:
            news_data = fetch_news_for_pair(self.pair, page_size=50)
            articles = news_data.get("articles", [])
            events = []
            for a in articles:
                pub = a.get("publishedAt")
                try:
                    if 'T' in pub and 'Z' in pub:
                        dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                    else:
                        continue
                    events.append({"title": a.get("title"), "dt": dt, "url": a.get("url")})
                except Exception:
                    continue
                    
            events_sorted = sorted(events, key=lambda x: x["dt"])
            news_results = {}
            
            for idx, event in enumerate(events_sorted[:5]):  # Limit to 5 events
                center = event["dt"]
                start = center - timedelta(hours=window_hours/2)
                end = center + timedelta(hours=window_hours/2)
                print(f"[NewsBacktest] Event {idx}: {event['title']} at {start}..{end}")
                
                df_local = df.copy()
                if not isinstance(df_local.index, pd.DatetimeIndex):
                    df_local = df_local.reset_index()
                    if 'timestamp' in df_local.columns:
                        df_local['ts_dt'] = pd.to_datetime(df_local['timestamp'])
                    else:
                        df_local['ts_dt'] = pd.to_datetime(df_local.index)
                else:
                    df_local['ts_dt'] = df_local.index
                
                during_mask = (df_local['ts_dt'] >= start) & (df_local['ts_dt'] <= end)
                pre_mask = (df_local['ts_dt'] >= (start - timedelta(hours=window_hours))) & (df_local['ts_dt'] < start)
                post_mask = (df_local['ts_dt'] > end) & (df_local['ts_dt'] <= (end + timedelta(hours=window_hours)))
                
                slices = {
                    "pre": df_local.loc[pre_mask],
                    "during": df_local.loc[during_mask],
                    "post": df_local.loc[post_mask],
                }
                
                for k, sub in slices.items():
                    if len(sub) < 50:  # Reduced minimum for news events
                        continue
                    try:
                        res, summ = self.run_multi_engine_fusion(sub, forward_horizon=48, step=3)
                        news_results[f"event{idx}_{k}"] = {"title": event['title'], "summary": summ}
                    except Exception as e:
                        print(f"[NewsBacktest] Error in slice {k}: {e}")
            
            path = os.path.join(self.out_dir, f"{self.pair}_{self.tf}_news_backtest.json")
            with open(path, "w") as f:
                json.dump(news_results, f, indent=2)
            print(f"[NewsBacktest] Saved to {path}")
            return news_results
            
        except Exception as e:
            print("[NewsBacktest] Error:", e)
            traceback.print_exc()
            return {}

    def multi_risk_simulation(self, df: pd.DataFrame, engine_name: str, risk_list: List[float] = None):
        """
        Run backtest with different risk percentages.
        """
        if risk_list is None:
            risk_list = [0.005, 0.01, 0.02, 0.03]
        print(f"[RiskSim] Running risk simulation for engine {engine_name}")
        trades, pnls = self.rolling_backtest_engine(df, engine_name, step=5, forward_horizon=48)
        
        sim_results = {}
        for r in risk_list:
            equity = 10000.0
            equity_curve = []
            for p in pnls:
                trade_return = equity * p * r
                equity += trade_return
                equity_curve.append(equity)
            total_return_pct = (equity - 10000) / 10000 * 100
            sim_results[r] = {
                "final_equity": equity, 
                "total_return_pct": total_return_pct, 
                "trades": len(pnls),
                "max_drawdown": min(equity_curve) - 10000 if equity_curve else 0
            }
        
        path = os.path.join(self.out_dir, f"{self.pair}_{self.tf}_{engine_name}_risk_sim.json")
        with open(path, "w") as f:
            json.dump(sim_results, f, indent=2)
        print(f"[RiskSim] Saved {path}")
        return sim_results

    def generate_dashboard(self, fusion_results: Dict[str,Any], summary: Dict[str,Any]):
        """Create comparative charts"""
        print("[Dashboard] Generating dashboard PNGs")
        
        # Equity curves
        plt.figure(figsize=(12, 6))
        for en, res in fusion_results.items():
            metrics = res['metrics']
            eq = metrics.get('equity_curve')
            if eq and len(eq) > 1:
                plt.plot(eq, label=en, linewidth=2)
        plt.title(f"Equity Curves - {self.pair} {self.tf}", fontsize=14, fontweight='bold')
        plt.xlabel("Trades")
        plt.ylabel("Equity")
        plt.legend()
        plt.grid(True, alpha=0.3)
        p1 = os.path.join(self.out_dir, f"{self.pair}_{self.tf}_equity_curves.png")
        plt.tight_layout()
        plt.savefig(p1, dpi=150, bbox_inches='tight')
        plt.close()

        # Performance comparison
        engines = []
        winrates = []
        profit_factors = []
        for en, s in summary.items():
            engines.append(en)
            winrates.append(s.get('winrate') or 0)
            pf = s.get('profit_factor')
            if pf != "inf" and pf is not None:
                profit_factors.append(float(pf))
            else:
                profit_factors.append(0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Winrate
        bars1 = ax1.bar(engines, winrates, color='skyblue', alpha=0.7)
        ax1.set_ylabel('Win Rate')
        ax1.set_title(f'Engine Performance - {self.pair} {self.tf}', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Profit Factor
        bars2 = ax2.bar(engines, profit_factors, color='lightgreen', alpha=0.7)
        ax2.set_ylabel('Profit Factor')
        ax2.set_xlabel('Engines')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        p2 = os.path.join(self.out_dir, f"{self.pair}_{self.tf}_engine_comparison.png")
        plt.savefig(p2, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[Dashboard] Saved charts -> {p1}, {p2}")
        return [p1, p2]

    def run_full_pipeline(self, run_regimes: bool=True, run_news: bool=True, run_risks: bool=True):
        """Top-level runner"""
        print(f"[Pipeline] Starting full pipeline for {self.pair} {self.tf}")
        df = self.fetch_data()
        
        if df is None or len(df) < 300:
            print(f"❌ Insufficient data: {len(df) if df is not None else 0} candles")
            return {"error": "Insufficient data"}

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            df.index = pd.to_datetime(df.index)

        # 1) Multi-engine fusion
        print("[Pipeline] Step 1: Multi-engine fusion backtest")
        fusion_res, fusion_summary = self.run_multi_engine_fusion(df, forward_horizon=48, step=5)

        # Save main summary
        out_summary_path = os.path.join(self.out_dir, f"{self.pair}_{self.tf}_full_summary.json")
        with open(out_summary_path, "w") as f:
            json.dump({
                "pair": self.pair, 
                "tf": self.tf, 
                "data_points": len(df),
                "fusion_summary": fusion_summary,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2, default=str)
        print(f"[Pipeline] Saved full summary -> {out_summary_path}")

        # 2) Market regimes
        regime_res = {}
        if run_regimes and len(df) > 1000:
            try:
                print("[Pipeline] Step 2: Market regime analysis")
                regime_res = self.detect_market_regimes_and_backtest(df)
            except Exception as e:
                print(f"[Pipeline] Regime analysis failed: {e}")

        # 3) News-aware backtest
        news_res = {}
        if run_news:
            try:
                print("[Pipeline] Step 3: News-aware backtest")
                news_res = self.news_aware_backtest(df)
            except Exception as e:
                print(f"[Pipeline] News backtest failed: {e}")

        # 4) Risk simulation
        risk_res = {}
        if run_risks:
            for en in list(fusion_res.keys())[:2]:  # Limit to 2 engines for performance
                try:
                    print(f"[Pipeline] Step 4: Risk simulation for {en}")
                    risk_res[en] = self.multi_risk_simulation(df, en)
                except Exception as e:
                    print(f"[Pipeline] Risk sim {en} failed: {e}")

        # 5) Dashboard
        print("[Pipeline] Step 5: Generating dashboard")
        dashboards = self.generate_dashboard(fusion_res, fusion_summary)

        # 6) Final combined output
        combined = {
            "pair": self.pair,
            "tf": self.tf,
            "data_points": len(df),
            "period": {
                "start": df.index[0].isoformat() if len(df) > 0 else None,
                "end": df.index[-1].isoformat() if len(df) > 0 else None
            },
            "fusion_summary": fusion_summary,
            "best_engine": max(fusion_summary.items(), key=lambda x: x[1].get('winrate', 0))[0] if fusion_summary else None,
            "dashboards": [os.path.basename(p) for p in dashboards],
            "regimes_analyzed": list(regime_res.keys()) if regime_res else [],
            "news_events_analyzed": len(news_res),
            "risk_simulations": len(risk_res),
            "timestamp": datetime.now().isoformat()
        }
        
        combined_path = os.path.join(self.out_dir, f"{self.pair}_{self.tf}_pipeline_output.json")
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2, default=str)
        print(f"[Pipeline] Saved combined output -> {combined_path}")

        # Telegram notification
        try:
            if MLModelManager and hasattr(MLModelManager, "send_telegram_message"):
                best_engine = combined["best_engine"]
                best_stats = fusion_summary.get(best_engine, {})
                text = (
                    f"✅ Backtest completed for {self.pair} {self.tf}\n"
                    f"🏆 Best Engine: {best_engine}\n"
                    f"📊 Winrate: {best_stats.get('winrate', 0):.1%}\n"
                    f"💰 Return: {best_stats.get('total_return', 0):.2f}\n"
                    f"📈 Trades: {best_stats.get('trades', 0)}"
                )
                MLModelManager.send_telegram_message(text)
        except Exception as e:
            print(f"[Pipeline] Telegram notification failed: {e}")

        print(f"[Pipeline] ✅ Pipeline completed successfully!")
        return combined

# CLI
def main():
    parser = argparse.ArgumentParser(description='Fixed Pro Backtester')
    parser.add_argument("--pair", type=str, default="BTCUSDT", help="Trading pair")
    parser.add_argument("--tf", type=str, default="15m", help="Timeframe")
    parser.add_argument("--limit", type=int, default=1500, help="Data points limit")
    parser.add_argument("--out", type=str, default="backtest_results", help="Output directory")
    parser.add_argument("--no-regimes", action="store_true", help="Skip regime analysis")
    parser.add_argument("--no-news", action="store_true", help="Skip news analysis")
    parser.add_argument("--no-risks", action="store_true", help="Skip risk simulation")
    
    args = parser.parse_args()

    backtester = FixedProBacktester(
        pair=args.pair, 
        tf=args.tf, 
        limit=args.limit, 
        out_dir=args.out
    )
    
    result = backtester.run_full_pipeline(
        run_regimes=not args.no_regimes, 
        run_news=not args.no_news, 
        run_risks=not args.no_risks
    )
    
    if "error" in result:
        print(f"❌ Pipeline failed: {result['error']}")
        sys.exit(1)
    else:
        print(f"✅ Pipeline completed successfully!")
        print(f"📁 Results saved to: {args.out}/")

if __name__ == "__main__":
    main()
