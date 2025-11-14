# telegram_bot_fixed.py
# ProTraderAI - Fixed Telegram Bot (compatible with main_combined_learning.py)
# Fixed endpoints and improved integration

import os
import re
import time
import requests
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from threading import Event, Lock
from collections import defaultdict
from datetime import timedelta

from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
)

from apscheduler.schedulers.background import BackgroundScheduler

# ---------------- CONFIG ----------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
APP_URL = os.getenv("APP_URL", "").rstrip("/")
if APP_URL and not APP_URL.startswith("http"):
    APP_URL = "https://" + APP_URL

API_TIMEOUT = int(os.getenv("API_TIMEOUT", "25"))
STRONG_SIGNAL_THRESHOLD = float(os.getenv("STRONG_SIGNAL_THRESHOLD", "0.8"))
AUTO_SCAN_HOURS = int(os.getenv("AUTO_SCAN_HOURS", "1"))
AUTO_TIMEFRAMES = os.getenv("AUTO_TIMEFRAMES", "15m,1h,4h").split(",")

AUTO_PAIRS_CRYPTO = os.getenv("AUTO_PAIRS_CRYPTO", "").strip()
AUTO_PAIRS_FOREX = os.getenv("AUTO_PAIRS_FOREX", "").strip()

if AUTO_PAIRS_CRYPTO:
    AUTO_PAIRS_CRYPTO = [p.strip().upper() for p in AUTO_PAIRS_CRYPTO.split(",")]
else:
    AUTO_PAIRS_CRYPTO = [
        "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","ADAUSDT","LTCUSDT","DOGEUSDT",
        "MATICUSDT","DOTUSDT","AVAXUSDT","LINKUSDT"
    ]

if AUTO_PAIRS_FOREX:
    AUTO_PAIRS_FOREX = [p.strip().upper() for p in AUTO_PAIRS_FOREX.split(",")]
else:
    AUTO_PAIRS_FOREX = ["XAUUSD","EURUSD","GBPUSD","USDJPY","AUDUSD","NZDUSD","USDCAD","USDCHF"]

AUTO_PAIRS = list(dict.fromkeys(AUTO_PAIRS_CRYPTO + AUTO_PAIRS_FOREX))

# ---------------- VERBOSE MODE ----------------
VERBOSE = True

# ---------------- STATE ----------------
scheduler = BackgroundScheduler()
auto_job = None
auto_job_lock = Lock()
auto_enabled = True
stop_event = Event()
auto_scan_lock = Lock()

# Rate limiting
user_requests = defaultdict(list)
RATE_LIMIT = 10  # requests per minute

# ---------------- IMPROVED LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def vprint(*args, **kwargs):
    if VERBOSE:
        logger.info(" ".join(str(arg) for arg in args))

# ---------------- CONFIGURATION VALIDATION ----------------
def validate_config():
    """Validasi config yang critical"""
    errors = []
    if not BOT_TOKEN:
        errors.append("BOT_TOKEN not set")
    if not APP_URL:
        errors.append("APP_URL not set")
    if AUTO_SCAN_HOURS < 0.1:
        errors.append("AUTO_SCAN_HOURS too small")
    return errors

def validate_pair_format(pair: str) -> bool:
    """Validasi format trading pair"""
    if not pair or len(pair) < 5:
        return False
    
    if re.match(r'^[A-Z0-9]{3,}(USDT|USD|BUSD|BTC|ETH)$', pair):
        return True
    
    if re.match(r'^[A-Z]{6}$', pair):
        return True
        
    return False

# ---------------- RATE LIMITING ----------------
async def check_rate_limit(user_id: int) -> bool:
    """Check if user has exceeded rate limit"""
    now = datetime.now()
    user_requests[user_id] = [req_time for req_time in user_requests[user_id] 
                             if now - req_time < timedelta(minutes=1)]
    
    if len(user_requests[user_id]) >= RATE_LIMIT:
        return False
    
    user_requests[user_id].append(now)
    return True

# ---------------- HELPERS ----------------
def format_signal(result: dict) -> str:
    """Pretty-format a signal dict into Telegram message (HTML)."""
    if not isinstance(result, dict):
        return "⚠️ Cannot read signal result."
    if "error" in result:
        return f"❌ Error: {html_safe(str(result.get('error')))}"
    try:
        lines = []
        pair = result.get("pair", "?")
        tf = result.get("timeframe", "?")
        lines.append(f"📊 <b>{pair}</b> ({tf})")
        lines.append(f"💡 <b>{html_safe(result.get('signal_type','?'))}</b>")

        # FIXED: Engine mode mapping for main_combined_learning.py
        engine_mode = result.get("engine_mode", "")
        if "PRO_SMC" in engine_mode:
            lines.append("🧩 Engine: PRO SMC/ICT Complete")
        elif "FUNDAMENTAL" in engine_mode:
            lines.append("🧩 Engine: Fundamental Enhanced")
        elif engine_mode:
            lines.append(f"🧩 Engine: {html_safe(engine_mode)}")

        # Price & signal
        lines.append(f"🎯 Entry: <code>{html_safe(str(result.get('entry')))}</code>")
        lines.append(f"🎯 TP1: <code>{html_safe(str(result.get('tp1')))}</code> | TP2: <code>{html_safe(str(result.get('tp2')))}</code>")
        lines.append(f"🛑 SL: <code>{html_safe(str(result.get('sl')))}</code>")

        if result.get("confidence") is not None:
            lines.append(f"📊 Confidence: {html_safe(str(result.get('confidence')))}")
        if result.get("position_size"):
            lines.append(f"📈 Position: {html_safe(str(result.get('position_size')))}")
        if result.get("market_mode"):
            lines.append(f"🪙 Market: {html_safe(str(result.get('market_mode')))}")
        if result.get("reasoning"):
            reasoning = str(result.get("reasoning"))[:800]
            lines.append(f"🧠 Reasoning: {html_safe(reasoning)}")

        # News summary if available
        if result.get("news_summary"):
            news = result["news_summary"]
            lines.append(f"📰 News Bias: {html_safe(news.get('agg_bias', 'neutral'))}")
            if news.get("urgency"):
                lines.append("🚨 Urgent News")

        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ Format error: {e}"

def html_safe(s: str) -> str:
    """Escape HTML special chars for safe Telegram HTML mode."""
    if s is None:
        return ""
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))

def parse_pair_tf(text: str):
    """
    Parse many input formats into (PAIR, TF).
    """
    if not text:
        return None, "15m"
    t = text.upper().replace("/", " ").replace("_", " ").strip()
    # timeframe
    tf_match = re.search(r"(\d+\s*[MHDW])", t)
    tf = tf_match.group(1).replace(" ", "").lower() if tf_match else "15m"
    # remove verbs and tokens that are not pair
    t_clean = re.sub(r"\b(ANALISA|ANALYZE|ANALYSE|CHECK|FORCE|SCALP|INFO|ALL)\b", " ", t, flags=re.IGNORECASE).strip()
    # aliases
    aliases = {
        "GOLD": "XAUUSD", "EMAS": "XAUUSD",
        "BITCOIN": "BTCUSDT", "BTC": "BTCUSDT",
        "ETH": "ETHUSDT", "SOL": "SOLUSDT", "EUR": "EURUSD"
    }
    for a, v in aliases.items():
        if a in t_clean:
            return v, tf
    # find pair tokens like BTC USDT or BTCUSDT
    m = re.search(r"([A-Z0-9]{3,6})\s*([A-Z]{3,4})", t_clean)
    if m:
        pair = (m.group(1) + m.group(2)).upper()
        if not validate_pair_format(pair):
            vprint(f"[WARNING] Invalid pair format: {pair}")
            return None, tf
        return pair, tf
    m2 = re.search(r"([A-Z]{3,6}(?:USDT|USD|EUR|JPY|GBP|IDR|BTC|ETH))", t_clean)
    if m2:
        pair = m2.group(1).upper()
        if not validate_pair_format(pair):
            vprint(f"[WARNING] Invalid pair format: {pair}")
            return None, tf
        return pair, tf
    # fallback: first token
    token = t_clean.split()[0] if t_clean.split() else None
    if token:
        pair = token.replace(" ", "").upper()
        if not validate_pair_format(pair):
            vprint(f"[WARNING] Invalid pair format: {pair}")
            return None, tf
        return pair, tf
    return None, tf

def send_request_get(endpoint: str, params: dict = None, timeout: int = API_TIMEOUT):
    """Send GET request to backend APP_URL with error wrapper."""
    if not APP_URL:
        vprint("[REQUEST] APP_URL not configured")
        return {"error": "APP_URL not configured"}
    url = f"{APP_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    vprint(f"[REQUEST GET] {url} params={params}")
    try:
        r = requests.get(url, params=params, timeout=timeout)
        try:
            j = r.json()
            vprint(f"[RESPONSE] {endpoint} -> {j if isinstance(j, dict) else str(j)[:250]}")
            return j
        except Exception:
            vprint(f"[RESPONSE] invalid json: {r.text[:250]}")
            return {"error": f"invalid_json_response: {r.text}"}
    except Exception as e:
        vprint(f"[REQUEST ERROR] {e}")
        return {"error": str(e)}

# ---------------- COMPATIBLE ENGINE AGGREGATOR ----------------
def aggregate_compatible_engines(pair: str, tf: str):
    """
    Call only endpoints that exist in main_combined_learning.py
    """
    results = {}
    try:
        # PRO TECHNICAL (exists)
        try:
            res_pro = send_request_get("pro_signal", params={"pair": pair, "tf_entry": tf})
            results["pro_signal"] = res_pro
        except Exception as e:
            results["pro_signal"] = {"error": str(e)}

        time.sleep(0.25)
        
        # ENHANCED PRO SIGNAL (exists)
        try:
            res_enhanced = send_request_get("pro_signal_enhanced", params={"pair": pair, "tf_entry": tf})
            results["enhanced_signal"] = res_enhanced
        except Exception as e:
            results["enhanced_signal"] = {"error": str(e)}

        time.sleep(0.25)
        
        # ICT ANALYSIS (exists)
        try:
            res_ict = send_request_get("ict_analysis", params={"pair": pair, "tf": tf})
            results["ict_analysis"] = res_ict
        except Exception as e:
            results["ict_analysis"] = {"error": str(e)}

        time.sleep(0.25)
        
        # NEWS SUMMARY (exists)
        try:
            res_news = send_request_get("news_summary", params={"pair": pair, "page_size": 10})
            results["news_summary"] = res_news
        except Exception as e:
            results["news_summary"] = {"error": str(e)}

    except Exception as e:
        vprint("[AGGREGATE] Unexpected error:", e)
    return results

def format_multi_message(pair: str, tf: str, agg: dict) -> str:
    """Format aggregated multi-engine message into one Telegram HTML text."""
    parts = []
    header = f"🤖 <b>Multi-Engine Report</b>\n📌 Pair: <b>{pair}</b> | TF: <b>{tf}</b>\n⏱ {datetime.utcnow().isoformat()} UTC\n"
    parts.append(header)
    
    # PRO Signal
    pro = agg.get("pro_signal")
    if pro:
        parts.append("🔷 <b>PRO Signal</b>\n" + (format_signal(pro) if isinstance(pro, dict) and "error" not in pro else f"⚠️ PRO error: {html_safe(str(pro.get('error')))}"))
    
    # Enhanced Signal
    enhanced = agg.get("enhanced_signal")
    if enhanced:
        parts.append("🚀 <b>Enhanced PRO</b>\n" + (format_signal(enhanced) if isinstance(enhanced, dict) and "error" not in enhanced else f"⚠️ Enhanced error: {html_safe(str(enhanced.get('error')))}"))
    
    # ICT Analysis
    ict = agg.get("ict_analysis")
    if ict:
        if isinstance(ict, dict) and "error" not in ict:
            if "signal_info" in ict:
                si = ict.get("signal_info", {})
                parts.append(f"🧩 <b>ICT Analysis</b>\nBias: {html_safe(str(si.get('bias')))} | Signal: {html_safe(str(si.get('signal_type')))} | Confidence: {html_safe(str(si.get('confidence')))}")
            else:
                parts.append("🧩 <b>ICT Analysis</b>\n" + format_signal(ict) if isinstance(ict, dict) else str(ict))
        else:
            parts.append(f"⚠️ ICT error: {html_safe(str(ict.get('error')))}")
    
    # News Summary
    news = agg.get("news_summary")
    if news:
        if isinstance(news, dict) and "error" not in news:
            summarizer = news.get("summarizer", {})
            parts.append(f"📰 <b>News Summary</b>\nBias: {html_safe(summarizer.get('agg_bias', 'neutral'))} | Impact: {html_safe(summarizer.get('impact', 'LOW'))}\nSummary: {html_safe(summarizer.get('combined_summary', '')[:200])}")
        else:
            parts.append(f"⚠️ News error: {html_safe(str(news.get('error')))}")
    
    # Join parts and ensure length reasonable
    message = "\n\n".join(parts)
    if len(message) > 4000:
        message = message[:3900] + "\n\n... (truncated)"
    return message

# ---------------- AUTO-SCAN LOGIC ----------------
def auto_check_and_send(app):
    """
    Iterate AUTO_PAIRS and AUTO_TIMEFRAMES; call /pro_signal; send to CHAT_ID if strong.
    """
    if not auto_scan_lock.acquire(blocking=False):
        vprint("[AUTO] Scan already running, skipping...")
        return
        
    try:
        bot = app.bot
        now = datetime.utcnow().isoformat()
        vprint(f"[AUTO] Auto-scan start {now} - pairs {len(AUTO_PAIRS)} TF {AUTO_TIMEFRAMES}")
        for pair in AUTO_PAIRS:
            for tf in AUTO_TIMEFRAMES:
                try:
                    params = {"pair": pair, "tf_entry": tf}
                    res = send_request_get("pro_signal", params=params)
                    if not isinstance(res, dict):
                        vprint(f"[AUTO] non-dict response for {pair} {tf}: {res}")
                        continue
                    if "error" in res:
                        vprint(f"[AUTO] {pair} {tf} -> error: {res.get('error')}")
                        continue

                    engine = res.get("engine_mode", "unknown")
                    vprint(f"[AUTO] {pair} {tf} -> engine={engine}, signal_type={res.get('signal_type')}, conf={res.get('confidence')}")

                    conf = float(res.get("confidence", 0) or 0)
                    if conf >= STRONG_SIGNAL_THRESHOLD and res.get("signal_type") and res.get("signal_type") != "WAIT":
                        msg = format_signal(res)
                        try:
                            if not CHAT_ID:
                                vprint("[AUTO] CHAT_ID not configured; skipping send.")
                            else:
                                bot.send_message(chat_id=int(CHAT_ID), text=msg, parse_mode="HTML")
                                vprint(f"[AUTO] Sent strong signal {pair} {tf} (conf={conf})")
                        except Exception as e:
                            vprint(f"[AUTO ERROR] send_message failed for {pair} {tf}: {e}")
                    else:
                        vprint(f"[AUTO] {pair} {tf} no strong signal (conf={conf})")
                    time.sleep(0.6)
                except Exception as e:
                    vprint(f"[AUTO EXC] {pair} {tf}: {e}")
                    time.sleep(0.3)
        vprint(f"[AUTO] Auto-scan finished at {datetime.utcnow().isoformat()}")
    finally:
        auto_scan_lock.release()

def start_auto_job(app):
    global auto_job
    with auto_job_lock:
        if auto_job is None:
            vprint("[AUTO] Scheduling job (interval hours=%s)" % AUTO_SCAN_HOURS)
            try:
                auto_job = scheduler.add_job(
                    lambda: auto_check_and_send(app), 
                    'interval', 
                    hours=AUTO_SCAN_HOURS, 
                    next_run_time=None,
                    max_instances=1
                )
                vprint("[AUTO] Job scheduled.")
            except Exception as e:
                vprint(f"[AUTO] Failed to schedule job: {e}")
        else:
            vprint("[AUTO] Job already running.")

def stop_auto_job():
    global auto_job
    with auto_job_lock:
        if auto_job is not None:
            try:
                auto_job.remove()
            except Exception:
                pass
            auto_job = None
            vprint("[AUTO] Job removed.")
        else:
            vprint("[AUTO] No auto job to remove.")

# ---------------- TELEGRAM HANDLERS ----------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    vprint("[CMD] /start from", update.effective_user.id if update.effective_user else "unknown")
    msg = (
        "🤖 <b>ProTraderAI - Assistant</b>\n\n"
        "Send commands like:\n"
        "- <code>BTCUSDT 15m</code> or <code>analisa BTCUSDT 15m</code>\n"
        "- <code>force BTCUSDT 15m</code> (show all signals)\n"
        "- <code>BTCUSDT 15m ALL</code> (call all engines)\n\n"
        "Commands:\n"
        "/status - model status\n"
        "/logs - recent signals\n"
        "/performance - AI performance\n"
        "/retrain - retrain ML model\n"
        "/auto_on - enable auto-scan\n"
        "/auto_off - disable auto-scan\n"
        "/backtest - run backtest\n"
    )
    await update.message.reply_text(msg, parse_mode="HTML")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    vprint("[CMD] /status")
    try:
        res = send_request_get("health", params=None)
        if "error" in res:
            await update.message.reply_text(f"⚠️ {res.get('error')}")
            return
        msg = (
            "📊 <b>AI Agent Status</b>\n\n"
            f"🟢 Status: {res.get('status', 'unknown')}\n"
            f"📦 Service: {res.get('service', 'unknown')}\n"
            f"🔖 Version: {res.get('version', 'unknown')}\n"
            f"⏰ Time: {datetime.utcnow().isoformat()} UTC"
        )
        await update.message.reply_text(msg, parse_mode="HTML")
    except Exception as e:
        vprint("[ERROR] status_cmd:", e)
        await update.message.reply_text(f"❌ Failed to get status.\nError: {e}")

async def logs_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    vprint("[CMD] /logs")
    try:
        # Try to get recent trade from trade log
        trade_log_file = "trade_log.csv"
        if os.path.exists(trade_log_file):
            with open(trade_log_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    last_trade = rows[-1]
                    msg = (
                        "📋 <b>Last Trade Log</b>\n\n"
                        f"🪙 Pair: {last_trade.get('pair', '?')}\n"
                        f"🕒 Timeframe: {last_trade.get('timeframe', '?')}\n"
                        f"💡 Signal: {last_trade.get('signal_type', '?')}\n"
                        f"🎯 Entry: {last_trade.get('entry', '?')}\n"
                        f"🎯 TP1: {last_trade.get('tp1', '?')} | TP2: {last_trade.get('tp2', '?')}\n"
                        f"🛑 SL: {last_trade.get('sl', '?')}\n"
                        f"📊 Confidence: {last_trade.get('confidence', '?')}\n\n"
                        f"🧠 Reasoning:\n{last_trade.get('reasoning', '-')}"
                    )
                    await update.message.reply_text(msg, parse_mode="HTML")
                    return
        
        await update.message.reply_text("⚠️ No trade logs found yet.")
    except Exception as e:
        vprint("[ERROR] logs_cmd:", e)
        await update.message.reply_text(f"❌ Failed to get logs.\nError: {e}")

async def backtest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args or len(args) < 2:
        await update.message.reply_text("Usage: /backtest PAIR TF\nExample: /backtest BTCUSDT 15m")
        return
        
    pair = args[0].upper()
    tf = args[1].lower()
    
    if not validate_pair_format(pair):
        await update.message.reply_text("❌ Invalid pair format")
        return
        
    allowed_tfs = ["1m", "5m", "15m", "1h", "4h", "1d"]
    if tf not in allowed_tfs:
        await update.message.reply_text(f"❌ Invalid timeframe. Allowed: {', '.join(allowed_tfs)}")
        return
    
    # FIXED: Use correct backtester file and parameters
    outdir = "backtest_results"
    cmd = [
        "python", "pro_backtester_full.py",
        "--pair", pair, 
        "--tf", tf, 
        "--out", outdir
    ]
    
    await update.message.reply_text(f"Running backtest {pair} {tf} — this may take a few minutes...")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate(timeout=300)
        
        if proc.returncode == 0:
            await update.message.reply_text(f"✅ Backtest completed for {pair} {tf}\nResults saved to {outdir}/")
            
            # Try to send summary file if exists
            summary_file = f"{outdir}/{pair}_{tf}_full_summary.json"
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    fusion = summary.get('fusion_summary', {})
                    msg = "📊 <b>Backtest Summary</b>\n\n"
                    for engine, stats in fusion.items():
                        msg += f"<b>{engine}</b>\n"
                        msg += f"Trades: {stats.get('trades', 0)} | Winrate: {stats.get('winrate', 0)}\n"
                        msg += f"Return: {stats.get('total_return', 0):.2f} | PF: {stats.get('profit_factor', 0)}\n\n"
                    await update.message.reply_text(msg, parse_mode="HTML")
        else:
            await update.message.reply_text(f"❌ Backtest failed:\n{err}")
            
    except subprocess.TimeoutExpired:
        await update.message.reply_text("⏰ Backtest timeout - process took too long")
    except Exception as e:
        await update.message.reply_text(f"❌ Backtest error: {e}")

async def performance_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    vprint("[CMD] /performance")
    try:
        # Create simple performance report from trade logs
        trade_log_file = "trade_log.csv"
        if os.path.exists(trade_log_file):
            with open(trade_log_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    total_trades = len(rows)
                    winning_trades = len([r for r in rows if float(r.get('confidence', 0)) > 0.7])
                    winrate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                    
                    msg = (
                        "📈 <b>Performance Report</b>\n\n"
                        f"📊 Total Signals: {total_trades}\n"
                        f"🏆 High Confidence Signals: {winning_trades}\n"
                        f"🎯 Effective Winrate: {winrate:.1f}%\n"
                        f"⏰ Period: From {rows[0].get('timestamp', '?')} to {rows[-1].get('timestamp', '?')}"
                    )
                    await update.message.reply_text(msg, parse_mode="HTML")
                    return
        
        await update.message.reply_text("⚠️ No performance data available yet.")
    except Exception as e:
        vprint("[ERROR] performance_cmd:", e)
        await update.message.reply_text(f"❌ Failed to get performance data.\nError: {e}")

async def auto_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global auto_enabled
    auto_enabled = True
    vprint("[CMD] /auto_on")
    try:
        app = context.application
        start_auto_job(app)
        await update.message.reply_text("✅ Auto-scan enabled.")
    except Exception as e:
        vprint("[ERROR] auto_on_cmd:", e)
        await update.message.reply_text(f"❌ Failed to enable auto-scan: {e}")

async def auto_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global auto_enabled
    auto_enabled = False
    vprint("[CMD] /auto_off")
    try:
        stop_auto_job()
        await update.message.reply_text("⛔ Auto-scan disabled.")
    except Exception as e:
        vprint("[ERROR] auto_off_cmd:", e)
        await update.message.reply_text(f"❌ Failed to disable auto-scan: {e}")

async def retrain_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    vprint("[CMD] /retrain")
    try:
        await update.message.reply_text("🧠 Retraining AI models... please wait (this may take time) ⏳")

        res = send_request_get("retrain_historical", params=None)
        
        if "error" in res:
            await update.message.reply_text(f"❌ Failed to retrain models.\nError: {res.get('error')}")
            return

        msg = (
            "✅ <b>Models retrained successfully!</b>\n\n"
            f"📊 Status: {res.get('status', 'completed')}\n"
        )
        
        # Add results if available
        if 'result' in res:
            result = res['result']
            if 'rf' in result:
                rf = result['rf']
                msg += f"🌲 RF: CV Acc={rf.get('mean_cv_accuracy')}±{rf.get('std_cv_accuracy')}\n"
            if 'xgb' in result:
                xgb = result['xgb']
                msg += f"📈 XGB: CV Acc={xgb.get('mean_cv_accuracy')}±{xgb.get('std_cv_accuracy')}\n"
        
        await update.message.reply_text(msg, parse_mode="HTML")

    except Exception as e:
        vprint("[ERROR] retrain_cmd:", e)
        await update.message.reply_text(f"❌ Retrain failed.\nError: {e}")

async def manual_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip() if update.message.text else ""
    vprint("[MSG] manual_message received:", text[:200])
    if not text:
        return
        
    user_id = update.effective_user.id
    if not await check_rate_limit(user_id):
        await update.message.reply_text("⏳ Too many requests. Please wait a minute.")
        return
        
    t_low = text.lower()
    is_force = t_low.startswith("force")
    if is_force:
        text = text.replace("force", "", 1).strip()

    multi_flag = False
    if re.search(r"\ball\b", text, flags=re.IGNORECASE):
        multi_flag = True
        text = re.sub(r"\ball\b", " ", text, flags=re.IGNORECASE).strip()

    pair, tf = parse_pair_tf(text)
    if not pair:
        await update.message.reply_text("❌ Cannot detect pair from message.")
        return

    # Multi-engine request
    if multi_flag:
        await update.message.reply_text(f"🔀 Running Multi-Engine scan for {pair} ({tf}) ...")
        vprint(f"[MULTI] Aggregating for {pair} {tf}")
        agg = aggregate_compatible_engines(pair, tf)
        msg = format_multi_message(pair, tf, agg)
        try:
            await update.message.reply_text(msg, parse_mode="HTML")
        except Exception as e:
            vprint("[MULTI] send failed:", e)
            await update.message.reply_text(msg)
        return

    # Normal single-engine request
    await update.message.reply_text(f"🔍 Analyzing {pair} ({tf}) ...")
    vprint(f"[ANALYSE] requesting pro_signal pair={pair} tf_entry={tf}")
    res = send_request_get("pro_signal", params={"pair": pair, "tf_entry": tf})

    vprint(f"[ANALYSE] response for {pair} ({tf}): {res}")
    if "error" in res:
        await update.message.reply_text(f"❌ Error: {res.get('error')}")
        return

    engine_mode = res.get("engine_mode", "unknown")
    vprint(f"[ANALYSE] engine_mode={engine_mode}, signal_type={res.get('signal_type')}, confidence={res.get('confidence')}")

    conf = float(res.get("confidence", 0) or 0)
    if (not is_force) and conf < STRONG_SIGNAL_THRESHOLD:
        await update.message.reply_text(
            f"⚠️ No strong signal for {pair} ({tf}).\nCurrent confidence: {conf}",
            parse_mode="HTML"
        )
        return

    await update.message.reply_text(format_signal(res), parse_mode="HTML")

# ---------------- SETUP & RUN ----------------
def main():
    vprint("[STARTUP] Starting telegram_bot_fixed.py")
    
    config_errors = validate_config()
    if config_errors:
        print("❌ Configuration errors:")
        for err in config_errors:
            print(f"   - {err}")
        return
    
    if not BOT_TOKEN:
        print("❌ BOT_TOKEN not set in environment.")
        return
        
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Register handlers
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("logs", logs_cmd))
    app.add_handler(CommandHandler("performance", performance_cmd))
    app.add_handler(CommandHandler("backtest", backtest_cmd))
    app.add_handler(CommandHandler("auto_on", auto_on_cmd))
    app.add_handler(CommandHandler("auto_off", auto_off_cmd))
    app.add_handler(CommandHandler("retrain", retrain_cmd))

    # Message handlers
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, manual_message))

    # Start scheduler and auto job
    scheduler.start()
    start_auto_job(app)

    vprint(f"[STARTUP] Bot running. Auto-scan every {AUTO_SCAN_HOURS} hour(s). Pairs: {len(AUTO_PAIRS)} TF: {AUTO_TIMEFRAMES}")
    try:
        app.run_polling()
    except KeyboardInterrupt:
        vprint("[SHUTDOWN] Keyboard interrupt received")
    finally:
        stop_event.set()
        try:
            stop_auto_job()
            scheduler.shutdown(wait=True)
        except Exception:
            pass
        vprint("[SHUTDOWN] Bot stopped completely")

if __name__ == "__main__":
    main()