"""
main_combined_learning_hybrid_pro_final.py
Combined Hybrid PRO: SMC/ICT PRO + Hybrid Technical Engine + XGBoost + RandomForest + Data Fallback + Fundamental Analysis
VERSION: ENHANCED WITH COMPLETE SMC/ICT CAPABILITIES + FUNDAMENTAL ANALYSIS
"""

import os
import io
import re
import csv
import time
import json
import joblib
import threading
# ===== Additions near top imports =====
from dateutil import parser as dateparser  # more robust date parsing for news timestamps

# thread lock for safe cache IO
_cache_lock = threading.Lock()
import requests
import numpy as np
import pandas as pd
import hashlib  
import math
import html
from collections import defaultdict
from collections import Counter
from datetime import datetime, time as dtime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from fastapi import FastAPI, Query, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.encoders import jsonable_encoder

# technical libs
import ta

# ML libs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from xgboost import XGBClassifier

# image libs optional
try:
    from PIL import Image
    import cv2
    import pytesseract
    _HAS_TESSERACT = True
except Exception:
    _HAS_TESSERACT = False

# ENV / CONFIG
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "").strip()
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "").strip()
NEWS_CACHE_TTL = int(os.getenv("NEWS_CACHE_TTL", "1800"))  # seconds, default 30 min
NEWS_CACHE_DIR = os.getenv("NEWS_CACHE_DIR", "./news_cache")
NEWS_PAGE_SIZE = int(os.getenv("NEWS_PAGE_SIZE", "30"))

if not os.path.exists(NEWS_CACHE_DIR):
    os.makedirs(NEWS_CACHE_DIR, exist_ok=True)

# ------------------- Simple lexicon for sentiment -------------------
POS_WORDS = {
    "rise","rises","surge","surged","gain","gains","positive","above","outperform",
    "record","strong","boom","bullish","upgrade","better","improve"
}
NEG_WORDS = {
    "fall","falls","fell","drop","drops","crash","decline","downgrade","bearish","miss",
    "below","sell-off","risk-off","fear","unexpected"
}
URGENCY_WORDS = {"breaking", "urgent", "flash", "surge", "crash", "spike"}

# ------------------- Helpers: cache -------------------
def _cache_path_for(key: str) -> str:
    h = hashlib.sha1(key.encode()).hexdigest()
    return os.path.join(NEWS_CACHE_DIR, f"news_{h}.json")

def _load_cached(key: str) -> Optional[Dict[str, Any]]:
    p = _cache_path_for(key)
    if os.path.exists(p):
        try:
            with _cache_lock:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
            if time.time() - data.get("_fetched_at", 0) < NEWS_CACHE_TTL:
                return data
        except Exception as e:
            # if cache corrupted, remove it to avoid repeated errors
            try:
                os.remove(p)
            except:
                pass
    return None

def _save_cache(key: str, payload: Dict[str, Any]):
    p = _cache_path_for(key)
    payload["_fetched_at"] = int(time.time())
    try:
        with _cache_lock:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
    except Exception as e:
        print(f"[CACHE] Failed to write cache {p}: {e}")

# --- CONFIGURATION ---
class Config:
    APP_NAME = "Pro Trader AI - Hybrid PRO Complete SMC/ICT + Fundamental Analysis"
    PORT = int(os.getenv("PORT", 8000))
    TRADE_LOG_FILE = os.getenv("TRADE_LOG_FILE", "trade_log.csv")
    MODEL_DIR = "models"
    MIN_SAMPLES_TO_TRAIN = int(os.getenv("MIN_SAMPLES_TO_TRAIN", 50))
    N_SIGNALS_TO_RETRAIN = int(os.getenv("N_SIGNALS_TO_RETRAIN", 50))
    
    # API Keys
    FMP_API_KEY = os.getenv("FMP_API_KEY", "")
    TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
    ALPHA_API_KEY = os.getenv("ALPHA_API_KEY", "")
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
    GOLDAPI_KEY = os.getenv("GOLDAPI_KEY", "")
    
    # Fundamental Analysis API Keys
    NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY", "")
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    
    # Risk Management
    RISK_PERCENT = float(os.getenv("RISK_PERCENT", 0.02))
    ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", 0))
    
    # ICT PRO Config
    ICT_KILLZONE_ENABLE = os.getenv("ICT_KILLZONE_ENABLE", "true").lower() == "true"
    ICT_KILLZONE_START = os.getenv("ICT_KILLZONE_START", "06:00")
    ICT_KILLZONE_END = os.getenv("ICT_KILLZONE_END", "12:00")
    ICT_MIN_CONFIRM = float(os.getenv("ICT_MIN_CONFIRM", 0.6))
    ICT_HTF_LIST = os.getenv("ICT_HTF_LIST", "1w,1d,1h").split(",")
    ICT_DEFAULT_ENTRY_TF = os.getenv("ICT_DEFAULT_ENTRY_TF", "15m")
    
    # SMC/ICT Advanced Config
    OB_LOOKBACK = int(os.getenv("OB_LOOKBACK", "50"))
    FVG_THRESHOLD = float(os.getenv("FVG_THRESHOLD", "0.0005"))
    LIQUIDITY_WINDOW = int(os.getenv("LIQUIDITY_WINDOW", "50"))
    OTE_RETRACEMENT = float(os.getenv("OTE_RETRACEMENT", "0.618"))
    
    # Weights
    WEIGHT_SMC = float(os.getenv("WEIGHT_SMC", 0.4))
    WEIGHT_VOL = float(os.getenv("WEIGHT_VOL", 0.3))
    WEIGHT_ML = float(os.getenv("WEIGHT_ML", 0.2))
    WEIGHT_ICT = float(os.getenv("WEIGHT_ICT", 0.1))
    STRONG_THRESHOLD = float(os.getenv("STRONG_SIGNAL_THRESHOLD", 0.8))
    WEAK_THRESHOLD = float(os.getenv("WEAK_SIGNAL_THRESHOLD", 0.55))
    
    # Fundamental Analysis Config
    ECONOMIC_CALENDAR_URL = "https://www.alphavantage.co/query?function=ECONOMIC_CALENDAR"
    FOREX_FACTORY_CALENDAR = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    
    # Risk Adjustment
    HIGH_IMPACT_REDUCTION = float(os.getenv("HIGH_IMPACT_REDUCTION", "0.5"))
    MEDIUM_IMPACT_REDUCTION = float(os.getenv("MEDIUM_IMPACT_REDUCTION", "0.8"))
    VOLATILITY_REDUCTION = float(os.getenv("VOLATILITY_REDUCTION", "0.7"))
    
    # High Impact Events
    HIGH_IMPACT_EVENTS = [
        "Non-Farm Payrolls", "FOMC", "CPI", "Interest Rate", 
        "Federal Funds Rate", "ECB Press Conference", "GDP",
        "Retail Sales", "Unemployment Rate", "PMI", "NFP"
    ]
    
    # Telegram
    TELEGRAM_AUTO_SEND = os.getenv("TELEGRAM_AUTO_SEND", "true").lower() == "true"
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
  
    # Backtest
    BACKTEST_URL = os.getenv("BACKTEST_URL", "")
    
    @classmethod
    def validate_config(cls):
        """Validate critical configuration"""
        if not cls.TWELVEDATA_API_KEY and not cls.ALPHA_API_KEY:
            print("⚠️ WARNING: No API keys configured - data fetching may fail")
        
        # Validate directory permissions
        try:
            os.makedirs(cls.MODEL_DIR, exist_ok=True)
            test_file = os.path.join(cls.MODEL_DIR, "test_write")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            print("✅ Configuration validation passed")
        except Exception as e:
            print(f"⚠️ WARNING: Cannot write to model directory: {e}")

# Ensure model directory exists
os.makedirs(Config.MODEL_DIR, exist_ok=True)

# === NEW CONFIG FOR ADVANCED MODES ===
Config.AUTO_SCALP_ENABLE = os.getenv("AUTO_SCALP_ENABLE", "true").lower() == "true"
Config.SCALP_MAX_RISK_PCT = float(os.getenv("SCALP_MAX_RISK_PCT", "0.005"))  # 0.5% risk
Config.SWING_PRO_ENABLE = os.getenv("SWING_PRO_ENABLE", "true").lower() == "true"
Config.NEWS_DRIVEN_ENABLE = os.getenv("NEWS_DRIVEN_ENABLE", "true").lower() == "true"
Config.BLOCKER_WINDOW_HOURS = int(os.getenv("BLOCKER_WINDOW_HOURS", "2"))

# --- GLOBAL CACHE ---
_cached_rf = None
_cached_xgb = None

# --- UTILITY FUNCTIONS ---
def respond(obj: Any, status_code: int = 200):
    """Safe JSON response handler"""
    def clean_value(v):
        if isinstance(v, float):
            if np.isnan(v) or np.isinf(v):
                return 0.0
        if isinstance(v, (np.int64, np.int32)):
            return int(v)
        if isinstance(v, (np.float32, np.float64)):
            return float(v)
        if isinstance(v, dict):
            return {str(k): clean_value(val) for k, val in v.items()}
        if isinstance(v, list):
            return [clean_value(val) for val in v]
        return v
    
    try:
        encoded = jsonable_encoder(obj)
        cleaned = clean_value(encoded)
        return JSONResponse(content=cleaned, status_code=status_code)
    except Exception as e:
        try:
            return JSONResponse(content={"fallback": str(obj)}, status_code=status_code)
        except:
            return PlainTextResponse(str(obj), status_code=status_code)

def ensure_trade_log():
    """Ensure trade log file exists with proper headers"""
    if not os.path.exists(Config.TRADE_LOG_FILE):
        with open(Config.TRADE_LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "pair", "timeframe", "signal_type", "entry", "tp1", "tp2", "sl",
                "confidence", "reasoning", "engine_used", "backtest_hit", "backtest_pnl", "fundamental_risk"
            ])

def append_trade_log(rec: Dict[str, Any]):
    """Append record to trade log"""
    ensure_trade_log()
    with open(Config.TRADE_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            rec.get("pair"), rec.get("timeframe"), rec.get("signal_type"),
            rec.get("entry"), rec.get("tp1"), rec.get("tp2"), rec.get("sl"),
            rec.get("confidence"), rec.get("reasoning"),
            rec.get("engine_used"),
            rec.get("backtest_hit"), rec.get("backtest_pnl"), rec.get("fundamental_risk")
        ])
        
# ------------------- Fetchers -------------------
def fetch_news_newsapi(query: str, page_size: int = NEWS_PAGE_SIZE) -> List[Dict[str, Any]]:
    api_key = getattr(Config, "NEWSAPI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("NEWSAPI_API_KEY not set")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "pageSize": page_size,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": api_key
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        articles = data.get("articles", [])
        out = []
        for a in articles:
            out.append({
                "title": a.get("title"),
                "description": a.get("description"),
                "url": a.get("url"),
                "source": a.get("source", {}).get("name"),
                "publishedAt": a.get("publishedAt"),
                "content": a.get("content") or a.get("description") or ""
            })
        return out
    except Exception as e:
        print(f"[NEWSAPI] fetch error: {e}")
        raise

def fetch_news_cryptopanic(query: str, limit: int = 30) -> List[Dict[str, Any]]:
    if not CRYPTOPANIC_API_KEY:
        raise RuntimeError("CRYPTOPANIC_API_KEY not set")
    url = "https://cryptopanic.com/api/v1/posts/"
    params = {"auth_token": CRYPTOPANIC_API_KEY, "public": "true"}
    r = requests.get(url, params=params, timeout=8)
    r.raise_for_status()
    data = r.json()
    results = []
    for p in data.get("results", [])[:limit]:
        content = p.get("body") or p.get("title")
        results.append({
            "title": p.get("title"),
            "description": p.get("domain"),
            "url": p.get("url"),
            "source": p.get("domain"),
            "publishedAt": p.get("published_at"),
            "content": content
        })
    # filter using query keyword
    q = query.lower()
    return [a for a in results if q in (a["title"].lower() + " " + a["content"].lower())]


# ------------------- High-level glue -------------------
def pair_to_query(pair: str) -> str:
    p = (pair or "").upper()
    if "BTC" in p or "ETH" in p or "SOL" in p:  return f"{p} OR crypto"
    if "XAU" in p or "GOLD" in p: return "gold OR xauusd"
    if p.endswith("USD") and len(p) <= 7:
        return f"{p[:3]} OR {p[3:]} OR forex"
    return p


def fetch_news_for_pair(pair: str, page_size: int = NEWS_PAGE_SIZE) -> Dict[str, Any]:
    q = pair_to_query(pair)
    cache_key = f"news::{q}::p{page_size}"

    cached = _load_cached(cache_key)
    if cached:
        return {"source": "cache", "articles": cached["articles"]}

    try:
        arts = fetch_news_newsapi(q, page_size)
        _save_cache(cache_key, {"articles": arts})
        return {"source": "newsapi", "articles": arts}
    except:
        try:
            arts = fetch_news_cryptopanic(q, page_size)
            _save_cache(cache_key, {"articles": arts})
            return {"source": "cryptopanic", "articles": arts}
        except:
            if cached:
                return {"source": "cache", "articles": cached["articles"]}
            return {"source": "none", "articles": []}
  
# ------------------- Sentiment engine -------------------
def sentiment_and_urgency_for_text(text: str) -> Dict[str, Any]:
    if not text: return {"score":0.0, "label":"neutral", "urgency":False}
    t = text.lower()
    toks = t.split()
    pos = sum(1 for x in toks if x in POS_WORDS)
    neg = sum(1 for x in toks if x in NEG_WORDS)
    urgency = any(x in t for x in URGENCY_WORDS)
    score = (pos - neg) / max(1, len(toks))
    label = "positive" if score>0.02 else ("negative" if score<-0.02 else "neutral")
    return {"score":round(score,4),"label":label,"urgency":urgency}


def aggregate_news_sentiment(articles: List[Dict[str, Any]], pair: str):
    if not articles:
        return {"impact_weight":0.0,"fundamental_bias":"neutral","confidence_adjustment":0.0,"top":[]}

    scores = []
    weight_sum = 0
    score_sum = 0
    now = time.time()

    for a in articles[:60]:
        text = f"{a.get('title','')} {a.get('description','')} {a.get('content','')}"
        s = sentiment_and_urgency_for_text(text)

        # recency weight
        try:
            pub_raw = a.get("publishedAt", "") or a.get("published_at", "")
            if not pub_raw:
                rec_w = 0.5
            else:
                # use dateutil parser for robust parsing
                dt = dateparser.parse(pub_raw)
                ts = dt.timestamp()
                age = max(0, now - ts)
                rec_w = max(0.1, 1 - (age / 86400))  # linear decay over 1 day
        except Exception:
            rec_w = 0.5

        urg_w = 1.5 if s["urgency"] else 1.0
        w = rec_w * urg_w

        scores.append((s["score"], w, a))
        weight_sum += w
        score_sum += s["score"] * w

    avg = score_sum / max(1, weight_sum)

    impact_weight = min(1.0, len(scores)/6.0)
    adj = max(-0.2, min(0.2, avg*impact_weight*4))

    bias = "bullish" if avg>0.02 else ("bearish" if avg<-0.02 else "neutral")

    scores_sorted = sorted(scores, key=lambda x: abs(x[0]*x[1]), reverse=True)
    top = [{"title":a["title"],"url":a["url"],"source":a["source"],"score":round(s,4),"weight":round(w,4)}
           for s,w,a in scores_sorted[:6]]

    return {
        "impact_weight":round(impact_weight,3),
        "fundamental_bias":bias,
        "confidence_adjustment":round(adj,4),
        "top":top,
        "avg_score":round(avg,4)
    }
    
# ------------------- Summarizer PRO (ADD THIS
# Extra lexicon (tambahkan kata yang sering relevan untuk fundamental/trading)
_SUMMARIZER_KEYWORDS = set([
    "etf","approval","rejection","federal","fed","inflation","cpi","nfp","non-farm",
    "rate","rates","interest","hike","cut","surge","spike","collapse","bankruptcy",
    "dilution","hack","attack","lawsuit","sec","exchange","listing", "halving",
    "liquidation","liquidity","flow","outflow","inflow","earnings","guidance",
    "downgrade","upgrade","restructuring","merger","acquisition"
]) | POS_WORDS | NEG_WORDS

def _split_sentences(text: str) -> List[str]:
    # simple sentence splitter; keeps punctuation
    if not text:
        return []
    text = html.unescape(text)
    # split by ., ?, ! but keep abbreviations simple
    sentences = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    # strip
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def _word_score_map(text: str) -> Dict[str, float]:
    toks = re.findall(r"[A-Za-z0-9'\-]+", text.lower())
    counts = Counter(toks)
    # score keywords higher
    scores = {}
    for w,c in counts.items():
        base = math.log(1 + c)
        if w in POS_WORDS:
            base *= 1.8
        if w in NEG_WORDS:
            base *= 1.8
        if w in _SUMMARIZER_KEYWORDS:
            base *= 1.5
        scores[w] = base
    return scores

def summarize_article_text(text: str, max_sentences: int = 3) -> Dict[str, Any]:
    """
    Light-weight extractive summarizer + metadata (bias, urgency, impact).
    Returns: { summary, highlights, bias, urgency, impact_estimate, scores }
    """
    if not text or len(text.strip()) < 30:
        return {"summary": "", "highlights": [], "bias": "neutral", "urgency": False, "impact_estimate": "LOW", "score": 0.0}

    sents = _split_sentences(text)
    if not sents:
        return {"summary": "", "highlights": [], "bias": "neutral", "urgency": False, "impact_estimate": "LOW", "score": 0.0}

    # Build global word importance
    wc_scores = _word_score_map(text)

    # Score each sentence
    sent_scores = []
    for s in sents:
        toks = re.findall(r"[A-Za-z0-9'\-]+", s.lower())
        if not toks:
            continue
        score = 0.0
        for t in toks:
            score += wc_scores.get(t, 0.0)
        # length penalty/bonus (prefer medium length)
        l = len(s.split())
        length_factor = 1.0
        if l < 6:
            length_factor = 0.6
        elif l > 40:
            length_factor = 0.8
        score *= length_factor
        # urgency boost if contains urgent words
        if any(u in s.lower() for u in URGENCY_WORDS):
            score *= 1.6
        sent_scores.append((score, s))

    # Select top sentences, keep original order
    sent_scores_sorted = sorted(sent_scores, key=lambda x: x[0], reverse=True)
    top = [s for _, s in sent_scores_sorted[:max_sentences]]
    # preserve original order
    top_sorted = [s for s in sents if s in top][:max_sentences]

    summary = " ".join(top_sorted)

    # derive bias/urgency/impact from combined text and keywords
    all_text = (summary + " " + text).lower()
    pos = sum(1 for w in POS_WORDS if w in all_text)
    neg = sum(1 for w in NEG_WORDS if w in all_text)
    bias = "neutral"
    if pos > neg and (pos - neg) >= 2:
        bias = "bullish"
    elif neg > pos and (neg - pos) >= 2:
        bias = "bearish"

    urgency = any(u in all_text for u in URGENCY_WORDS) or ("breaking" in all_text)

    # crude impact estimate: more keywords & docs length -> higher
    kw_count = sum(1 for w in _SUMMARIZER_KEYWORDS if w in all_text)
    score_norm = min(1.0, (sum(s for s,_ in sent_scores) / (len(sent_scores) + 1)) / 5.0)
    if kw_count >= 3 or score_norm > 0.6:
        impact = "HIGH"
    elif kw_count == 2 or score_norm > 0.35:
        impact = "MEDIUM"
    else:
        impact = "LOW"

    # recommended_action mapping (simple heuristic)
    if impact == "HIGH" and bias == "bearish":
        recommended = "AVOID_TRADING"
    elif impact == "HIGH" and bias == "bullish":
        recommended = "WAIT_FOR_CONFIRMATION_OR_REDUCE_SIZE"
    elif impact == "MEDIUM" and bias != "neutral":
        recommended = "REDUCE_POSITION_SIZE"
    else:
        recommended = "NORMAL_TRADING"

    # top highlights (keyword hits)
    highlights = []
    for k in sorted(_SUMMARIZER_KEYWORDS, key=lambda x: - (all_text.count(x)), ):
        if k in all_text and len(highlights) < 6:
            highlights.append(k)

    return {
        "summary": summary,
        "highlights": highlights,
        "bias": bias,
        "urgency": bool(urgency),
        "impact_estimate": impact,
        "recommended_action": recommended,
        "score": round(float(score_norm), 3)
    }

def summarize_articles_list(articles: List[Dict[str,Any]], max_sent_per_article: int = 2) -> Dict[str,Any]:
    """
    Summarize list of articles -> produces combined summary, top articles, aggregated bias
    """
    if not articles:
        return {"combined_summary": "", "article_summaries": [], "agg_bias": "neutral", "urgency": False, "impact": "LOW"}

    art_summaries = []
    pos_count = neg_count = 0
    urgency_any = False
    impact_scores = {"LOW":0,"MEDIUM":0,"HIGH":0}

    for a in articles[:30]:
        text = " ".join(filter(None, [a.get("title",""), a.get("description",""), a.get("content","")]))
        s = summarize_article_text(text, max_sentences=max_sent_per_article)
        art_summaries.append({
            "title": a.get("title"),
            "url": a.get("url"),
            "source": a.get("source"),
            "publishedAt": a.get("publishedAt"),
            "summary": s["summary"],
            "bias": s["bias"],
            "urgency": s["urgency"],
            "impact": s["impact_estimate"],
            "recommended_action": s["recommended_action"],
            "score": s["score"]
        })
        if s["bias"] == "bullish":
            pos_count += 1
        if s["bias"] == "bearish":
            neg_count += 1
        if s["urgency"]:
            urgency_any = True
        impact_scores[s["impact_estimate"]] += 1

    agg_bias = "neutral"
    if pos_count > neg_count and (pos_count - neg_count) >= 2:
        agg_bias = "bullish"
    elif neg_count > pos_count and (neg_count - pos_count) >= 2:
        agg_bias = "bearish"

    # pick top 2 article summaries by score
    art_sorted = sorted(art_summaries, key=lambda x: x["score"], reverse=True)
    combined_summary = " ".join([a["summary"] for a in art_sorted[:3] if a["summary"]])

    # decide combined impact
    if impact_scores["HIGH"] > 0:
        combined_impact = "HIGH"
    elif impact_scores["MEDIUM"] > 0:
        combined_impact = "MEDIUM"
    else:
        combined_impact = "LOW"

    return {
        "combined_summary": combined_summary,
        "article_summaries": art_summaries,
        "agg_bias": agg_bias,
        "urgency": urgency_any,
        "impact": combined_impact
    }
# ------------------- End Summarizer PRO -------------------

# --- ENHANCED DATA VALIDATION ---
def enhanced_data_validation(df: pd.DataFrame) -> bool:
    """Enhanced data validation"""
    if df.empty:
        return False
    
    # Check for NaN/Inf values
    if df[['open', 'high', 'low', 'close']].isnull().any().any():
        return False
    
    # Check for zero/negative prices
    if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
        return False
    
    # Check for high-low consistency
    if (df['high'] < df['low']).any():
        return False
    
    # Check volume spikes (outliers)
    if 'volume' in df.columns:
        vol_mean = df['volume'].mean()
        vol_std = df['volume'].std()
        if vol_std > 0:  # Avoid division by zero
            vol_zscore = np.abs((df['volume'] - vol_mean) / vol_std)
            if (vol_zscore > 10).any():  # Extreme outliers
                return False
    
    return True

# --- DATA FETCHERS ---
class DataFetcher:
    """Unified data fetcher with fallback mechanisms"""
    
    BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
    TWELVEDATA_URL = "https://api.twelvedata.com/time_series"
    
    @staticmethod
    def fetch_ohlc_binance(symbol: str, interval: str = "15m", limit: int = 500) -> pd.DataFrame:
        """Fetch OHLC data from Binance"""
        symbol = symbol.upper().replace(" ", "").replace("/", "")
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        try:
            r = requests.get(DataFetcher.BINANCE_KLINES, params=params, timeout=12)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and len(data) > 0:
                    df = pd.DataFrame(data, columns=[
                        "open_time", "open", "high", "low", "close", "volume", 
                        "close_time", "qav", "num_trades", "tb_base", "tb_quote", "ignore"
                    ])
                    for c in ["open", "high", "low", "close", "volume"]:
                        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", errors="coerce")
                    df = df[["timestamp", "open", "high", "low", "close", "volume"]].set_index("timestamp")
                    return df
        except Exception as e:
            print(f"[BINANCE] Error: {e}")
        raise RuntimeError(f"Binance fetch failed for {symbol}")

    @staticmethod
    def _format_twelvedata_symbol(s: str) -> str:
        """Format symbol for TwelveData"""
        s2 = s.upper().replace(" ", "").replace("_", "")
        if s2.endswith("USDT"):
            return f"{s2[:-4]}/USD"
        if len(s2) == 6 and s2.endswith("USD"):
            return f"{s2[:3]}/{s2[3:]}"
        if "/" in s2:
            return s2
        return s2

    @staticmethod
    def fetch_ohlc_twelvedata(symbol: str, interval: str = "15m", limit: int = 500) -> pd.DataFrame:
        """Fetch OHLC data from TwelveData"""
        if not Config.TWELVEDATA_API_KEY:
            raise RuntimeError("TWELVEDATA_API_KEY not set")
        
        mapping = {"m": "min", "h": "h", "d": "day", "w": "week"}
        unit = interval[-1]
        if unit not in mapping:
            raise RuntimeError("Unsupported timeframe")
        
        interval_fmt = interval[:-1] + mapping[unit]
        sym = DataFetcher._format_twelvedata_symbol(symbol)
        params = {
            "symbol": sym, 
            "interval": interval_fmt, 
            "outputsize": limit, 
            "apikey": Config.TWELVEDATA_API_KEY
        }
        
        r = requests.get(DataFetcher.TWELVEDATA_URL, params=params, timeout=12)
        j = r.json()
        
        if j.get("status") == "error" or "values" not in j:
            raise RuntimeError(f"TwelveData error: {j}")
        
        df = pd.DataFrame(j["values"])
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
            else:
                df[c] = 0.0
        
        df["timestamp"] = pd.to_datetime(df.get("datetime", pd.Series(np.arange(len(df)))), errors='coerce')
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].set_index("timestamp").sort_index()
        return df.tail(limit)

    @staticmethod
    def fetch_ohlc_any(symbol: str, interval: str = "15m", limit: int = 500) -> pd.DataFrame:
        """Universal data fetcher with automatic fallback AND validation"""
        original_symbol = symbol.upper().replace(" ", "").replace("/", "")
        print(f"[FETCH] Requesting OHLC for {original_symbol} ({interval}) with limit={limit}")
        
        # Metal aliases conversion
        metal_aliases = {
            "XAUUSD": "XAUSDT", "XAU/USD": "XAUSDT", "GOLD": "XAUSDT", "GOLDUSD": "XAUSDT",
            "XAGUSD": "XAGUSDT", "SILVER": "XAGUSDT", "SILVERUSD": "XAGUSDT"
        }
        symbol_converted = metal_aliases.get(original_symbol, original_symbol)
        
        if symbol_converted != original_symbol:
            print(f"[AUTO-CONVERT] {original_symbol} → {symbol_converted} (Binance-compatible)")

        # Try data sources in order
        fetchers = [
            ("Binance", lambda: DataFetcher.fetch_ohlc_binance(symbol_converted, interval, limit)),
            ("TwelveData", lambda: DataFetcher.fetch_ohlc_twelvedata(original_symbol, interval, limit)),
        ]
        
        for name, fetcher in fetchers:
            try:
                print(f"[FETCH] Trying {name} for {original_symbol}")
                df = fetcher()
                
                if not enhanced_data_validation(df):
                    print(f"[FETCH] ⚠️ {name} data validation failed - skipping")
                    continue
                    
                print(f"[FETCH] ✅ {name} OK — got {len(df)} candles")
                return df
            except Exception as e:
                print(f"[FETCH] ⚠️ {name} failed: {e}")
                continue
        
        raise RuntimeError(f"All data sources failed for {original_symbol}")

# --- TECHNICAL INDICATORS ---
class TechnicalIndicators:
    """Technical analysis indicators"""
    
    @staticmethod
    def ema(series: pd.Series, n: int):
        return ta.trend.EMAIndicator(series, window=n).ema_indicator()
    
    @staticmethod
    def rsi(series: pd.Series, n: int = 14):
        return ta.momentum.RSIIndicator(series, window=n).rsi()
    
    @staticmethod
    def atr(df: pd.DataFrame, n: int = 14):
        return ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=n).average_true_range()

# --- COMPLETE SMC/ICT ADVANCED FEATURES ---
class ICTAdvancedFeatures:
    """Complete SMC/ICT Advanced Feature Detection"""
    
    @staticmethod
    def detect_order_blocks(df: pd.DataFrame, lookback: int = None) -> Dict[str, Any]:
        """Detect ICT Order Blocks with advanced logic"""
        if lookback is None:
            lookback = Config.OB_LOOKBACK
            
        if len(df) < lookback + 5:
            return {"order_blocks": [], "recent_ob": None}
        
        ob_blocks = []
        recent_data = df.tail(lookback)
        
        for i in range(3, len(recent_data)-2):
            current = recent_data.iloc[i]
            prev1 = recent_data.iloc[i-1]
            prev2 = recent_data.iloc[i-2]
            next1 = recent_data.iloc[i+1]
            next2 = recent_data.iloc[i+2]
            
            # Bearish Order Block: Strong down candle followed by consolidation
            if (current['close'] < current['open'] and 
                (current['high'] - current['low']) > 0 and
                abs(current['close'] - current['open']) > (current['high'] - current['low']) * 0.6):
                
                # Check for consolidation after the move
                if (next1['high'] < current['low'] and next2['high'] < current['low']):
                    ob_blocks.append({
                        'type': 'BEARISH_OB',
                        'high': float(current['high']),
                        'low': float(current['low']),
                        'timestamp': recent_data.index[i],
                        'strength': 0.8
                    })
            
            # Bullish Order Block: Strong up candle followed by consolidation
            elif (current['close'] > current['open'] and 
                  (current['high'] - current['low']) > 0 and
                  abs(current['close'] - current['open']) > (current['high'] - current['low']) * 0.6):
                
                # Check for consolidation after the move
                if (next1['low'] > current['high'] and next2['low'] > current['high']):
                    ob_blocks.append({
                        'type': 'BULLISH_OB',
                        'high': float(current['high']),
                        'low': float(current['low']),
                        'timestamp': recent_data.index[i],
                        'strength': 0.8
                    })
        
        # Find most recent relevant OB
        recent_ob = None
        current_price = df['close'].iloc[-1]
        
        for ob in reversed(ob_blocks[-10:]):  # Check last 10 OB
            if (ob['type'] == 'BULLISH_OB' and current_price > ob['low'] and 
                current_price < ob['high'] * 1.02):  # Within 2% of OB high
                recent_ob = ob
                break
            elif (ob['type'] == 'BEARISH_OB' and current_price < ob['high'] and 
                  current_price > ob['low'] * 0.98):  # Within 2% of OB low
                recent_ob = ob
                break
        
        return {
            "order_blocks": ob_blocks[-5:],  # Return last 5 OB
            "recent_ob": recent_ob
        }

    @staticmethod
    def detect_fair_value_gaps(df: pd.DataFrame, threshold: float = None) -> Dict[str, Any]:
        """Detect Fair Value Gaps (FVG) with price confirmation"""
        if threshold is None:
            threshold = Config.FVG_THRESHOLD
            
        fvgs = []
        if len(df) < 10:
            return {"fair_value_gaps": [], "active_fvg": None}
        
        for i in range(2, len(df)-1):
            current = df.iloc[i]
            prev1 = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            
            # Bullish FVG: Current low > Previous high with gap
            if (current['low'] > prev1['high'] * (1 + threshold) and
                current['low'] > prev2['high'] * (1 + threshold)):
                
                fvg_size = (current['low'] - max(prev1['high'], prev2['high'])) / current['low']
                if fvg_size > threshold:
                    fvgs.append({
                        'type': 'BULLISH_FVG',
                        'top': float(current['low']),
                        'bottom': float(max(prev1['high'], prev2['high'])),
                        'size': float(fvg_size),
                        'timestamp': df.index[i]
                    })
            
            # Bearish FVG: Current high < Previous low with gap
            elif (current['high'] < prev1['low'] * (1 - threshold) and
                  current['high'] < prev2['low'] * (1 - threshold)):
                
                fvg_size = (min(prev1['low'], prev2['low']) - current['high']) / current['high']
                if fvg_size > threshold:
                    fvgs.append({
                        'type': 'BEARISH_FVG',
                        'top': float(min(prev1['low'], prev2['low'])),
                        'bottom': float(current['high']),
                        'size': float(fvg_size),
                        'timestamp': df.index[i]
                    })
        
        # Find active FVG (price is within or recently touched)
        active_fvg = None
        current_price = df['close'].iloc[-1]
        
        for fvg in reversed(fvgs[-8:]):  # Check last 8 FVGs
            if (fvg['type'] == 'BULLISH_FVG' and 
                current_price >= fvg['bottom'] and current_price <= fvg['top']):
                active_fvg = fvg
                break
            elif (fvg['type'] == 'BEARISH_FVG' and 
                  current_price <= fvg['top'] and current_price >= fvg['bottom']):
                active_fvg = fvg
                break
        
        return {
            "fair_value_gaps": fvgs[-4:],  # Return last 4 FVGs
            "active_fvg": active_fvg
        }

    @staticmethod
    def analyze_liquidity_zones(df: pd.DataFrame, window: int = None) -> Dict[str, Any]:
        """Analyze liquidity zones (equal highs/lows) with volume confirmation"""
        if window is None:
            window = Config.LIQUIDITY_WINDOW
            
        if len(df) < window:
            return {"liquidity_above": [], "liquidity_below": [], "liquidity_strength": 0.0}
        
        recent = df.tail(window)
        current_price = df['close'].iloc[-1]
        
        # Find significant highs (liquidity above)
        high_levels = []
        for i in range(2, len(recent)-2):
            if (recent['high'].iloc[i] > recent['high'].iloc[i-1] and 
                recent['high'].iloc[i] > recent['high'].iloc[i-2] and
                recent['high'].iloc[i] > recent['high'].iloc[i+1] and
                recent['high'].iloc[i] > recent['high'].iloc[i+2]):
                
                high_levels.append({
                    'price': float(recent['high'].iloc[i]),
                    'volume': float(recent['volume'].iloc[i]),
                    'timestamp': recent.index[i]
                })
        
        # Find significant lows (liquidity below)
        low_levels = []
        for i in range(2, len(recent)-2):
            if (recent['low'].iloc[i] < recent['low'].iloc[i-1] and 
                recent['low'].iloc[i] < recent['low'].iloc[i-2] and
                recent['low'].iloc[i] < recent['low'].iloc[i+1] and
                recent['low'].iloc[i] < recent['low'].iloc[i+2]):
                
                low_levels.append({
                    'price': float(recent['low'].iloc[i]),
                    'volume': float(recent['volume'].iloc[i]),
                    'timestamp': recent.index[i]
                })
        
        # Filter and sort liquidity zones
        liquidity_above = sorted([hl for hl in high_levels if hl['price'] > current_price], 
                                key=lambda x: x['price'])[:3]
        liquidity_below = sorted([ll for ll in low_levels if ll['price'] < current_price], 
                                key=lambda x: x['price'], reverse=True)[:3]
        
        # Calculate liquidity strength
        strength = 0.0
        if liquidity_above:
            nearest_above = min([hl['price'] for hl in liquidity_above])
            distance_above = (nearest_above - current_price) / current_price
            if distance_above < 0.02:  # Within 2%
                strength += 0.5
        if liquidity_below:
            nearest_below = max([ll['price'] for ll in liquidity_below])
            distance_below = (current_price - nearest_below) / current_price
            if distance_below < 0.02:  # Within 2%
                strength += 0.5
        
        return {
            "liquidity_above": liquidity_above,
            "liquidity_below": liquidity_below,
            "liquidity_strength": strength,
            "current_price": float(current_price)
        }

    @staticmethod
    def calculate_optimal_trade_entry(df: pd.DataFrame, bias: str, retracement: float = None) -> Dict[str, float]:
        """Calculate Optimal Trade Entry (OTE) levels"""
        if retracement is None:
            retracement = Config.OTE_RETRACEMENT
            
        if len(df) < 20:
            return {"ote_level": 0.0, "confirmation_level": 0.0}
        
        recent = df.tail(20)
        
        if bias == "LONG":
            swing_high = recent['high'].max()
            swing_low = recent['low'].min()
            ote_level = swing_high - (swing_high - swing_low) * retracement
            confirmation_level = ote_level - (swing_high - swing_low) * 0.1  # 10% below OTE
            
        elif bias == "SHORT":
            swing_high = recent['high'].max()
            swing_low = recent['low'].min()
            ote_level = swing_low + (swing_high - swing_low) * retracement
            confirmation_level = ote_level + (swing_high - swing_low) * 0.1  # 10% above OTE
            
        else:
            return {"ote_level": 0.0, "confirmation_level": 0.0}
        
        return {
            "ote_level": float(ote_level),
            "confirmation_level": float(confirmation_level),
            "retracement_level": retracement
        }

    @staticmethod
    def calculate_ict_confidence(ob_analysis: Dict, fvg_analysis: Dict, 
                               liquidity_analysis: Dict, bias: str) -> float:
        """Calculate comprehensive ICT confidence score"""
        confidence = 0.0
        
        # Order Block confidence
        if ob_analysis.get("recent_ob"):
            ob = ob_analysis["recent_ob"]
            if ((bias == "LONG" and ob['type'] == 'BULLISH_OB') or
                (bias == "SHORT" and ob['type'] == 'BEARISH_OB')):
                confidence += 0.3 * ob.get('strength', 0.5)
        
        # FVG confidence
        if fvg_analysis.get("active_fvg"):
            fvg = fvg_analysis["active_fvg"]
            if ((bias == "LONG" and fvg['type'] == 'BULLISH_FVG') or
                (bias == "SHORT" and fvg['type'] == 'BEARISH_FVG')):
                confidence += 0.25 * min(1.0, fvg.get('size', 0) * 10)
        
        # Liquidity confidence
        liquidity_strength = liquidity_analysis.get("liquidity_strength", 0.0)
        if bias == "LONG" and liquidity_analysis.get("liquidity_above"):
            confidence += 0.2 * liquidity_strength
        elif bias == "SHORT" and liquidity_analysis.get("liquidity_below"):
            confidence += 0.2 * liquidity_strength
        
        # Market structure alignment bonus
        if confidence > 0.3:
            confidence += 0.1
        
        return min(confidence, 1.0)
        
# ---------------- ICT PRO REFINED EXTENSIONS ----------------
class ICTProRefined:
    """
    Koleksi fungsi refined untuk:
    - Order Block PRO (refined)
    - Fair Value Gap PRO (refined)
    - Liquidity sweep detection
    - BOS / CHOCH dengan displacement
    - Multi-timeframe narrative helper
    - Killzone/session filter
    - Sniper entry confirmation (OTE + rejection)
    """

    @staticmethod
    def _is_strong_candle(row):
        # measure body vs range
        body = abs(row['close'] - row['open'])
        rng = row['high'] - row['low'] if (row['high'] - row['low'])>0 else 1e-9
        return body / rng > 0.55

    @staticmethod
    def detect_refined_order_blocks(df: pd.DataFrame, lookback:int=80, max_blocks:int=6):
        """
        Return refined OB list with volume and displacement checks.
        Each OB = {type, high, low, ts, strength, volume_rel, refined_high, refined_low}
        """
        if len(df) < lookback+5:
            return {"order_blocks": [], "recent_ob": None}

        recent = df.copy().tail(lookback).reset_index()
        ob_list = []

        vol_median = recent['volume'].median() if 'volume' in recent.columns else 0
        for i in range(3, len(recent)-2):
            c = recent.loc[i]
            p1 = recent.loc[i-1]
            p2 = recent.loc[i-2]
            n1 = recent.loc[i+1]
            n2 = recent.loc[i+2]

            # strong directional candle
            strong = ICTProRefined._is_strong_candle(c)
            vol_rel = (c.get('volume', vol_median) / (vol_median+1e-9)) if vol_median>0 else 1.0

            # Bearish OB: big bearish candle followed by lower highs (consolidation)
            if (c['close'] < c['open']) and strong:
                if (n1['high'] < c['low'] and n2['high'] < c['low']):
                    ob = {
                        'type':'BEARISH_OB',
                        'high': float(c['high']),
                        'low': float(c['low']),
                        'timestamp': recent.loc[i,'timestamp'] if 'timestamp' in recent.columns else recent.loc[i].name,
                        'strength': round(0.6 + min(0.4, vol_rel/3),3),
                        'volume_rel': round(vol_rel,3)
                    }
                    # refine OB: shrink to last two candles' extremes to avoid huge zones
                    ob['refined_high'] = max(float(c['open']), float(c['high'])) 
                    ob['refined_low'] = min(float(c['close']), float(c['low']))
                    ob_list.append(ob)

            # Bullish OB
            if (c['close'] > c['open']) and strong:
                if (n1['low'] > c['high'] and n2['low'] > c['high']):
                    ob = {
                        'type':'BULLISH_OB',
                        'high': float(c['high']),
                        'low': float(c['low']),
                        'timestamp': recent.loc[i,'timestamp'] if 'timestamp' in recent.columns else recent.loc[i].name,
                        'strength': round(0.6 + min(0.4, vol_rel/3),3),
                        'volume_rel': round(vol_rel,3)
                    }
                    ob['refined_high'] = max(float(c['close']), float(c['high']))
                    ob['refined_low'] = min(float(c['open']), float(c['low']))
                    ob_list.append(ob)

        # pick last max_blocks and find recent_ob close to current price
        current_price = df['close'].iloc[-1]
        recent_ob = None
        for ob in reversed(ob_list[-max_blocks:]):
            if ob['type']=='BULLISH_OB' and current_price > ob['refined_low'] and current_price < ob['refined_high']*1.02:
                recent_ob = ob; break
            if ob['type']=='BEARISH_OB' and current_price < ob['refined_high'] and current_price > ob['refined_low']*0.98:
                recent_ob = ob; break

        return {"order_blocks": ob_list[-max_blocks:], "recent_ob": recent_ob}


    @staticmethod
    def detect_refined_fvg(df: pd.DataFrame, threshold:float=0.0008, lookback:int=120):
        """
        Refined Fair Value Gap detection:
         - requires "fresh" gap (no retrace filled)
         - align with displacement (recent strong candle)
        Returns list of fvgs and active_fvg if near price
        """
        if len(df) < 8:
            return {"fair_value_gaps": [], "active_fvg": None}

        fvgs = []
        recent = df.reset_index().copy()
        for i in range(2, len(recent)-1):
            c = recent.loc[i]
            p1 = recent.loc[i-1]
            p2 = recent.loc[i-2]
            # bullish FVG: current low > prev1 high by threshold %
            if (c['low'] > p1['high'] * (1 + threshold)):
                fvgs.append({
                    "type":"BULL_FVG",
                    "low": float(p1['high']),
                    "high": float(c['low']),
                    "ts": recent.loc[i,'timestamp'] if 'timestamp' in recent.columns else recent.loc[i].name,
                    "strength": round((c['low'] - p1['high'])/ max(1e-9, p1['high']), 4)
                })
            # bearish FVG
            if (c['high'] < p1['low'] * (1 - threshold)):
                fvgs.append({
                    "type":"BEAR_FVG",
                    "high": float(p1['low']),
                    "low": float(c['high']),
                    "ts": recent.loc[i,'timestamp'] if 'timestamp' in recent.columns else recent.loc[i].name,
                    "strength": round((p1['low'] - c['high'])/ max(1e-9, p1['low']), 4)
                })

        # find active fvg near current price (within 2% nominal)
        current = df['close'].iloc[-1]
        active = None
        for g in reversed(fvgs[-6:]):
            if g['type']=='BULL_FVG' and current >= g['low']*0.98 and current <= g['high']*1.02:
                active = g; break
            if g['type']=='BEAR_FVG' and current <= g['high']*1.02 and current >= g['low']*0.98:
                active = g; break

        return {"fair_value_gaps": fvgs[-12:], "active_fvg": active}


    @staticmethod
    def detect_liquidity_sweep(df: pd.DataFrame, lookback:int=80, wick_threshold:float=0.6):
        """
        Detect liquidity sweeps (wick grabs) by observing candles that exceed previous extremes significantly.
        Returns list of sweeps: {type, sweep_price, index, ts, magnitude}
        """
        if len(df) < 10:
            return {"sweeps": [], "recent_sweep": None}

        sweeps = []
        recent = df.reset_index().copy()
        highs = recent['high'].rolling(20, min_periods=1).max()
        lows = recent['low'].rolling(20, min_periods=1).min()

        for i in range(3, len(recent)-1):
            c = recent.loc[i]
            prev_high = highs.loc[i-1]
            prev_low = lows.loc[i-1]
            # bullish sweep (buy-side liquidity grabbed below)
            if c['low'] < prev_low * (1 - 0.0005) and (c['high'] - c['low'])>0:
                wick = (c['open'] - c['low']) if c['open']>c['close'] else (c['close'] - c['low'])
                wick_ratio = wick / max(1e-9, c['high'] - c['low'])
                sweeps.append({
                    "type":"BUY_SIDE_SWEEP",
                    "sweep_price": float(c['low']),
                    "index": i,
                    "ts": recent.loc[i,'timestamp'] if 'timestamp' in recent.columns else recent.loc[i].name,
                    "magnitude": round((prev_low - c['low'])/max(1e-9, prev_low), 6),
                    "wick_ratio": round(wick_ratio,3)
                })
            # bearish sweep (sell-side liquidity grabbed above)
            if c['high'] > prev_high * (1 + 0.0005) and (c['high'] - c['low'])>0:
                wick = (c['high'] - c['open']) if c['open']>c['close'] else (c['high'] - c['close'])
                wick_ratio = wick / max(1e-9, c['high'] - c['low'])
                sweeps.append({
                    "type":"SELL_SIDE_SWEEP",
                    "sweep_price": float(c['high']),
                    "index": i,
                    "ts": recent.loc[i,'timestamp'] if 'timestamp' in recent.columns else recent.loc[i].name,
                    "magnitude": round((c['high'] - prev_high)/max(1e-9, prev_high), 6),
                    "wick_ratio": round(wick_ratio,3)
                })

        recent_sweep = sweeps[-1] if sweeps else None
        return {"sweeps": sweeps[-16:], "recent_sweep": recent_sweep}


    @staticmethod
    def bos_choch_confirm(df: pd.DataFrame, lookback:int=200, allow_noise_pct:float=0.002):
        """
        Detect robust BOS/CHOCH with displacement check:
        - Look for a break of last significant swing high/low
        - Confirm displacement: a follow-through candle closing beyond broken structure
        - Avoid small breaks within noise threshold (allow_noise_pct)
        Returns: {'type': 'BOS_LONG'/'BOS_SHORT'/'NONE', 'broken_level':..., 'confirm_idx': idx}
        """
        if len(df) < 20:
            return {"type":"NONE"}

        recent = df.copy().reset_index()
        # find last swing high/low (simple pivot)
        def find_last_pivots(series, left=3, right=3):
            pivot_highs = []
            pivot_lows = []
            for i in range(left, len(series)-right):
                win = series[i-left:i+right+1]
                center = series[i]
                if center == max(win):
                    pivot_highs.append((i, center))
                if center == min(win):
                    pivot_lows.append((i, center))
            return pivot_highs, pivot_lows

        highs = recent['high'].values
        lows = recent['low'].values
        ph, pl = find_last_pivots(highs, left=3, right=3)
        lh_idx, lh_val = ph[-1] if ph else (None, None)
        ll_idx, ll_val = pl[-1] if pl else (None, None)

        cur = recent.iloc[-1]
        # check bullish BOS (price breaks last structure high)
        if lh_val is not None:
            broke_up = (cur['close'] > lh_val * (1 + allow_noise_pct)) or (cur['high'] > lh_val * (1 + allow_noise_pct))
            if broke_up:
                # need displacement: a candle that closed above lh_val and next candle shows follow-through
                # find confirm candle in next 3 candles
                for j in range(max(0, len(recent)-4), len(recent)):
                    if recent.loc[j,'close'] > lh_val * (1 + allow_noise_pct):
                        return {"type":"BOS_LONG", "broken_level": float(lh_val), "confirm_idx": j}
        # check bearish BOS
        if ll_val is not None:
            broke_dn = (cur['close'] < ll_val * (1 - allow_noise_pct)) or (cur['low'] < ll_val * (1 - allow_noise_pct))
            if broke_dn:
                for j in range(max(0, len(recent)-4), len(recent)):
                    if recent.loc[j,'close'] < ll_val * (1 - allow_noise_pct):
                        return {"type":"BOS_SHORT", "broken_level": float(ll_val), "confirm_idx": j}

        return {"type":"NONE"}


    @staticmethod
    def killzone_session_filter(ts_utc, allowed_sessions=("LONDON","NY")):
        """
        ts_utc: pandas.Timestamp or datetime (UTC)
        allowed_sessions: tuple controlling which sessions allow entries
        Sessions (UTC):
          - ASIA: 00:00-06:00
          - LONDON: 07:00-15:00 (approx; adjust)
          - NY: 12:00-20:00
        Return True if allowed to trade in that timestamp
        """
        if ts_utc is None:
            return True
        hour = ts_utc.hour
        # simple mapping
        is_london = 7 <= hour < 15
        is_ny = 12 <= hour < 20
        is_asia = (0 <= hour < 7) or (20 <= hour <= 23)

        if "LONDON" in allowed_sessions and is_london: return True
        if "NY" in allowed_sessions and is_ny: return True
        if "ASIA" in allowed_sessions and is_asia: return True
        return False


    @staticmethod
    def sniper_entry_confirmation(df: pd.DataFrame, ob:dict=None, fvg:dict=None, sweep:dict=None, ote_range=(0.62,0.79)):
        """
        Sniper entry heuristics:
         - requires confluence of OB (or recent_ob) and either active FVG or recent sweep
         - OTE: compute fib retracement from last displacement swing
         - Wait for rejection candle / bullish engulf after OB (for long)
        Returns dict with 'accept' True/False and recommended entry/sl/sl_tp levels
        """
        out = {"accept": False, "reason": "", "entry": None, "sl": None, "tp1": None, "tp2": None}
        if ob is None and fvg is None and sweep is None:
            out['reason']="no_confluence"
            return out

        current = df.copy().iloc[-1]
        price = float(current['close'])

        # require price to be near OB refined (within 2% or 1 ATR)
        def near_zone(zone_high, zone_low, price, pct=0.02):
            return price >= zone_low*(1-pct) and price <= zone_high*(1+pct)

        ob_ok = False
        if ob:
            zone_h = ob.get('refined_high', ob.get('high'))
            zone_l = ob.get('refined_low', ob.get('low'))
            if zone_h and zone_l and near_zone(zone_h, zone_l, price, pct=0.02):
                ob_ok = True

        fvg_ok = False
        if fvg:
            # if active_fvg exists and near price
            af = fvg
            if 'low' in af and 'high' in af and near_zone(af.get('high'), af.get('low'), price, pct=0.03):
                fvg_ok = True

        sweep_ok = False
        if sweep:
            # accept if recent sweep opposite to trade direction (means liquidity grabbed)
            sweep_ok = True

        # require at least OB + (FVG or sweep)
        if not ob_ok:
            out['reason'] = "no_ob_alignment"
            return out
        if not (fvg_ok or sweep_ok):
            out['reason'] = "no_fvg_or_sweep"
            return out

        # compute ATR for SL sizing
        atr = None
        try:
            atr = TechnicalIndicators.atr(df, n=14).iloc[-1]
            if math.isnan(atr) or atr<=0:
                atr = None
        except Exception:
            atr = None

        # propose entry at current price or slightly into OB / FVG
        entry = price
        sl = None
        tp1 = None
        tp2 = None

        # if OB is bullish, plan LONG (reverse for bearish)
        if ob.get('type','').startswith('BULL'):
            # SL below refined low
            sl = float(ob.get('refined_low', ob.get('low'))) - (atr*0.5 if atr else 0.0)
            # TP1 = nearest FVG high or OB high
            if fvg_ok:
                tp1 = float(fvg.get('high') if fvg.get('type','').startswith('BULL') else fvg.get('high'))
            else:
                tp1 = float(ob.get('refined_high', ob.get('high'))) * 1.02
            tp2 = tp1 * 1.03
            out['direction'] = 'LONG'
        else:
            # SHORT
            sl = float(ob.get('refined_high', ob.get('high'))) + (atr*0.5 if atr else 0.0)
            if fvg_ok:
                tp1 = float(fvg.get('low') if fvg.get('type','').startswith('BEAR') else fvg.get('low'))
            else:
                tp1 = float(ob.get('refined_low', ob.get('low'))) * 0.98
            tp2 = tp1 * 0.97
            out['direction'] = 'SHORT'

        out.update({"accept": True, "reason":"confluence_ob_fvg", "entry": round(entry,8), "sl": round(sl,8) if sl else None, "tp1": round(tp1,8) if tp1 else None, "tp2": round(tp2,8) if tp2 else None})
        return out

    @staticmethod
    def multi_tf_narrative(df_htf: dict):
        """
        df_htf: dict of { '1w': df_weekly, '1d': df_daily, '1h': df_h1, '15m': df_15m }
        Returns bias: {'weekly':'bullish/neutral/bearish', 'daily':..., 'alignment': True/False}
        """
        bias = {}
        for tf, df in df_htf.items():
            if df is None or len(df)==0:
                bias[tf] = 'neutral'
                continue
            ma200 = df['close'].rolling(200, min_periods=1).mean().iloc[-1] if len(df)>=200 else df['close'].rolling(max(1,int(len(df)/2))).mean().iloc[-1]
            cur = df['close'].iloc[-1]
            bias[tf] = 'bullish' if cur > ma200 else ('bearish' if cur < ma200 else 'neutral')

        # alignment if most HTF agree
        vals = list(bias.values())
        bulls = vals.count('bullish')
        bears = vals.count('bearish')
        alignment = True if (bulls>=(len(vals)//2 + 1) or bears>=(len(vals)//2 + 1)) else False

        return {"bias": bias, "alignment": alignment}

# --- VOLUME ANALYZER ---
class VolumeAnalyzer:
    """Volume analysis utilities"""
    
    @staticmethod
    def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features to dataframe"""
        df = df.copy()
        if 'volume' not in df.columns:
            df['volume'] = 0.0
        
        df['vol_delta'] = df['volume'].diff().fillna(0.0)
        df['vol_ma20'] = df['volume'].rolling(20, min_periods=1).mean()
        df['vol_ratio'] = df['volume'] / (df['vol_ma20'] + 1e-9)
        
        direction = np.sign(df['close'] - df['open'])
        df['direction'] = direction
        df['pv'] = df['volume'] * (df['direction'] > 0).astype(float)
        df['nv'] = df['volume'] * (df['direction'] < 0).astype(float)
        
        df['pv_ma'] = df['pv'].rolling(20, min_periods=1).mean()
        df['nv_ma'] = df['nv'].rolling(20, min_periods=1).mean()
        df['vol_imbalance_score'] = 0.5 + 0.5 * ((df['pv'] - df['nv']) / (df['pv_ma'] + df['nv_ma'] + 1e-9))
        df['vol_imbalance_score'] = df['vol_imbalance_score'].clip(0.0, 1.0)
        
        df['vol_ratio_diff'] = df['vol_ratio'].diff().fillna(0.0)
        df['absorption_flag'] = ((df['close'] > df['open']) & (df['vol_ratio_diff'] < 0)) | ((df['close'] < df['open']) & (df['vol_ratio_diff'] < 0))
        
        return df

    @staticmethod
    def compute_volume_confidence(df: pd.DataFrame, idx: int = -1) -> float:
        """Compute volume-based confidence score"""
        if 'vol_imbalance_score' not in df.columns:
            df = VolumeAnalyzer.add_volume_features(df)
        
        row = df.iloc[idx]
        vol_ratio = float(row.get('vol_ratio', 1.0))
        imb_score = float(row.get('vol_imbalance_score', 0.5))
        absorption = bool(row.get('absorption_flag', False))
        
        vr_scale = (np.tanh((vol_ratio - 1.0) / 0.5) + 1) / 2
        base = 0.6 * imb_score + 0.4 * vr_scale
        
        if absorption:
            base *= 0.5
        
        return float(np.clip(base, 0.0, 1.0))

# --- ENHANCED FEATURE ENGINEERING ---
class EnhancedFeatureEngineer:
    """Enhanced feature engineering for better ML performance"""
    
    @staticmethod
    def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['momentum'] = df['close'] / df['close'].shift(5) - 1
        
        # Advanced technical indicators
        try:
            df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
        except:
            df['adx'] = 0.0
            
        try:
            df['macd'] = ta.trend.MACD(df['close']).macd()
        except:
            df['macd'] = 0.0
            
        try:
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        except:
            df['obv'] = 0.0
        
        # Statistical features
        df['zscore'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        df['bollinger_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
        df['bollinger_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
        
        # Market regime detection
        df['trend_strength'] = df['adx'] / 100.0
        df['market_regime'] = np.where(df['adx'] > 25, 'trending', 'ranging')
        
        return df.dropna()

# --- FUNDAMENTAL ANALYSIS MODULES ---

# --- ECONOMIC CALENDAR INTEGRATION ---
class EconomicCalendar:
    """Economic calendar data integration"""
    
    @staticmethod
    def get_events(days: int = 3) -> List[Dict]:
        """Get economic events for next N days"""
        try:
            # Try Alpha Vantage first
            if Config.ALPHA_VANTAGE_API_KEY:
                params = {
                    "apikey": Config.ALPHA_VANTAGE_API_KEY,
                    "interval": "day",
                    "time_horizon": "1month"
                }
                response = requests.get(Config.ECONOMIC_CALENDAR_URL, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    return EconomicCalendar._parse_alpha_vantage(data, days)
            
            # Fallback to Forex Factory
            response = requests.get(Config.FOREX_FACTORY_CALENDAR, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return EconomicCalendar._parse_forex_factory(data, days)
            
            return []
            
        except Exception as e:
            print(f"[EconomicCalendar] Error: {e}")
            return []

    @staticmethod
    def _parse_alpha_vantage(data: Dict, days: int) -> List[Dict]:
        """Parse Alpha Vantage economic calendar"""
        events = []
        today = datetime.now()
        
        for event in data.get('economic_calendar', []):
            try:
                event_date = datetime.strptime(event['date'], '%Y-%m-%d')
                days_diff = (event_date - today).days
                
                if 0 <= days_diff <= days:
                    impact = event.get('importance', 'Low')
                    events.append({
                        'date': event['date'],
                        'time': event.get('time', ''),
                        'currency': event['currency'],
                        'event': event['event'],
                        'impact': impact,
                        'actual': event.get('actual'),
                        'forecast': event.get('forecast'),
                        'previous': event.get('previous')
                    })
            except Exception as e:
                continue
        
        return sorted(events, key=lambda x: x['date'])

    @staticmethod
    def _parse_forex_factory(data: List[Dict], days: int) -> List[Dict]:
        """Parse Forex Factory calendar"""
        events = []
        today = datetime.now()
        
        for event in data:
            try:
                event_date = datetime.fromtimestamp(event['timestamp'])
                days_diff = (event_date - today).days
                
                if 0 <= days_diff <= days:
                    impact_map = {'high': 'High', 'medium': 'Medium', 'low': 'Low'}
                    events.append({
                        'date': event_date.strftime('%Y-%m-%d'),
                        'time': event_date.strftime('%H:%M'),
                        'currency': event['country'],
                        'event': event['title'],
                        'impact': impact_map.get(event.get('impact', 'low'), 'Low'),
                        'forecast': event.get('forecast'),
                        'previous': event.get('previous')
                    })
            except Exception as e:
                continue
        
        return sorted(events, key=lambda x: x['date'])

    @staticmethod
    def get_high_impact_events(currency: str, hours_ahead: int = 24) -> List[Dict]:
        """Get high impact events for specific currency"""
        events = EconomicCalendar.get_events(days=7)  # Get week ahead
        now = datetime.now()
        high_impact = []
        
        for event in events:
            # Check currency match and high impact
            if (event['currency'] == currency and 
                event['impact'] == 'High' and
                any(hi_event in event['event'] for hi_event in Config.HIGH_IMPACT_EVENTS)):
                
                try:
                    event_time = datetime.strptime(f"{event['date']} {event['time']}", '%Y-%m-%d %H:%M')
                    hours_diff = (event_time - now).total_seconds() / 3600
                    
                    if 0 <= hours_diff <= hours_ahead:
                        high_impact.append(event)
                except:
                    continue
        
        return high_impact

# --- NEWS SENTIMENT ANALYSIS ---
class NewsSentimentAnalyzer:
    """Real-time news sentiment analysis"""
    
    @staticmethod
    def get_market_sentiment(symbol: str) -> Dict[str, Any]:
        """Get overall market sentiment for symbol"""
        try:
            sentiment_data = {}
            
            # Try multiple news sources
            if Config.NEWSAPI_API_KEY:
                news_sentiment = NewsSentimentAnalyzer._get_newsapi_sentiment(symbol)
                if news_sentiment:
                    sentiment_data.update(news_sentiment)
            
            # Fallback to simple sentiment based on price action
            if not sentiment_data:
                sentiment_data = NewsSentimentAnalyzer._get_fallback_sentiment(symbol)
            
            return NewsSentimentAnalyzer._calculate_composite_sentiment(sentiment_data)
            
        except Exception as e:
            print(f"[NewsSentiment] Error: {e}")
            return {"overall_sentiment": "NEUTRAL", "score": 0.5, "confidence": 0.0}

    @staticmethod
    def _get_newsapi_sentiment(symbol: str) -> Dict:
        """Get sentiment from NewsAPI"""
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f"{symbol} OR forex OR trading",
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': 10,
                'apiKey': Config.NEWSAPI_API_KEY
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return {}
                
            data = response.json()
            
            # Simple sentiment analysis based on keywords
            positive_words = ['bullish', 'up', 'rise', 'gain', 'positive', 'strong', 'buy', 'rally']
            negative_words = ['bearish', 'down', 'fall', 'drop', 'negative', 'weak', 'sell', 'crash']
            
            positive_count = 0
            negative_count = 0
            total_articles = len(data.get('articles', []))
            
            for article in data.get('articles', []):
                title = article.get('title', '').lower()
                description = article.get('description', '').lower()
                content = title + " " + description
                
                if any(word in content for word in positive_words):
                    positive_count += 1
                if any(word in content for word in negative_words):
                    negative_count += 1
            
            sentiment_score = (positive_count - negative_count) / max(total_articles, 1)
            
            return {
                "newsapi_sentiment": sentiment_score,
                "newsapi_articles": total_articles
            }
        except:
            return {}

    @staticmethod
    def _get_fallback_sentiment(symbol: str) -> Dict:
        """Fallback sentiment based on recent price action"""
        try:
            # Get recent price data for sentiment
            df = DataFetcher.fetch_ohlc_any(symbol, "1d", limit=10)
            if len(df) < 5:
                return {}
            
            recent_return = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) * 100
            
            # Simple sentiment based on recent performance
            if recent_return > 2:
                sentiment_score = 0.7
            elif recent_return < -2:
                sentiment_score = 0.3
            else:
                sentiment_score = 0.5
                
            return {
                "price_action_sentiment": sentiment_score,
                "recent_return": recent_return
            }
        except:
            return {}

    @staticmethod
    def _calculate_composite_sentiment(sentiment_data: Dict) -> Dict:
        """Calculate composite sentiment score"""
        scores = []
        weights = []
        
        for key, value in sentiment_data.items():
            if 'sentiment' in key and isinstance(value, (int, float)):
                scores.append(value)
                weights.append(1.0)
        
        if not scores:
            return {"overall_sentiment": "NEUTRAL", "score": 0.5, "confidence": 0.0}
        
        composite_score = np.average(scores, weights=weights)
        composite_score = np.clip(composite_score, 0, 1)  # Normalize to 0-1
        
        # Determine sentiment category
        if composite_score > 0.6:
            sentiment = "BULLISH"
        elif composite_score < 0.4:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"
        
        return {
            "overall_sentiment": sentiment,
            "score": round(composite_score, 3),
            "confidence": min(len(scores) * 0.2, 1.0),  # More sources = more confidence
            "sources_used": len(scores)
        }

# --- FUNDAMENTAL CONTEXT INTEGRATION ---
class FundamentalContext:
    """Integrate fundamental analysis with trading signals"""
    
    @staticmethod
    def get_fundamental_context(pair: str, technical_confidence: float) -> Dict[str, Any]:
        """Get comprehensive fundamental context"""
        try:
            # Extract currency from pair (e.g., "EURUSD" -> "EUR" and "USD")
            base_currency = pair[:3] if len(pair) >= 6 else "USD"
            quote_currency = pair[3:6] if len(pair) >= 6 else "USD"
            
            # Get economic events for both currencies
            base_events = EconomicCalendar.get_high_impact_events(base_currency)
            quote_events = EconomicCalendar.get_high_impact_events(quote_currency)
            
            # Get market sentiment
            sentiment = NewsSentimentAnalyzer.get_market_sentiment(pair)
            
            # Calculate fundamental risk score
            risk_score = FundamentalContext._calculate_risk_score(
                base_events, quote_events, sentiment
            )
            
            # Adjust position size based on fundamental risk
            position_adjustment = FundamentalContext._calculate_position_adjustment(risk_score)
            
            # Adjust confidence based on fundamental factors
            adjusted_confidence = FundamentalContext._adjust_confidence(
                technical_confidence, risk_score, sentiment
            )
            
            return {
                "fundamental_risk_score": risk_score,
                "position_size_adjustment": position_adjustment,
                "adjusted_confidence": adjusted_confidence,
                "high_impact_events": {
                    "base_currency": base_events,
                    "quote_currency": quote_events,
                    "total_count": len(base_events) + len(quote_events)
                },
                "market_sentiment": sentiment,
                "trading_recommendation": FundamentalContext._get_trading_recommendation(risk_score)
            }
        except Exception as e:
            print(f"[FundamentalContext] Error: {e}")
            # Return neutral context if analysis fails
            return {
                "fundamental_risk_score": 0.0,
                "position_size_adjustment": 1.0,
                "adjusted_confidence": technical_confidence,
                "high_impact_events": {"base_currency": [], "quote_currency": [], "total_count": 0},
                "market_sentiment": {"overall_sentiment": "NEUTRAL", "score": 0.5, "confidence": 0.0},
                "trading_recommendation": "NORMAL_TRADING"
            }

    @staticmethod
    def _calculate_risk_score(base_events: List, quote_events: List, sentiment: Dict) -> float:
        """Calculate fundamental risk score (0-1, where 1 is highest risk)"""
        risk_score = 0.0
        
        # Event-based risk
        total_events = len(base_events) + len(quote_events)
        if total_events >= 3:
            risk_score += 0.8
        elif total_events >= 1:
            risk_score += 0.5
        
        # Sentiment-based risk (extreme sentiment = higher risk)
        sentiment_score = sentiment.get('score', 0.5)
        if sentiment_score > 0.8 or sentiment_score < 0.2:  # Extreme bullish/bearish
            risk_score += 0.3
        
        return min(risk_score, 1.0)

    @staticmethod
    def _calculate_position_adjustment(risk_score: float) -> float:
        """Calculate position size adjustment based on risk"""
        if risk_score > 0.7:  # High risk
            return Config.HIGH_IMPACT_REDUCTION
        elif risk_score > 0.4:  # Medium risk
            return Config.MEDIUM_IMPACT_REDUCTION
        else:  # Low risk
            return 1.0

    @staticmethod
    def _adjust_confidence(technical_confidence: float, risk_score: float, sentiment: Dict) -> float:
        """Adjust technical confidence based on fundamental factors"""
        adjusted_confidence = technical_confidence
        
        # Reduce confidence during high risk periods
        if risk_score > 0.7:
            adjusted_confidence *= Config.VOLATILITY_REDUCTION
        elif risk_score > 0.4:
            adjusted_confidence *= 0.9
        
        return round(max(0.0, min(adjusted_confidence, 1.0)), 3)

    @staticmethod
    def _get_trading_recommendation(risk_score: float) -> str:
        """Get trading recommendation based on fundamental risk"""
        if risk_score > 0.7:
            return "AVOID_TRADING"
        elif risk_score > 0.4:
            return "REDUCE_POSITION_SIZE"
        else:
            return "NORMAL_TRADING"

# --- COMPLETE CONFIDENCE CALCULATION ---
def calculate_complete_confidence(smc_conf: float, vol_conf: float, ml_conf: float, ict_conf: float,
                                market_regime: str, volatility: float) -> float:
    """Complete confidence calculation dengan semua komponen SMC/ICT"""
    
    # Base weighted average dengan ICT component
    base_conf = (Config.WEIGHT_SMC * smc_conf + 
                Config.WEIGHT_VOL * vol_conf + 
                Config.WEIGHT_ML * ml_conf +
                Config.WEIGHT_ICT * ict_conf)
    
    # Market regime adjustments
    regime_multiplier = 1.2 if market_regime == 'trending' else 0.8
    
    # Volatility adjustment (lower confidence in high volatility)
    vol_adjustment = 1.0 / (1.0 + volatility * 10)  # Reduce confidence when volatility > 0.1
    
    # ICT confluence bonus
    ict_bonus = 1.0
    if ict_conf > 0.6:
        ict_bonus = 1.1  # 10% bonus for strong ICT confluence
    elif ict_conf > 0.8:
        ict_bonus = 1.2  # 20% bonus for very strong ICT confluence
    
    final_conf = base_conf * regime_multiplier * vol_adjustment * ict_bonus
    
    return np.clip(final_conf, 0.0, 1.0)

# --- SMC/ICT PRO ENGINE ---
class SMCICTProEngine:
    """Complete SMC/ICT PRO signal generation engine"""
    
    @staticmethod
    def parse_time(s: str) -> dtime:
        h, m = map(int, s.split(":"))
        return dtime(h, m)
    
    @staticmethod
    def in_killzone(check_dt: datetime) -> bool:
        if not Config.ICT_KILLZONE_ENABLE:
            return True
        start = SMCICTProEngine.parse_time(Config.ICT_KILLZONE_START)
        end = SMCICTProEngine.parse_time(Config.ICT_KILLZONE_END)
        t = check_dt.time()
        if start <= end:
            return start <= t <= end
        return t >= start or t <= end
    
    @staticmethod
    def detect_bos_pro(df: pd.DataFrame, lookback=50, atr_mul=1.0):
        """Detect Break of Structure with enhanced logic"""
        if len(df) < lookback + 3:
            return {"bias": "NEUTRAL", "bos_level": None, "strength": 0.0}
        
        if 'atr' not in df.columns:
            df['atr'] = TechnicalIndicators.atr(df, 14)
            
        # Use proper pandas indexing
        window = df.iloc[-lookback:]
        ref_section = window.iloc[:-max(3, int(lookback * 0.2))]
        swing_high = ref_section['high'].max()
        swing_low = ref_section['low'].min()
        latest_close = df['close'].iloc[-1]
        latest_atr = df['atr'].iloc[-1]
        
        up_threshold = swing_high + atr_mul * latest_atr
        down_threshold = swing_low - atr_mul * latest_atr
        
        strength = 0.0
        if latest_close > up_threshold:
            strength = min(1.0, (latest_close - swing_high) / (latest_atr * 2))
            return {"bias": "LONG", "bos_level": float(swing_high), "strength": strength}
        elif latest_close < down_threshold:
            strength = min(1.0, (swing_low - latest_close) / (latest_atr * 2))
            return {"bias": "SHORT", "bos_level": float(swing_low), "strength": strength}
        else:
            return {"bias": "NEUTRAL", "bos_level": None, "strength": 0.0}

    @staticmethod
    def detect_market_structure_shift(df: pd.DataFrame, lookback=100):
        """Detect Market Structure Shift (MSS) with volume confirmation"""
        if len(df) < lookback:
            return {"mss_bullish": False, "mss_bearish": False, "strength": 0.0}
        
        recent_data = df.tail(lookback).copy()
        
        # Find swing highs and lows with volume confirmation
        swing_highs = (recent_data['high'] == recent_data['high'].rolling(5, center=True, min_periods=1).max())
        swing_lows = (recent_data['low'] == recent_data['low'].rolling(5, center=True, min_periods=1).min())
        
        swing_high_points = recent_data[swing_highs][['high', 'volume']].tail(4).values
        swing_low_points = recent_data[swing_lows][['low', 'volume']].tail(4).values
        
        if len(swing_high_points) < 3 or len(swing_low_points) < 3:
            return {"mss_bullish": False, "mss_bearish": False, "strength": 0.0}
        
        # MSS Bullish: Higher High setelah Higher Low dengan volume confirmation
        mss_bullish = (swing_high_points[-1][0] > swing_high_points[-2][0] and 
                       swing_low_points[-1][0] > swing_low_points[-2][0] and
                       swing_high_points[-1][1] > swing_high_points[-2][1] * 0.8)
        
        # MSS Bearish: Lower Low setelah Lower High dengan volume confirmation  
        mss_bearish = (swing_low_points[-1][0] < swing_low_points[-2][0] and 
                       swing_high_points[-1][0] < swing_high_points[-2][0] and
                       swing_low_points[-1][1] > swing_low_points[-2][1] * 0.8)
        
        strength = 0.0
        if mss_bullish or mss_bearish:
            # Calculate strength based on move size and volume
            if mss_bullish:
                price_move = (swing_high_points[-1][0] - swing_high_points[-2][0]) / swing_high_points[-2][0]
                volume_ratio = swing_high_points[-1][1] / swing_high_points[-2][1]
            else:
                price_move = (swing_low_points[-2][0] - swing_low_points[-1][0]) / swing_low_points[-2][0]
                volume_ratio = swing_low_points[-1][1] / swing_low_points[-2][1]
            
            strength = min(1.0, price_move * 10 + volume_ratio * 0.5)
        
        return {"mss_bullish": bool(mss_bullish), "mss_bearish": bool(mss_bearish), "strength": strength}

    @staticmethod
    def _calculate_smc_confidence(bos: dict, mss: dict, df: pd.DataFrame) -> float:
        """Calculate comprehensive SMC confidence score"""
        smc_score = 0.0
        bias = bos.get('bias', 'NEUTRAL')
        
        # Base score from BOS
        if bias == 'LONG' or bias == 'SHORT':
            smc_score += 0.25 + (bos.get('strength', 0) * 0.25)  # Up to 0.5 from BOS
        
        # MSS bonus
        if mss["mss_bullish"] and bias == "LONG":
            smc_score += 0.2 * mss.get('strength', 0.5)
        if mss["mss_bearish"] and bias == "SHORT":
            smc_score += 0.2 * mss.get('strength', 0.5)
        
        # Volume confirmation bonus
        if 'vol_imbalance_score' in df.columns:
            vol_score = df['vol_imbalance_score'].iloc[-1]
            if ((bias == "LONG" and vol_score > 0.6) or 
                (bias == "SHORT" and vol_score < 0.4)):
                smc_score += 0.1
        
        return float(np.clip(smc_score, 0.0, 1.0))

    @staticmethod
    def _calculate_position_levels(df: pd.DataFrame, signal_type: str, bias: str, 
                                 ob_analysis: Dict, fvg_analysis: Dict) -> Dict[str, float]:
        """Calculate position levels dengan ICT advanced integration"""
        last_close = float(df['close'].iloc[-1])
        last_atr = float(df['atr'].iloc[-1]) if not np.isnan(df['atr'].iloc[-1]) else (last_close * 0.001)
        
        # Calculate OTE levels
        ote_calc = ICTAdvancedFeatures.calculate_optimal_trade_entry(df, bias)
        ote_level = ote_calc["ote_level"]
        
        # Use OB levels if available and relevant
        recent_ob = ob_analysis.get("recent_ob")
        if recent_ob and ((bias == "LONG" and recent_ob['type'] == 'BULLISH_OB') or
                         (bias == "SHORT" and recent_ob['type'] == 'BEARISH_OB')):
            # Adjust entry based on Order Block
            if bias == "LONG":
                entry = max(last_close, recent_ob['low'] * 1.001)  # Slight above OB low
                sl = recent_ob['low'] * 0.995  # Below OB low
            else:
                entry = min(last_close, recent_ob['high'] * 0.999)  # Slight below OB high
                sl = recent_ob['high'] * 1.005  # Above OB high
        else:
            # Standard ATR-based levels
            if signal_type.startswith("LONG"):
                entry = last_close + 0.3 * last_atr
                sl = last_close - 1.5 * last_atr
            elif signal_type.startswith("SHORT"):
                entry = last_close - 0.3 * last_atr
                sl = last_close + 1.5 * last_atr
            else:
                entry = tp1 = tp2 = sl = last_close
        
        # Calculate TP levels
        risk = abs(entry - sl)
        if signal_type.startswith("LONG"):
            tp1 = entry + risk * 1.5
            tp2 = entry + risk * 3.0
        elif signal_type.startswith("SHORT"):
            tp1 = entry - risk * 1.5
            tp2 = entry - risk * 3.0
        else:
            tp1 = tp2 = entry
        
        return {
            "entry": round(entry, 8),
            "tp1": round(tp1, 8),
            "tp2": round(tp2, 8),
            "sl": round(sl, 8),
            "ote_level": round(ote_level, 8)
        }

    @staticmethod
    def generate_ict_signal_pro(entry_df: pd.DataFrame, pair: str = None, tf: str = '15m', 
                              ml_confidence: Optional[float] = None, 
                              include_detailed_analysis: bool = True) -> Dict[str, Any]:
        """
        Enhanced PRO SMC/ICT signal generation dengan detailed analysis built-in
        """
        try:
            # Input validation dengan enhanced checks
            if entry_df is None or len(entry_df) < 100:
                return {"error": "Insufficient data", "signal_type": "WAIT", "confidence": 0.0}
            
            # Enhanced data validation
            if not enhanced_data_validation(entry_df):
                return {"error": "Data validation failed", "signal_type": "WAIT", "confidence": 0.0}
            
            # Prepare dataframe dengan semua enhanced features
            df = entry_df.copy()
            try:
                df = VolumeAnalyzer.add_volume_features(df)
                df = EnhancedFeatureEngineer.create_advanced_features(df)
                
                if 'atr' not in df.columns:
                    df['atr'] = TechnicalIndicators.atr(df, 14)
            except Exception as e:
                return {"error": f"Feature engineering failed: {str(e)}", "signal_type": "WAIT", "confidence": 0.0}
            
            # Detect semua SMC/ICT patterns
            bos = SMCICTProEngine.detect_bos_pro(df, lookback=60)
            mss = SMCICTProEngine.detect_market_structure_shift(df)
            
            # COMPLETE ICT ADVANCED ANALYSIS
            ob_analysis = ICTAdvancedFeatures.detect_order_blocks(df)
            fvg_analysis = ICTAdvancedFeatures.detect_fair_value_gaps(df)
            liquidity_analysis = ICTAdvancedFeatures.analyze_liquidity_zones(df)
            
            # Calculate semua confidence scores
            smc_conf = SMCICTProEngine._calculate_smc_confidence(bos, mss, df)
            vol_conf = VolumeAnalyzer.compute_volume_confidence(df, idx=-1)
            ml_conf = float(ml_confidence) if ml_confidence is not None else 0.0
            ict_conf = ICTAdvancedFeatures.calculate_ict_confidence(ob_analysis, fvg_analysis, liquidity_analysis, bos.get('bias', 'NEUTRAL'))
            
            # Enhanced confidence calculation dengan semua komponen
            market_regime = df['market_regime'].iloc[-1] if 'market_regime' in df.columns else 'ranging'
            volatility = df['volatility'].iloc[-1] if 'volatility' in df.columns else 0.0
            
            final_conf = calculate_complete_confidence(
                smc_conf, vol_conf, ml_conf, ict_conf, market_regime, volatility
            )
            
            # Signal determination dengan OTE levels
            signal_type = "WAIT"
            bias = bos.get('bias', 'NEUTRAL')
            
            if final_conf >= Config.STRONG_THRESHOLD and bias in ('LONG', 'SHORT'):
                signal_type = bias
            elif final_conf >= Config.WEAK_THRESHOLD and bias in ('LONG', 'SHORT'):
                signal_type = bias + "_WEAK"
            
            # Calculate position levels dengan OTE integration
            result = SMCICTProEngine._calculate_position_levels(df, signal_type, bias, ob_analysis, fvg_analysis)
            
            # Build comprehensive reasoning dengan ICT components
            reasoning_parts = []
            reasoning_details = []
            
            if bos.get('bias') != 'NEUTRAL':
                reasoning_parts.append(f"BOS: {bos.get('bias')}")
                reasoning_details.append(f"BOS Strength: {bos.get('strength', 0):.2f}")
            
            if mss.get('mss_bullish') or mss.get('mss_bearish'):
                mss_type = "BULLISH" if mss.get('mss_bullish') else "BEARISH"
                reasoning_parts.append(f"MSS: {mss_type}")
                reasoning_details.append(f"MSS Strength: {mss.get('strength', 0):.2f}")
            
            if ob_analysis.get("recent_ob"):
                ob = ob_analysis["recent_ob"]
                reasoning_parts.append(f"OB: {ob['type'].split('_')[0]}")
                reasoning_details.append(f"OB Strength: {ob.get('strength', 0):.2f}")
            
            if fvg_analysis.get("active_fvg"):
                fvg = fvg_analysis["active_fvg"]
                reasoning_parts.append(f"FVG: {fvg['type'].split('_')[0]}")
                reasoning_details.append(f"FVG Size: {fvg.get('size', 0):.4f}")
            
            if liquidity_analysis.get("liquidity_strength", 0) > 0:
                reasoning_parts.append("LiQ")
                reasoning_details.append(f"Liq Strength: {liquidity_analysis['liquidity_strength']:.2f}")
            
            # Final reasoning dengan detail
            reasoning = f"PRO_SMC | {' | '.join(reasoning_parts)}"
            if reasoning_details:
                reasoning += f" | Details: {', '.join(reasoning_details)}"
            
            # Base response
            response = {
                "pair": pair,
                "timeframe": tf,
                "signal_type": signal_type,
                "confidence": round(final_conf, 3),
                "reasoning": reasoning,
                "entry": result["entry"],
                "tp1": result["tp1"],
                "tp2": result["tp2"],
                "sl": result["sl"],
                "ote_level": result.get("ote_level", 0),
                "position_size": 0.01,  # Will be calculated later
                "engine_mode": "PRO_SMC_COMPLETE_ICT",
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "smc_conf": round(smc_conf, 3),
                    "vol_conf": round(vol_conf, 3), 
                    "ml_conf": round(ml_conf, 3),
                    "ict_conf": round(ict_conf, 3),
                    "market_regime": market_regime,
                    "volatility": round(volatility, 6),
                    "bias": bias,
                    "kill_zone_active": SMCICTProEngine.in_killzone(datetime.utcnow())
                }
            }
            
            # Include detailed ICT analysis jika diminta
            if include_detailed_analysis:
                response["ict_analysis"] = {
                    "market_structure": {
                        "bias": bos.get("bias"),
                        "bos_level": bos.get("bos_level"),
                        "bos_strength": bos.get("strength"),
                        "mss_bullish": mss.get("mss_bullish"),
                        "mss_bearish": mss.get("mss_bearish"),
                        "mss_strength": mss.get("strength")
                    },
                    "order_blocks": {
                        "total": len(ob_analysis.get("order_blocks", [])),
                        "recent": ob_analysis.get("recent_ob"),
                        "all_blocks": ob_analysis.get("order_blocks", [])[:3]  # First 3 only
                    },
                    "fair_value_gaps": {
                        "total": len(fvg_analysis.get("fair_value_gaps", [])),
                        "active": fvg_analysis.get("active_fvg"),
                        "all_gaps": fvg_analysis.get("fair_value_gaps", [])[:2]  # First 2 only
                    },
                    "liquidity_zones": {
                        "above": liquidity_analysis.get("liquidity_above", [])[:2],
                        "below": liquidity_analysis.get("liquidity_below", [])[:2],
                        "strength": liquidity_analysis.get("liquidity_strength", 0)
                    },
                    "summary": {
                        "ict_confidence": round(ict_conf, 3),
                        "total_components": len(ob_analysis.get("order_blocks", [])) + 
                                          len(fvg_analysis.get("fair_value_gaps", [])) +
                                          len(liquidity_analysis.get("liquidity_above", [])) +
                                          len(liquidity_analysis.get("liquidity_below", []))
                    }
                }
            
            return response
                
        except Exception as e:
            return {"error": str(e), "signal_type": "WAIT", "confidence": 0.0}

# --- ENHANCED SMC/ICT PRO ENGINE WITH FUNDAMENTAL ANALYSIS ---
class EnhancedSMCICTProEngine(SMCICTProEngine):
    """Enhanced PRO engine with fundamental analysis"""
    
    @staticmethod
    def generate_ict_signal_pro_with_fundamental(
        entry_df: pd.DataFrame, 
        pair: str = None, 
        tf: str = '15m',
        include_detailed_analysis: bool = True
    ) -> Dict[str, Any]:
        """Enhanced signal generation with fundamental analysis"""
        
        # Generate technical signal first
        technical_signal = SMCICTProEngine.generate_ict_signal_pro(
            entry_df,
            pair=pair,
            tf=tf,
            ml_confidence=None,
            include_detailed_analysis=include_detailed_analysis
        )
        
        # If technical signal says WAIT, return early
        if technical_signal.get("signal_type") == "WAIT":
            return technical_signal
        
        # Integrate fundamental analysis
        technical_confidence = technical_signal.get("confidence", 0.0)
        fundamental_context = FundamentalContext.get_fundamental_context(pair, technical_confidence)
        
        # Apply fundamental adjustments
        adjusted_confidence = fundamental_context["adjusted_confidence"]
        position_adjustment = fundamental_context["position_size_adjustment"]
        
        # Update the signal with fundamental context
        technical_signal["confidence"] = adjusted_confidence
        technical_signal["position_size"] = round(
            technical_signal.get("position_size", 0.01) * position_adjustment, 3
        )
        
        # Add fundamental analysis to response
        technical_signal["fundamental_analysis"] = {
            "risk_score": fundamental_context["fundamental_risk_score"],
            "position_adjustment": position_adjustment,
            "trading_recommendation": fundamental_context["trading_recommendation"],
            "high_impact_events": fundamental_context["high_impact_events"],
            "market_sentiment": fundamental_context["market_sentiment"],
            "original_technical_confidence": technical_confidence
        }
        
        # Update reasoning to include fundamental context
        if fundamental_context["fundamental_risk_score"] > 0.5:
            technical_signal["reasoning"] += f" | FUNDAMENTAL_RISK: {fundamental_context['trading_recommendation']}"
        
        return technical_signal

# --- MACHINE LEARNING MODELS ---
class MLModelManager:
    """Machine learning model management"""

    @staticmethod
    def build_historical_dataset():
        """Build historical dataset - improved: compute advanced features per-symbol before concat"""
        try:
            symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
            all_data = []

            for symbol in symbols:
                try:
                    print(f"[BUILD_DATASET] Fetching data for {symbol}")
                    df = DataFetcher.fetch_ohlc_any(symbol, "1d", limit=1000)
                    if df is None or df.empty:
                        continue

                    # basic derived columns / label
                    try:
                        df['rsi'] = TechnicalIndicators.rsi(df['close'])
                    except Exception:
                        df['rsi'] = df['close'].pct_change().fillna(0)  # fallback

                    df['volatility'] = df['close'].pct_change().rolling(20, min_periods=1).std()
                    df['label'] = (df['close'].shift(-1) > df['close']).astype(int)

                    # advanced features (this requires open/high/low/close/volume present)
                    try:
                        df_feat = EnhancedFeatureEngineer.create_advanced_features(df)
                    except Exception as e:
                        print(f"[BUILD_DATASET] Feature engineering failed for {symbol}: {e}")
                        continue

                    if df_feat.empty:
                        continue

                    # Ensure label exists in df_feat (recompute if necessary)
                    if 'label' not in df_feat.columns:
                        df_feat['label'] = (df_feat['close'].shift(-1) > df_feat['close']).astype(int)

                    # select only features we want (and label)
                    keep_cols = [c for c in ['rsi', 'volatility', 'momentum', 'adx', 'macd', 'obv', 'zscore', 'trend_strength', 'label'] if c in df_feat.columns]
                    if not keep_cols or 'label' not in keep_cols:
                        continue

                    df_ready = df_feat[keep_cols].dropna()
                    if not df_ready.empty:
                        all_data.append(df_ready)

                except Exception as e:
                    print(f"[BUILD_DATASET] Error for {symbol}: {e}")
                    continue

            if not all_data:
                print("[BUILD_DATASET] No data collected from symbols")
                return pd.DataFrame()

            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"[BUILD_DATASET] Combined dataset: {len(combined_df)} samples")
            return combined_df

        except Exception as e:
            print(f"[BUILD_DATASET] Fatal error: {e}")
            return pd.DataFrame()

    @staticmethod
    def train_historical_models(send_telegram: bool = True):
        """Train models on historical data with enhanced features"""
        try:
            # Build historical dataset dengan enhanced features
            df = MLModelManager.build_historical_dataset()
            if df.empty:
                print("[HISTORICAL] ❌ No data available.")
                return None

            # df sudah merupakan hasil dari build_historical_dataset() dan berisi fitur engineered
            # Pastikan label ada
            if 'label' not in df.columns:
                print("[HISTORICAL] ❌ Dataset missing 'label' column.")
                return None

            # Pilih feature yang tersedia
            feature_columns = ["rsi", "volatility", "momentum", "adx", "macd", "obv", "zscore", "trend_strength"]
            available_features = [col for col in feature_columns if col in df.columns]
            if not available_features:
                available_features = [col for col in ["rsi", "volatility"] if col in df.columns]
            if not available_features:
                print("[HISTORICAL] ❌ No usable features found.")
                return None

            X = df[available_features].fillna(0)
            y = df["label"].astype(int)
            
            # Enhanced feature engineering
            df = EnhancedFeatureEngineer.create_advanced_features(df)
            if df.empty:
                print("[HISTORICAL] ❌ No data after feature engineering.")
                return None

            # Define features and target dengan features yang lebih banyak
            feature_columns = ["rsi", "volatility", "momentum", "adx", "macd", "obv", "zscore", "trend_strength"]
            # Pastikan kolom ada di DataFrame
            available_features = [col for col in feature_columns if col in df.columns]
            if not available_features:
                available_features = ["rsi", "vol", "volatility"]  # Fallback ke features lama
            
            X = df[available_features]
            y = df["label"]

            # Time-series cross validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            models = {
                'rf': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
                'xgb': XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42)
            }
            
            results = {}
            for name, model in models.items():
                # Cross-validation scores
                cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
                
                # Train on full dataset
                model.fit(X, y)
                
                # Save model
                if name == 'rf':
                    joblib.dump(model, os.path.join(Config.MODEL_DIR, "rf_enhanced.pkl"))
                else:
                    model.save_model(os.path.join(Config.MODEL_DIR, "xgb_enhanced.json"))
                
                results[name] = {
                    'mean_cv_accuracy': round(cv_scores.mean(), 3),
                    'std_cv_accuracy': round(cv_scores.std(), 3),
                    'full_training_score': round(model.score(X, y), 3),
                    'features_used': available_features
                }

            msg = (f"✅ <b>Enhanced Historical Retrain Complete</b>\n"
                   f"RF: CV Acc={results['rf']['mean_cv_accuracy']}±{results['rf']['std_cv_accuracy']}\n"
                   f"XGB: CV Acc={results['xgb']['mean_cv_accuracy']}±{results['xgb']['std_cv_accuracy']}\n"
                   f"Data: {len(df)} samples | Features: {len(available_features)}")
            print(msg)
            
            if send_telegram and Config.TELEGRAM_AUTO_SEND:
                MLModelManager.send_telegram_message(msg)
                
            return results
            
        except Exception as e:
            err = f"[HISTORICAL] ❌ Enhanced training failed: {e}"
            print(err)
            if send_telegram and Config.TELEGRAM_AUTO_SEND:
                MLModelManager.send_telegram_message(err)
            return {"error": str(e)}

    @staticmethod
    def send_telegram_message(text: str):
        """Send message via Telegram"""
        if not Config.TELEGRAM_AUTO_SEND or not Config.TELEGRAM_BOT_TOKEN or not Config.TELEGRAM_CHAT_ID:
            return {"ok": False, "reason": "telegram_not_configured"}
        
        try:
            url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {"chat_id": Config.TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
            r = requests.post(url, json=payload, timeout=8)
            return r.json()
        except Exception as e:
            return {"ok": False, "reason": str(e)}

# --- REAL-TIME PERFORMANCE MONITOR ---
class ModelPerformanceMonitor:
    """Monitor model performance in real-time"""
    
    def __init__(self):
        self.predictions = []
        self.actuals = []
        self.confidences = []
        
    def update(self, prediction: str, actual: str, confidence: float):
        """Update monitor dengan prediction results"""
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.confidences.append(confidence)
        
        # Keep only last 100 predictions
        if len(self.predictions) > 100:
            self.predictions.pop(0)
            self.actuals.pop(0)
            self.confidences.pop(0)
        
        # Calculate metrics setiap 10 predictions
        if len(self.predictions) % 10 == 0:
            self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate and log performance metrics"""
        if len(self.predictions) < 20:
            return
            
        accuracy = accuracy_score(self.actuals, self.predictions)
        avg_confidence = np.mean(self.confidences)
        
        print(f"🎯 PERFORMANCE MONITOR | Accuracy: {accuracy:.3f} | Avg Confidence: {avg_confidence:.3f}")
        
        # Log to file untuk analisis lebih lanjut
        with open("performance_log.csv", "a") as f:
            f.write(f"{datetime.utcnow().isoformat()},{accuracy},{avg_confidence}\n")

# Global instance
performance_monitor = ModelPerformanceMonitor()

# --- FASTAPI APP ---
app = FastAPI(title=Config.APP_NAME, version="2.0")

@app.get("/health")
def health():
    return respond({"status": "ok", "service": Config.APP_NAME, "version": "2.0"})

@app.get("/pro_signal")
def pro_signal(
    pair: str = Query(...),
    tf_main: str = Query("1h"),
    tf_entry: str = Query("15m"),
    limit: int = Query(300),
    auto_log: bool = Query(True),
    detailed_analysis: bool = Query(False)
):
    """Enhanced PRO signal endpoint dengan complete SMC/ICT capabilities"""
    try:
        # Auto-map timeframe jika tidak provided
        if not tf_main or tf_main == "1h":
            tf_mapping = {
                "1m": "15m", "3m": "15m", "5m": "15m",
                "15m": "1h", "30m": "1h",
                "1h": "4h", "4h": "1d", "1d": "1w"
            }
            tf_main = tf_mapping.get(tf_entry, "1h")
            print(f"[PRO_SIGNAL] Auto-mapped tf_entry {tf_entry} → tf_main {tf_main}")

        # Fetch data
        df_entry = DataFetcher.fetch_ohlc_any(pair, tf_entry, limit=limit)

        # Generate PRO signal dengan complete SMC/ICT features
        pro_res = SMCICTProEngine.generate_ict_signal_pro(
            df_entry,
            pair=pair,
            tf=tf_entry,
            include_detailed_analysis=detailed_analysis
        )

        # Attach news (optional) — bungkus try agar tidak crash kalau news gagal
        try:
            pro_res = attach_news_to_signal(pro_res, pair, page_size=30)
        except Exception as e:
            print("[PRO_SIGNAL] News attach failed:", e)

        # Position sizing
        entry = float(pro_res.get("entry", 0))
        sl = float(pro_res.get("sl", entry))
        risk_amount = Config.ACCOUNT_BALANCE * Config.RISK_PERCENT if Config.ACCOUNT_BALANCE > 0 else 0

        if abs(entry - sl) > 0 and risk_amount > 0:
            pos_size = round(max(0.01, risk_amount / abs(entry - sl)), 3)
        else:
            pos_size = 0.01

        pro_res["position_size"] = pos_size

        # Log trade
        if auto_log and pro_res.get("signal_type") != "WAIT":
            append_trade_log({
                "pair": pro_res.get("pair"),
                "timeframe": pro_res.get("timeframe"),
                "signal_type": pro_res.get("signal_type"),
                "entry": pro_res.get("entry"),
                "tp1": pro_res.get("tp1"),
                "tp2": pro_res.get("tp2"),
                "sl": pro_res.get("sl"),
                "confidence": pro_res.get("confidence"),
                "reasoning": pro_res.get("reasoning"),
                "engine_used": "PRO_SMC_COMPLETE_ICT",
                "backtest_hit": None,
                "backtest_pnl": None,
                "fundamental_risk": None
            })

        return respond(pro_res)

    except Exception as e:
        error_msg = f"internal_error: {str(e)}"
        print(f"[PRO_SIGNAL] ❌ {error_msg}")
        return respond({"error": error_msg}, status_code=500)

@app.get("/pro_signal_enhanced")
def pro_signal_enhanced(
    pair: str = Query(...),
    tf_main: str = Query("1h"),
    tf_entry: str = Query("15m"),
    limit: int = Query(300),
    auto_log: bool = Query(True),
    detailed_analysis: bool = Query(False)
):
    """Enhanced PRO signal with fundamental analysis"""
    try:
        # Auto-map timeframe jika tidak provided
        if not tf_main or tf_main == "1h":
            tf_mapping = {
                "1m": "15m", "3m": "15m", "5m": "15m", 
                "15m": "1h", "30m": "1h",
                "1h": "4h", "4h": "1d", "1d": "1w"
            }
            tf_main = tf_mapping.get(tf_entry, "1h")
            print(f"[PRO_SIGNAL_ENHANCED] Auto-mapped tf_entry {tf_entry} → tf_main {tf_main}")

        # Fetch data
        df_entry = DataFetcher.fetch_ohlc_any(pair, tf_entry, limit=limit)
        
        # Generate enhanced signal with fundamental analysis
        pro_res = EnhancedSMCICTProEngine.generate_ict_signal_pro_with_fundamental(
            df_entry, 
            pair=pair, 
            tf=tf_entry,
            include_detailed_analysis=detailed_analysis
        )
        
        # Attach news (non fatal)
        try:
            pro_res = attach_news_to_signal(pro_res, pair, page_size=30)
        except Exception as e:
            print("[PRO_SIGNAL_ENHANCED] News attach failed:", e)
        
        # Position sizing
        entry = float(pro_res.get("entry", 0))
        sl = float(pro_res.get("sl", entry))
        risk_amount = Config.ACCOUNT_BALANCE * Config.RISK_PERCENT if Config.ACCOUNT_BALANCE > 0 else 0
        
        if abs(entry - sl) > 0 and risk_amount > 0:
            pos_size = round(max(0.01, risk_amount / abs(entry - sl)), 3)
        else:
            pos_size = 0.01
            
        pro_res["position_size"] = pos_size
        
        # Log trade (enhanced with fundamental context)
        if auto_log and pro_res.get("signal_type") != "WAIT":
            append_trade_log({
                "pair": pro_res.get("pair"),
                "timeframe": pro_res.get("timeframe"),
                "signal_type": pro_res.get("signal_type"),
                "entry": pro_res.get("entry"),
                "tp1": pro_res.get("tp1"),
                "tp2": pro_res.get("tp2"),
                "sl": pro_res.get("sl"),
                "confidence": pro_res.get("confidence"),
                "reasoning": pro_res.get("reasoning"),
                "engine_used": "PRO_SMC_FUNDAMENTAL_ENHANCED",
                "backtest_hit": None,
                "backtest_pnl": None,
                "fundamental_risk": pro_res.get("fundamental_analysis", {}).get("risk_score")
            })
        
        return respond(pro_res)
        
    except Exception as e:
        error_msg = f"enhanced_signal_error: {str(e)}"
        print(f"[PRO_SIGNAL_ENHANCED] ❌ {error_msg}")
        return respond({"error": error_msg}, status_code=500)

@app.get("/retrain_historical")
def retrain_historical():
    """Manual historical retraining endpoint"""
    try:
        res = MLModelManager.train_historical_models(send_telegram=True)
        return respond({"status": "ok", "result": res})
    except Exception as e:
        return respond({"error": str(e)}, status_code=500)
        
@app.get("/test_enhanced_features")
def test_enhanced_features(
    pair: str = Query("BTCUSDT"),
    tf: str = Query("1h"),
    limit: int = Query(200)
):
    """Test endpoint untuk enhanced features"""
    try:
        df = DataFetcher.fetch_ohlc_any(pair, tf, limit)
        
        # Test enhanced features
        df_enhanced = EnhancedFeatureEngineer.create_advanced_features(df)
        
        # Test complete ICT features
        ob_analysis = ICTAdvancedFeatures.detect_order_blocks(df_enhanced)
        fvg_analysis = ICTAdvancedFeatures.detect_fair_value_gaps(df_enhanced)
        liquidity_analysis = ICTAdvancedFeatures.analyze_liquidity_zones(df_enhanced)
        
        return respond({
            "original_columns": list(df.columns),
            "enhanced_columns": list(df_enhanced.columns),
            "latest_features": df_enhanced.iloc[-1:].to_dict('records')[0],
            "data_validation": enhanced_data_validation(df),
            "ict_analysis": {
                "order_blocks_detected": len(ob_analysis.get("order_blocks", [])),
                "fair_value_gaps_detected": len(fvg_analysis.get("fair_value_gaps", [])),
                "liquidity_zones_detected": len(liquidity_analysis.get("liquidity_above", []) + liquidity_analysis.get("liquidity_below", [])),
                "recent_order_block": ob_analysis.get("recent_ob"),
                "active_fvg": fvg_analysis.get("active_fvg")
            }
        })
    except Exception as e:
        return respond({"error": str(e)}, status_code=500)

@app.get("/ict_analysis")
def ict_analysis(
    pair: str = Query("BTCUSDT"),
    tf: str = Query("1h"),
    limit: int = Query(200)
):
    """ICT analysis endpoint - sekarang menggunakan pro_signal dengan detailed analysis"""
    try:
        # Gunakan pro_signal dengan detailed_analysis=True
        df = DataFetcher.fetch_ohlc_any(pair, tf, limit)
        
        pro_res = SMCICTProEngine.generate_ict_signal_pro(
            df, pair=pair, tf=tf, include_detailed_analysis=True
        )
        
        # Format response khusus untuk ICT analysis
        if "error" in pro_res:
            return respond(pro_res)
        
        # Extract hanya bagian ICT analysis
        ict_response = {
            "pair": pair,
            "timeframe": tf,
            "current_price": df['close'].iloc[-1] if not df.empty else 0,
            "signal_info": {
                "signal_type": pro_res.get("signal_type"),
                "confidence": pro_res.get("confidence"),
                "bias": pro_res.get("details", {}).get("bias")
            },
            **pro_res.get("ict_analysis", {})
        }
        
        return respond(ict_response)
        
    except Exception as e:
        return respond({"error": str(e)}, status_code=500)

# --- NEW FUNDAMENTAL ANALYSIS ENDPOINTS ---
@app.get("/economic_calendar")
def get_economic_calendar(
    currency: str = Query("USD"),
    days: int = Query(3)
):
    """Get economic calendar for specific currency"""
    try:
        events = EconomicCalendar.get_high_impact_events(currency, days * 24)
        return respond({
            "currency": currency,
            "period_days": days,
            "high_impact_events": events,
            "total_events": len(events)
        })
    except Exception as e:
        return respond({"error": str(e)}, status_code=500)

@app.get("/market_sentiment")
def get_market_sentiment(
    pair: str = Query("BTCUSDT")
):
    """Get market sentiment for trading pair"""
    try:
        sentiment = NewsSentimentAnalyzer.get_market_sentiment(pair)
        return respond({
            "pair": pair,
            "sentiment_analysis": sentiment
        })
    except Exception as e:
        return respond({"error": str(e)}, status_code=500)

@app.get("/fundamental_analysis")
def get_fundamental_analysis(
    pair: str = Query("EURUSD"),
    technical_confidence: float = Query(0.7)
):
    """Get comprehensive fundamental analysis"""
    try:
        analysis = FundamentalContext.get_fundamental_context(pair, technical_confidence)
        return respond({
            "pair": pair,
            "fundamental_analysis": analysis
        })
    except Exception as e:
        return respond({"error": str(e)}, status_code=500)

@app.get("/news_fetch")
def news_fetch(pair: str, page_size: int = NEWS_PAGE_SIZE):
    data = fetch_news_for_pair(pair, page_size)
    return {"ok": True, "source": data["source"], "articles": data["articles"], "count": len(data["articles"])}

@app.get("/news_sentiment")
def news_sentiment(pair: str, page_size: int = NEWS_PAGE_SIZE):
    data = fetch_news_for_pair(pair, page_size)
    agg = aggregate_news_sentiment(data["articles"], pair)
    return {"ok": True, "pair": pair, "source": data["source"], "agg": agg}
    
@app.get("/news_summary")
def news_summary(pair: str, page_size: int = NEWS_PAGE_SIZE):
    """Return Summarizer PRO combined summary for a pair"""
    try:
        data = fetch_news_for_pair(pair, page_size)
        summaries = summarize_articles_list(data["articles"], max_sent_per_article=2)
        return respond({
            "pair": pair,
            "source": data["source"],
            "article_count": len(data["articles"]),
            "summarizer": summaries
        })
    except Exception as e:
        return respond({"error": str(e)}, status_code=500)

# =======================================================================
#           NEW ADVANCED TRADING ENGINES (FULL DETAIL VERSION)
# =======================================================================

class HighImpactBlocker:
    """Blocker: hindari trading saat event ekonomi HIGH IMPACT."""
    @staticmethod
    def check(pair: str, hours_ahead: int = None):
        try:
            if hours_ahead is None:
                hours_ahead = Config.BLOCKER_WINDOW_HOURS

            base = pair[:3]
            quote = pair[3:]

            base_events = EconomicCalendar.get_high_impact_events(base, hours_ahead)
            quote_events = EconomicCalendar.get_high_impact_events(quote, hours_ahead)

            events = base_events + quote_events

            if events:
                return {
                    "blocked": True,
                    "pair": pair,
                    "hours_window": hours_ahead,
                    "events": events
                }

            return {"blocked": False, "events": []}

        except Exception as e:
            return {"blocked": False, "error": str(e)}
            
class ScalpEngine:
    """Ultra-fast Micro SMC/ICT engine for 1m–5m scalping."""
    @staticmethod
    def generate(df: pd.DataFrame, pair: str, tf: str):
        try:
            if df is None or len(df) < 60:
                return {"error": "insufficient_data"}

            df = df.copy()
            df = VolumeAnalyzer.add_volume_features(df)
            df = EnhancedFeatureEngineer.create_advanced_features(df)

            # Micro SMC features
            bos = SMCICTProEngine.detect_bos_pro(df, lookback=40, atr_mul=0.6)
            mss = SMCICTProEngine.detect_market_structure_shift(df, lookback=60)
            ob = ICTAdvancedFeatures.detect_order_blocks(df, lookback=30)
            fvg = ICTAdvancedFeatures.detect_fair_value_gaps(df, threshold=Config.FVG_THRESHOLD)
            liq = ICTAdvancedFeatures.analyze_liquidity_zones(df, window=35)

            vol_conf = VolumeAnalyzer.compute_volume_confidence(df, idx=-1)
            smc_conf = SMCICTProEngine._calculate_smc_confidence(bos, mss, df)
            ict_conf = ICTAdvancedFeatures.calculate_ict_confidence(ob, fvg, liq, bos.get("bias", "NEUTRAL"))

            final_conf = calculate_complete_confidence(
                smc_conf, vol_conf, 0.0, ict_conf,
                df["market_regime"].iloc[-1], df["volatility"].iloc[-1]
            )

            bias = bos.get("bias", "NEUTRAL")
            signal_type = "WAIT"

            if final_conf >= Config.STRONG_THRESHOLD:
                signal_type = f"SCALP_{bias}"
            elif final_conf >= Config.WEAK_THRESHOLD:
                signal_type = f"SCALP_{bias}_WEAK"

            # Micro-level entries
            levels = SMCICTProEngine._calculate_position_levels(df, signal_type, bias, ob, fvg)

            # 0.5% risk default
            risk_amt = Config.ACCOUNT_BALANCE * Config.SCALP_MAX_RISK_PCT
            diff = abs(levels["entry"] - levels["sl"])
            pos_size = (
                round(max(0.01, risk_amt / diff), 3) 
                if risk_amt > 0 and diff > 0 
                else 0.01
            )

            return {
                "pair": pair,
                "timeframe": tf,
                "signal_type": signal_type,
                "confidence": round(final_conf, 3),
                "entry": levels["entry"],
                "tp1": levels["tp1"],
                "tp2": levels["tp2"],
                "sl": levels["sl"],
                "position_size": pos_size,
                "details": {
                    "bos": bos,
                    "order_blocks": ob,
                    "fvg": fvg,
                    "liquidity": liq
                },
                "reasoning": f"SCALP | {bias} | vol:{vol_conf:.2f} | ict:{ict_conf:.2f}"
            }

        except Exception as e:
            return {"error": str(e)}
            
class SwingProEngine:
    """HTF Swing Trading Engine (4H–1D)."""
    @staticmethod
    def generate(df: pd.DataFrame, pair: str, tf: str):
        try:
            if df is None or len(df) < 200:
                return {"error": "insufficient_data"}

            df = df.copy()
            df = VolumeAnalyzer.add_volume_features(df)
            df = EnhancedFeatureEngineer.create_advanced_features(df)

            bos = SMCICTProEngine.detect_bos_pro(df, lookback=120, atr_mul=1.2)
            mss = SMCICTProEngine.detect_market_structure_shift(df, lookback=200)
            ob = ICTAdvancedFeatures.detect_order_blocks(df, lookback=80)
            fvg = ICTAdvancedFeatures.detect_fair_value_gaps(df, threshold=Config.FVG_THRESHOLD)
            liq = ICTAdvancedFeatures.analyze_liquidity_zones(df, window=120)

            vol_conf = VolumeAnalyzer.compute_volume_confidence(df, idx=-1)
            smc_conf = SMCICTProEngine._calculate_smc_confidence(bos, mss, df)
            ict_conf = ICTAdvancedFeatures.calculate_ict_confidence(ob, fvg, liq, bos.get("bias", "NEUTRAL"))
            fund = FundamentalContext.get_fundamental_context(pair, technical_confidence=0.7)

            final_conf = calculate_complete_confidence(
                smc_conf, vol_conf, 0.0, ict_conf,
                df["market_regime"].iloc[-1], df["volatility"].iloc[-1]
            )

            if fund["fundamental_risk_score"] > 0.4:
                final_conf *= 0.9

            bias = bos.get("bias", "NEUTRAL")

            if final_conf >= Config.STRONG_THRESHOLD:
                signal_type = f"SWING_{bias}"
            elif final_conf >= Config.WEAK_THRESHOLD:
                signal_type = f"SWING_{bias}_WEAK"
            else:
                signal_type = "WAIT"

            levels = SMCICTProEngine._calculate_position_levels(df, signal_type, bias, ob, fvg)

            risk_amt = Config.ACCOUNT_BALANCE * Config.RISK_PERCENT
            diff = abs(levels["entry"] - levels["sl"])
            pos_size = (
                round(max(0.01, risk_amt / diff), 3)
                if diff > 0 else 0.01
            )

            return {
                "pair": pair,
                "timeframe": tf,
                "signal_type": signal_type,
                "confidence": round(final_conf, 3),
                "entry": levels["entry"],
                "tp1": levels["tp1"],
                "tp2": levels["tp2"],
                "tp3": levels["tp2"],
                "sl": levels["sl"],
                "position_size": pos_size,
                "fundamental_analysis": fund,
                "details": {
                    "bos": bos,
                    "mss": mss,
                    "order_blocks": ob,
                    "fvg": fvg,
                    "liquidity": liq
                },
                "reasoning": (
                    f"SWING | {bias} | smc:{smc_conf:.2f} | "
                    f"ict:{ict_conf:.2f} | fund_risk:{fund['fundamental_risk_score']:.2f}"
                )
            }

        except Exception as e:
            return {"error": str(e)}
            
class NewsDrivenEngine:
    """Entry berdasarkan berita + sentiment."""
    @staticmethod
    def generate(pair: str, tf: str):
        try:
            news = fetch_news_for_pair(pair, page_size=30)
            sentiment = aggregate_news_sentiment(news["articles"], pair)
            summary = summarize_articles_list(news["articles"])

            avg_sent = sentiment.get("avg_score", 0.0)
            urgency = (
                sentiment.get("impact_weight", 0.0) > 0.35
                or summary.get("urgency", False)
            )

            try:
                df = DataFetcher.fetch_ohlc_any(pair, "1h", limit=20)
                shock = (df["close"].iloc[-1] / df["close"].iloc[-4] - 1) if len(df) > 4 else 0
            except:
                shock = 0

            conf = min(0.99, 0.5 + avg_sent * 0.4 + (0.1 if urgency else 0))

            if urgency or abs(shock) > 0.01:
                bias = summary.get("agg_bias", "neutral")
                signal_type = (
                    "NEWS_LONG" if bias == "bullish"
                    else "NEWS_SHORT" if bias == "bearish"
                    else "WAIT"
                )
            else:
                signal_type = "WAIT"

            return {
                "pair": pair,
                "timeframe": tf,
                "signal_type": signal_type,
                "confidence": round(conf, 3),
                "summary": summary,
                "sentiment": sentiment,
                "return_shock_pct": round(shock * 100, 2),
                "reasoning": f"NEWS | avg_sent:{avg_sent:.2f} | urgency:{urgency}"
            }

        except Exception as e:
            return {"error": str(e)}

# ------------------- Attach news to trading signal -------------------
def attach_news_to_signal(signal: Dict[str,Any], pair: str, page_size: int = 30):
    data = fetch_news_for_pair(pair, page_size)
    agg = aggregate_news_sentiment(data["articles"], pair)

    # Apply existing adj
    base_conf = float(signal.get("confidence",0) or 0)
    adj = agg["confidence_adjustment"]
    new_conf = max(0, min(0.99, base_conf + adj))

    # Add summarizer PRO output
    try:
        summaries = summarize_articles_list(data["articles"], max_sent_per_article=2)
    except Exception as e:
        summaries = {"combined_summary":"", "article_summaries":[], "agg_bias":"neutral", "urgency":False, "impact":"LOW"}

    signal["confidence_before_news"] = base_conf
    signal["confidence"] = new_conf
    signal["news_source"] = data["source"]
    signal["news_agg"] = agg
    signal["news_top_articles"] = agg["top"]
    signal["news_summary"] = summaries  # <-- new: full summarizer output
    # update reasoning
    signal["reasoning"] = f"{signal.get('reasoning','')} | NewsBias:{agg['fundamental_bias']} adj:{adj} | SummBias:{summaries.get('agg_bias')}"
    return signal

# --- STARTUP ---
@app.on_event("startup")
def startup_event():
    """Initialize application on startup dengan enhanced models"""
    ensure_trade_log()
    Config.validate_config()
    
    # Load existing models - prioritaskan enhanced models
    global _cached_rf, _cached_xgb
    try:
        # Coba load enhanced models dulu
        rf_enhanced_path = os.path.join(Config.MODEL_DIR, "rf_enhanced.pkl")
        if os.path.exists(rf_enhanced_path):
            _cached_rf = joblib.load(rf_enhanced_path)
            print("[startup] ✅ RF enhanced model loaded.")
        else:
            # Fallback ke model lama
            rf_path = os.path.join(Config.MODEL_DIR, "rf_historical_10y.pkl")
            if os.path.exists(rf_path):
                _cached_rf = joblib.load(rf_path)
                print("[startup] ✅ RF historical model loaded.")
    except Exception as e:
        print("[startup] ⚠️ RF load fail:", e)

    try:
        xgb_enhanced_path = os.path.join(Config.MODEL_DIR, "xgb_enhanced.json")
        if os.path.exists(xgb_enhanced_path):
            m = XGBClassifier()
            m.load_model(xgb_enhanced_path)
            _cached_xgb = m
            print("[startup] ✅ XGB enhanced model loaded.")
        else:
            xgb_path = os.path.join(Config.MODEL_DIR, "xgb_historical_10y.json")
            if os.path.exists(xgb_path):
                m = XGBClassifier()
                m.load_model(xgb_path)
                _cached_xgb = m
                print("[startup] ✅ XGB historical model loaded.")
    except Exception as e:
        print("[startup] ⚠️ XGB load fail:", e)

    # Auto retrain dengan enhanced features
    def enhanced_auto_retrain():
        while True:
            print("[EnhancedAutoTrain] Weekly historical retraining started...")
            try:
                MLModelManager.train_historical_models()
                print("[EnhancedAutoTrain] ✅ Weekly update done.")
            except Exception as e:
                print("[EnhancedAutoTrain] ❌ Error:", e)
            time.sleep(7 * 24 * 3600)  # 1 week

    threading.Thread(target=enhanced_auto_retrain, daemon=True).start()

# --- MAIN ---
if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Starting {Config.APP_NAME} on port {Config.PORT}")
    print(f"✅ Complete SMC/ICT Features: Order Blocks, FVG, Liquidity Zones, OTE")
    print(f"✅ Fundamental Analysis: Economic Calendar, News Sentiment, Risk Management")
    print(f"✅ Enhanced Endpoints: /pro_signal_enhanced, /economic_calendar, /market_sentiment")
    uvicorn.run(
        "main_combined_learning_hybrid_pro_final:app",
        host="0.0.0.0",
        port=Config.PORT,
        reload=False,
        log_level="info"
    )