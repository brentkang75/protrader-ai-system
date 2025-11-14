import os

class SharedConfig:
    # Data API Keys
    TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
    ALPHA_API_KEY = os.getenv("ALPHA_API_KEY", "") 
    FMP_API_KEY = os.getenv("FMP_API_KEY", "")
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
    
    # News API Keys
    NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY", "")
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "")
    
    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
    
    # Trading Config
    RISK_PERCENT = float(os.getenv("RISK_PERCENT", "0.02"))
    ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "10000"))
    
    @classmethod
    def validate_required_keys(cls):
        """Validate that required API keys are present"""
        required = [
            cls.TWELVEDATA_API_KEY, 
            cls.ALPHA_API_KEY,
            cls.NEWSAPI_API_KEY
        ]
        return all(required)
