import requests
import logging

def send_telegram_message(text: str, token: str, chat_id: str):
    """Send message via Telegram"""
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML"
        }
        response = requests.post(url, json=payload, timeout=10)
        return response.json()
    except Exception as e:
        logging.error(f"Telegram send error: {e}")
        return None

def health_check(url: str):
    """Check if service is healthy"""
    try:
        response = requests.get(f"{url}/health", timeout=10)
        return response.status_code == 200
    except:
        return False
