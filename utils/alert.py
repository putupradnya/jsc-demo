import os
import requests
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

def send_whatsapp_alert(message_body):
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_WHATSAPP_FROM")
    to_number = os.getenv("TWILIO_WHATSAPP_TO")

    client = Client(account_sid, auth_token)

    message = client.messages.create(
        from_=f"whatsapp:{from_number}",
        body=message_body,
        to=f"whatsapp:{to_number}"
    )
    print("[WA] Message sent:", message.sid)

# === Telegram ===
def send_telegram_message(message, photo_path=None):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    url_msg = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.get(url_msg, params={"chat_id": chat_id, "text": message})
    except Exception as e:
        print("[Telegram ERROR - text]", e)

    if photo_path and os.path.exists(photo_path):
        url_photo = f"https://api.telegram.org/bot{token}/sendPhoto"
        try:
            with open(photo_path, "rb") as photo:
                files = {"photo": photo}
                data = {"chat_id": chat_id}
                requests.post(url_photo, data=data, files=files)
        except Exception as e:
            print("[Telegram ERROR - photo]", e)

