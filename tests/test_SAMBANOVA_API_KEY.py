import os
from dotenv import load_dotenv
import requests
import json

# Путь к .env файлу       python test_SAMBANOVA_API_KEY.py
ENV_FILE_PATH = r'D:\Tinnitus-Search\.continue\.env'

# Загружаем .env
load_dotenv(ENV_FILE_PATH)

# Извлекаем ключ (если нет — ошибка)
api_key = os.getenv('SAMBANOVA_API_KEY')
if not api_key:
    raise ValueError("SAMBANOVA_API_KEY не найден в .env файле!")

# Базовый URL API
BASE_URL = 'https://api.sambanova.ai/v1'

# Headers с аутентификацией
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

def print_headers_info(response):
    """Выводит все релевантные headers, включая rate limits"""
    print("\n=== Rate Limits и другие Headers ===")
    for key, value in response.headers.items():
        if 'ratelimit' in key.lower() or 'x-' in key.lower():  # Фокус на limits и X-headers
            print(f"{key}: {value}")
    print("===================================")

# 1. Запрос на список моделей (/v1/models)
print("\n=== Запрашиваем список доступных моделей ===")
models_response = requests.get(f'{BASE_URL}/models', headers=headers)

if models_response.status_code == 200:
    models_data = models_response.json()
    print("Доступные модели:")
    print(json.dumps(models_data, indent=2))
    print_headers_info(models_response)
else:
    print(f"Ошибка: {models_response.status_code} - {models_response.text}")

# 2. Тестовый запрос на чат-комплетион (/v1/chat/completions) для проверки limits
# Используем минимальный промпт, чтобы не тратить токены
payload = {
    "model": "Qwen3-32B",  # Замените на вашу модель, если нужно (из списка выше)
    "messages": [{"role": "user", "content": "Hello, test request."}],
    "max_tokens": 5,  # Минимально, чтобы увидеть limits в headers
    "stream": False
}

print("\n=== Тестовый чат-запрос для проверки limits ===")
chat_response = requests.post(f'{BASE_URL}/chat/completions', headers=headers, json=payload)

if chat_response.status_code == 200:
    chat_data = chat_response.json()
    print("Ответ от API:")
    print(json.dumps(chat_data, indent=2))
    print_headers_info(chat_response)
else:
    print(f"Ошибка: {chat_response.status_code} - {chat_response.text}")

# Если нужно вывести ключ (не рекомендуется для безопасности)
# print(f"\nВаш ключ: {api_key}")