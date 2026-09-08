"""
test_connection.py — Alpaca 연결 테스트
API 키를 코드에 절대 직접 쓰지 않고, .env 파일에서만 읽어온다.

사용법:
1. .env.template을 복사해서 .env로 이름 바꾸기: cp .env.template .env
2. .env 파일을 열어서 placeholder를 실제(새로 발급받은!) 키로 교체
3. python test_connection.py 실행
"""
import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경변수 로드

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

if not api_key or api_key.startswith("여기에"):
    raise RuntimeError(
        ".env 파일에 실제 API 키가 설정되지 않았습니다. "
        ".env.template을 .env로 복사한 뒤 placeholder를 실제 키로 교체하세요."
    )

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta

client = StockHistoricalDataClient(api_key, secret_key)

request = StockBarsRequest(
    symbol_or_symbols=["AAPL"],
    timeframe=TimeFrame.Day,
    start=datetime.now() - timedelta(days=10),
)

bars = client.get_stock_bars(request)
df = bars.df

print("연결 성공! 최근 AAPL 일봉 데이터:")
print(df.tail())
