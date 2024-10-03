FROM python:3.12-slim

WORKDIR C:/Users/dylan/Codes/tradingbot_tests/Runner

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

CMD ["python", "./options_trader.py"]
