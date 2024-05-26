FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN apt-get update && \
    apt-get install -y build-essential python3-dev libta-lib0-dev libta-lib0 && \
    pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "tradingbot.wsgi"]
