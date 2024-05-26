FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y build-essential python3-dev wget && \
    pip install numpy

# Install TA_Lib from source
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

CMD exec gunicorn --bind :$PORT --workers 3 app:app
