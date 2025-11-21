FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC

WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app/
RUN mkdir -p /app/data

CMD ["python", "-u", "src/runner.py"]