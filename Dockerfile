FROM python:3.10-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install -r requirements.txt
