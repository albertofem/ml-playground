FROM python:3.8-slim as builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc git

COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

FROM python:3.8-slim

WORKDIR /app

COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

RUN pip install --no-cache /wheels/*

# Install java for Torchserve (TODO: move to different containers)

RUN apt-get update && \
    apt-get install -y --no-install-recommends wget gnupg2 software-properties-common

RUN wget -O- https://apt.corretto.aws/corretto.key | apt-key add -
RUN add-apt-repository 'deb https://apt.corretto.aws stable main'

RUN apt-get update; apt-get install -y java-11-amazon-corretto-jdk

COPY . /app