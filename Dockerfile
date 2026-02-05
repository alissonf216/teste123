FROM python:3.12.3
WORKDIR /app

ENV PYTHONPATH="/app"
ENV PORT=8013

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app/ .
COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

EXPOSE 8013
ENTRYPOINT ["/entrypoint.sh"]
