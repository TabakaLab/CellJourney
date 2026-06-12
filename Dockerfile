FROM python:3.11.7-slim

WORKDIR /app

COPY . .

RUN apt-get update \
    && apt-get install -y --no-install-recommends chromium \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

ENV HOST=0.0.0.0
ENV PORT=8080
ENV BROWSER_PATH=/usr/bin/chromium

EXPOSE 8080

CMD ["python", "celljourney.py", "--suppressbrowser"]
