FROM python:3.11.7-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENV HOST=0.0.0.0
ENV PORT=8080

EXPOSE 8080

CMD ["python", "celljourney.py"]
