FROM python:3.8-slim-buster

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

COPY templates /app/templates
COPY static /app/static

EXPOSE 5000

CMD [ "python", "./app.py" ]