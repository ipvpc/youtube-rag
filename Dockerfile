FROM python:3.10-slim

RUN apt update && apt install -y git ffmpeg

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN rm /app/requirements.txt

WORKDIR /app

COPY *.py /app/
COPY docs/ /app/docs/
COPY templates/ /app/templates/

EXPOSE 5004

CMD ["python", "app.py"]

