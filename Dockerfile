FROM python:3.12.6-slim

WORKDIR /usr/src/app
COPY . .

RUN apt-get update && apt-get install -y \
    libsndfile1 \ 
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app.py"]
