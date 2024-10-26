FROM python:3.12.6-slim

WORKDIR /app
   
RUN apt-get update && apt-get install -y --no-install-recommends git openssh-client \
    libsndfile1 \ 
    libgomp1 \
    cmake \
    make \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


ARG SSH_PRIVATE_KEY
RUN mkdir -p /root/.ssh/ && \
    echo "$SSH_PRIVATE_KEY" > /root/.ssh/id_ed25519 && \
    chmod 600 /root/.ssh/id_ed25519

RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

# Clone the repository at the specific tag
ARG REPO_URL=git@github.com:mostafashahin/speechgrid.git
ARG TAG_NAME=v1.0.0-alpha

RUN git clone --branch $TAG_NAME --single-branch $REPO_URL /app

RUN pip install -r requirements.txt

EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app.py"]
