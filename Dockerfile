FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev git && \
    rm -rf /var/lib/apt/lists/* 

# Avoid dialog during build
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3.10 

WORKDIR /app

# Mount the kaggle.json file (at Run time, to not expose the kaggle.json file in the image)
VOLUME /root/.kaggle

# Install kaggle, torch and torchvision, etc 
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY classify_gemma.py classify_gemma.py
COPY questions.csv questions.csv

# Clone the repo. This has download links to allow the model to be downloaded at runtime
RUN git clone https://github.com/google/gemma_pytorch.git

EXPOSE 5002

RUN chmod +x classify_gemma.py

CMD ["python3", "classify_gemma.py"]

# Sample build command
# docker build -t gemma_pytorch:latest .
# docker run -p 5002:5002 --gpu all -v ~/.kaggle/kaggle.json:/root/.kaggle/kaggle.json gemma_pytorch:latest

# Sample request
# curl -X POST -H "Content-Type: application/json" -d '{"question":"What is the square root of 5?"}' http://localhost:5002/categorize

