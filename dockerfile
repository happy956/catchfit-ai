FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

ENV PYTHON_VERSION=3.11

RUN apt-get update -y && \
    apt-get install -y git && \
    apt-get install -y python${PYTHON_VERSION} && \
    apt-get install -y python3-pip && \
    apt-get install -y nvidia-container-runtime && \
    apt-get clean

RUN mkdir /app/

WORKDIR /app/

RUN git clone https://github.com/team-healica/ai.git

WORKDIR /app/ai/

RUN pip3 install -r requirements.txt

RUN apt-get update -y && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get install -y libglib2.0-0

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "6000"]