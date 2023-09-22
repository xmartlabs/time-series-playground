FROM tensorflow/tensorflow:2.13.0-gpu-jupyter

WORKDIR /app
COPY requirements.txt /app

RUN pip install -r requirements.txt

RUN git config --global --add safe.directory /app
ENV SHELL=/bin/bash
ENTRYPOINT cd src && jupyter notebook --ip=0.0.0.0 --allow-root
