FROM tensorflow/tensorflow:2.13.0-gpu-jupyter

WORKDIR /app
COPY requirements.txt /app

RUN pip install -r requirements.txt

ENTRYPOINT cd src && jupyter notebook --ip=0.0.0.0
