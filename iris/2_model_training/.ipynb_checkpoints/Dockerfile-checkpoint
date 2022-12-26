FROM frolvlad/alpine-python-machinelearning
ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code
COPY . /code/
ENTRYPOINT ["python", "training_model.py"]