FROM ubuntu:18.04
LABEL maintainer Nasreen Nazer "nasreen@saal.ai"
RUN apt-get update
RUN apt-get install -y software-properties-common vim

#RUN apt-get update && apt-get install procps -y && apt install python3 python3-pip -y && pip3 install --upgrade pip
RUN apt-get --fix-missing update && apt-get --fix-broken install && apt-get install -y poppler-utils && apt-get install -y tesseract-ocr && \
    apt-get install -y libtesseract-dev && apt-get install -y libleptonica-dev && ldconfig && apt-get install -y python3.6 && \
    apt-get install -y python3-pip && apt install -y libsm6 libxext6
WORKDIR /src

COPY ./requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN python3 -m nltk.downloader averaged_perceptron_tagger
RUN apt-get install tesseract-ocr
RUN pip3 install pytesseract
COPY ./model /model/

COPY ./src /src/

WORKDIR /

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV FLASK_APP=src/ner_model.py
EXPOSE 5000
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]
