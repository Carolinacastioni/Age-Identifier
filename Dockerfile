FROM ubuntu

RUN apt update
RUN apt install -y python3 python3-pip

ADD . .

RUN pip3 install -r requirements.txt

CMD ["python3","api.py"]