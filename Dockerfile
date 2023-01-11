FROM python:3.8

ENV PORT=80

COPY requirements.txt .

RUN apt-get update

RUN apt-get install -y libgl1-mesa-glx

RUN pip install -r requirements.txt

COPY ./app ./app

CMD ["python", "./app/main.py"]