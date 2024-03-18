FROM python:3.10-slim-buster

WORKDIR /app

COPY model/requirements.txt ./model/

COPY model/torch_requirements.txt ./model/

RUN pip install --upgrade pip

RUN pip install -r ./model/requirements.txt

RUN pip install -r ./model/torch_requirements.txt --index-url https://download.pytorch.org/whl/cpu

RUN pip install flask

COPY . .

ENTRYPOINT ["python", "model/code/app.py"]