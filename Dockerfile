FROM cytomineuliege/software-python3-base:v2.8.3-py3.8.12-slim

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

RUN mkdir -p /app
ADD descriptor.json /app/descriptor.json
ADD run.py /app/run.py
ADD utils.py /app/utils.py
ADD models/ /models/

ENTRYPOINT ["python", "/app/run.py"]