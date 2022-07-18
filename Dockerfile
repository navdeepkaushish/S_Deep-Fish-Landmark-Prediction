FROM cytomineuliege/software-python3-base:v2.8.3-py3.8.12-slim

RUN pip install numpy tensorflow opencv-python-headless albumentations

RUN mkdir -p /app
ADD descriptor.json /app/descriptor.json
ADD run.py /app/run.py
ADD utils.py /app/utils.py

ENTRYPOINT ["python", "/app/run.py"]