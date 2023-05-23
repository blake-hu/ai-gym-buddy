FROM arm64v8/python:3.7

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install numpy
RUN pip install opencv-python
RUN pip install mediapipe
RUN pip install tensorflow

# RUN python3 driver.py

CMD [ "python3", "driver.py" ]