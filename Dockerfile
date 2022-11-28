FROM python:3.7.15

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN mkdir -p /home/proj/
COPY . /home/proj/
WORKDIR /home/proj/

RUN pip3 install -r requirements.txt
RUN pip3 list
RUN python3 train.py

# CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

