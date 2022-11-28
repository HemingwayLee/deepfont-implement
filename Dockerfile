FROM python:3.7.15

RUN mkdir -p /home/proj/
COPY . /home/proj/
WORKDIR /home/proj/

RUN pip3 install -r requirements.txt
RUN pip3 list
RUN python3 train.py

# CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

