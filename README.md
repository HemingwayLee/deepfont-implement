# deepfont-implement
* How to identify fonts in pictures using deep learning? we can do font recognition using DeepFont algorithm (from Adobe)
* This is from [Font_Recognition-DeepFont](https://github.com/robinreni96/Font_Recognition-DeepFont), in this repo
  * `requirements.txt` and `Dockerfile` are added so we can set it up easily
  * Github action is added so we can make sure Dockerfile do work
  * Data generation script `create_data.py` is provided
  * `train.py` and `eval.py` are provided so we can run training and evaluation without jupyter notebook
  
# How to run it locally
* setup
```
python3 -m "virtualenv" venv
source venv/bin/activate
pip3 install -r requirements.txt
```

* create dataset
  * `-c`: the number of generated images for each font in `fonts` folder
  * `-f`: the target folder
```
python3 create_data.py -c 1 -f data
```

* train 
  * `-e`: epoch
```
python3 train.py -e 1
```
