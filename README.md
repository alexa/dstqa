## DSTQA - Dialogue State Tracking via Question Answering

This repository contains code for the following paper:
Li Zhou, Kevin Small. Multi-domain Dialogue State Tracking as Dynamic Knowledge Graph Enhanced Question Answering. In NeurIPS 2019 Workshop on Conversational AI ([PDF](https://arxiv.org/pdf/1911.06192.pdf))

## Training and Evaluation
Step 1 - Install Dependency
pip install -r requirements.txt

Step 2 - Download Dataset
wget https://raw.githubusercontent.com/jasonwu0731/trade-dst/master/create_data.py
wget https://raw.githubusercontent.com/jasonwu0731/trade-dst/master/utils/mapping.pair
sed -i 's/utils\/mapping.pair/mapping.pair/g' create_data.py
python create_data.py 

Step 3 - Preprocess Dataset
python multiwoz_format.py all ./data ./data

Step 4 - Pre-calculate ELMO Embeddings
mkdir ./data/elmo_embeddings
bash calc_elmo.sh ./data ./data/elms_embeddings

Step 5 - Train
bash train_nosp.sh

Step 6 - Evaluation
bash predict.sh ./data/prediction.json

## License
This library is licensed under the Amazon Software License.

