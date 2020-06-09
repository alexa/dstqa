SAVE_MODEL_PATH=./model
ALLENNLP_PATH=/home/ubuntu/anaconda3/envs/pytorch_p36/bin/allennlp
rm -r ${SAVE_MODEL_PATH}
python ${ALLENNLP_PATH} train config_sp.jsonnet -s ${SAVE_MODEL_PATH} --include-package dstqa
