rm res
test_file=$1
ALLENNLP_PATH=/home/ubuntu/anaconda3/envs/pytorch_p36/bin/allennlp
for i in 500 497 494 491 488; do
python ${ALLENNLP_PATH} predict --cuda-device 0 --predictor dstqa --include-package dstqa --weights-file model/model_state_epoch_${i}.th model/model.tar.gz ${test_file} > ${i}
python formulate_pred_belief_state.py ${i} >> res
done
