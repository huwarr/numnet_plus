#!/bin/env bash
#SBATCH --job-name=eval_roberta_numnet_plus
source activate numnet-venv-new

set -xe

DATA_PATH=$1
DUMP_PATH=$2
TMSPAN=$3
PRE_PATH=$4

BERT_CONFIG="--roberta_model drop_dataset/roberta.large"

echo "Use tag_mspan model..."
MODEL_CONFIG="--gcn_steps 3 --use_gcn --tag_mspan"

echo "Starting evaluation..."
TEST_CONFIG="--eval_batch_size 4 --pre_path ${PRE_PATH} --data_mode dev --dump_path ${DUMP_PATH} \
             --inf_path ${DATA_PATH}"

python roberta_predict.py \
    ${TEST_CONFIG} \
    ${BERT_CONFIG} \
    ${MODEL_CONFIG}
