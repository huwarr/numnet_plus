#!/bin/env bash
#SBATCH --job-name=eval_roberta_numnet_plus
source activate numnet_venv_1_0

set -xe

DATA_PATH=$1
DUMP_PATH=$2
TMSPAN=$3
PRE_PATH=$4
PRETRAIN_PATH=$5

BERT_CONFIG="--encoder ${PRETRAIN_PATH}"

if [ ${TMSPAN} = tag_mspan ]; then
    echo "Use tag_mspan model..."
    MODEL_CONFIG="--gcn_steps 3 --use_gcn --tag_mspan"
else
    echo "Use mspan model..."
    MODEL_CONFIG="--gcn_steps 3 --use_gcn"
fi

echo "Starting evaluation..."
TEST_CONFIG="--eval_batch_size 32 --pre_path ${PRE_PATH} --data_mode dev --dump_path ${DUMP_PATH} \
             --inf_path ${DATA_PATH}"

python3 roberta_predict.py \
    ${TEST_CONFIG} \
    ${BERT_CONFIG} \
    ${MODEL_CONFIG}
