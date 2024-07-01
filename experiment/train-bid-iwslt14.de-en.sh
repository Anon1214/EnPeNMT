#!/usr/bin/env bash

SRC=psrc
TGT=ptgt
SHARE_TYPE=cycle_rev
DATA_PATH=data-bin/iwslt14.de-en
DP_BETA=0.5
CONTRASTIVE_LAMBDA=0.0625
MODEL_PATH=model-train/EnPeNMT/bid/iwslt14.de-en
BPE_ANCHOR_MODE=first
DEP_ATTN_LAYERS=0,1
VALID_SUBSET=valid

mkdir -p ${MODEL_PATH}
export CUDA_VISIBLE_DEVICES=0

python fairseq_cli/train.py ${DATA_PATH} \
  -s ${SRC} -t ${TGT} \
  --arch transformer_iwslt_de_en --share-all-embeddings \
  --optimizer adam --adam-betas '(0.9,0.98)' \
  --lr 0.0003 --lr-scheduler inverse_sqrt \
  --max-epoch 250 --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --criterion my_label_smoothed_cross_entropy_with_rdrop --label-smoothing 0.1 \
  --dropout 0.4 \
  --max-tokens 4608 \
  --no-progress-bar \
  --seed 64 \
  --fp16 \
  --eval-bleu \
  --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
  --eval-bleu-detok moses \
  --eval-bleu-remove-bpe \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --valid-subset ${VALID_SUBSET} \
  --keep-best-checkpoints 3 --no-epoch-checkpoints --num-workers 16 \
  --share-params-cross-layer --share-layer-num 2 --share-type ${SHARE_TYPE} \
  --load-dependency --dependency-child-loss-alpha ${DP_BETA} --dependency-parent-loss-beta ${DP_BETA} \
  --dependency-attn-layers ${DEP_ATTN_LAYERS} --bpe-anchor-mode ${BPE_ANCHOR_MODE} \
  --my-pretrain --contrastive-lambda ${CONTRASTIVE_LAMBDA} --validate-after-epochs 200 \
  --rebuild-batches --empty-cache-every-updates 150 \
  --save-dir ${MODEL_PATH} |
  tee -a ${MODEL_PATH}/train.log
