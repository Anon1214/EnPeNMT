#!/usr/bin/env bash

SRC=de
TGT=en
SHARE_TYPE=cycle_rev
DATA_PATH=data-bin/iwslt14.de-en
DP_BETA=0.5
CONTRASTIVE_LAMBDA=0.0625
RDROP_ALPHA=17
MODEL_PATH=model-train/EnPeNMT/uni/iwslt14.de-en
CKPT=model-train/EnPeNMT/bid/iwslt14.de-en/checkpoint.best_avg.pt
BPE_ANCHOR_MODE=first
DEP_ATTN_LAYERS=0,1
VALID_SUBSET=valid

MAX_EPOCH=250
LR=0.0005
WARMUP_UPDATES=4000
WARMUP_INIT_LR=1e-07
MAX_TOKENS=4096

mkdir -p ${MODEL_PATH}
if [ ! -f "${CKPT}" ]; then
   echo "Average checkpoint not found. Creating by top 3 best checkpoints."
   python scripts/average_checkpoints.py --inputs $(dirname "${CKPT}") --output "${CKPT}" --num-best-checkpoints 3
fi
export CUDA_VISIBLE_DEVICES=0

python fairseq_cli/train.py ${DATA_PATH} \
  -s ${SRC} -t ${TGT} \
  --arch transformer_iwslt_de_en --share-all-embeddings \
  --optimizer adam --adam-betas '(0.9,0.98)' \
  --lr ${LR} --lr-scheduler inverse_sqrt \
  --max-epoch ${MAX_EPOCH} --warmup-updates ${WARMUP_UPDATES} --warmup-init-lr ${WARMUP_INIT_LR} \
  --criterion my_label_smoothed_cross_entropy_with_rdrop --label-smoothing 0.1 \
  --rdrop-alpha ${RDROP_ALPHA} --dropout 0.3 \
  --max-tokens ${MAX_TOKENS} \
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
  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler --restore-file ${CKPT} \
  --save-dir ${MODEL_PATH} |
  tee -a ${MODEL_PATH}/train.log
