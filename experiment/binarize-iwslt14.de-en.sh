#!/usr/bin/env bash

SRC=de
TGT=en
DATASET_PATH=dataset-tokenized/iwslt14.de-en
DATA_PATH=data-bin/iwslt14.de-en
PRE_TRAIN=1
PRE_SRC=psrc
PRE_TGT=ptgt

CUDA_VISIBLE_DEVICES=0 python fairseq_cli/preprocess.py --source-lang ${SRC} --target-lang ${TGT} \
  --joined-dictionary \
  --trainpref ${DATASET_PATH}/train --validpref ${DATASET_PATH}/valid --testpref ${DATASET_PATH}/test \
  --destdir ${DATA_PATH} \
  --workers 20

if [ "$PRE_TRAIN" -ne 0 ]; then
  src_files=$(find "${DATASET_PATH}" -type f -name "*.${SRC}")
  for src_file in $src_files; do
    paste -d'\n' "${src_file}" "${src_file%.${SRC}}.${TGT}" > "${src_file%.${SRC}}.${PRE_SRC}"
    paste -d'\n' "${src_file%.${SRC}}.${TGT}" "${src_file}" > "${src_file%.${SRC}}.${PRE_TGT}"
  done

  CUDA_VISIBLE_DEVICES=0 python fairseq_cli/preprocess.py --source-lang ${PRE_SRC} --target-lang ${PRE_TGT} \
  --joined-dictionary \
  --trainpref ${DATASET_PATH}/train --validpref ${DATASET_PATH}/valid --testpref ${DATASET_PATH}/test \
  --destdir ${DATA_PATH} \
  --workers 20
fi

mkdir -p "${DATA_PATH}/dp"
for l in ${SRC} ${TGT} ${PRE_SRC} ${PRE_TGT}; do
  find "${DATASET_PATH}/dp" -type f -name "*.${l}" | while read file; do
    target_file="${DATA_PATH}/$(echo "$file" | sed "s|${DATASET_PATH}/||")"
    cp "${file}" "${target_file}"
  done
done
