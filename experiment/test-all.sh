#!/usr/bin/env bash

for dir in model-train; do
  find "${dir}" -type f \( -name "checkpoint_best.pt" -o -name "checkpoint.best_avg.pt" \) -printf '%h\n' | while read -r MODEL_PATH; do
    MODEL_PATH=${MODEL_PATH%/}
    echo "${SPLIT_LINE}"
    ckpt_avg="${MODEL_PATH}"/checkpoint.best_avg.pt
    gen_file_avg="${MODEL_PATH}"/result_avg.gen
    bleu_file_avg="${MODEL_PATH}"/result_avg.bleu

    echo "Model path: ${MODEL_PATH}"
    data_bin_name=$(echo "${MODEL_PATH##*/}" | cut -d '.' -f 1,2)
    DATA_PATH=data-bin/"${data_bin_name}"
    if ! [ -d "${DATA_PATH}" ]; then
      echo "No Bin Data found."
      continue
    fi
    echo "Found data path: ${DATA_PATH}"

    if ! [ -f "$bleu_file_avg" ] || [ $(tail -n 1 "$bleu_file_avg" | grep BLEU | wc -l) -ne 1 ]; then
      if [ ! -f "${ckpt_avg}" ]; then
        echo "Average checkpoint not found. Creating by top 3 best checkpoints."
        python scripts/average_checkpoints.py --inputs "${MODEL_PATH}" --output "${ckpt_avg}" --num-best-checkpoints 3
      fi

      python fairseq_cli/generate.py "${DATA_PATH}" \
        --path "${ckpt_avg}" \
        --beam 5 --remove-bpe --load-dependency > "${gen_file_avg}"
      bash scripts/compound_split_bleu.sh "${gen_file_avg}" > "${bleu_file_avg}"
    fi
  done
done
