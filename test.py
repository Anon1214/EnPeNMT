import re

fi = "/sda/yzh/my_param_sharing/dataset-tokenized-no-bpe/iwslt14.de-en/train.en"
pattern = r"&[a-zA-Z]+;"
with open(fi, 'r', encoding="utf-8") as f:
    text = f.read()
    print(set(re.findall(pattern, text)))
