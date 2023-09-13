# REIA : Entity Relation Extraction Based on Interaction Policy and Data Augmentation



## Install Requirements

`pip install -r requirements.txt`


## Pretrained Model

We use [BERT-Base-Uncased](https://huggingface.co/bert-base-uncased)

## Train

Use `train.py` to train the model. 

The example used can be found in `scripts/train_ace2005.sh`.

## Evaluate checkpoints:

Use `ckpt_eval.py` to evaluate the saved model.

The example used can be found in `scripts/ckpt_eval.sh`.