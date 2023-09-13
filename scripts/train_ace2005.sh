REPO=D:/Paper/review/Entity-Relation-As-Multi-Turn-QA-master
PRETRAINED_MODEL=D:/Paper/review/Entity-Relation-As-Multi-Turn-QA-master/bert-base-uncasedter

python $REPO/train.py \
--dataset_tag ace2005 \
--train_path D:/Paper/review/Entity-Relation-As-Multi-Turn-QA-master/data/cleaned_data/ACE2005/bert-base-uncased_overlap_15_window_300_threshold_1_max_distance_45/train.json \
--train_batch 20 \
--test_path D:/Paper/review/Entity-Relation-As-Multi-Turn-QA-master/data/cleaned_data/ACE2005/test.json \
--test_batch 20 \
--pretrained_model_path $PRETRAINED_MODEL \
--max_epochs 10 \
--warmup_ratio 0.1 \
--lr 2e-5 \
--theta 0.25 \
--window_size 300 \
--overlap 45 \
--threshold 5 \
--max_grad_norm 1 \
--test_eval \
--seed 0 \
--amp
