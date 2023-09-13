HOME=/home/Entity-Relation-As-Multi-Turn-QA-master
REPO=/home/Entity-Relation-As-Multi-Turn-QA-master


python $REPO/ckpt_eval.py \
--window_size 300 \
--overlap 45 \
--checkpoint_path $REPO/checkpoints/ace2004/2020_10_10_00_09_21/checkpoint_0.cpt \
--test_path $REPO/data/cleaned_data/ACE2004/test0.json \
--test_batch 20 \
--threshold 3