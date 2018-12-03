set -x
MODEL="resnet50_1111"

mkdir -p models/resnet/${MODEL}_finetune_KL_real
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}_finetune_KL_real \
--batch_size 64 \
--step 15 \
--epochs 40 \
--lr 1e-3  \
--p 0.5  \
--weight_decay 5e-4  \
--momentum 0.9  \
--load models/resnet/${MODEL}_finetune1/model_36* \
2>&1 | tee models/resnet/${MODEL}_finetune_KL_real/${MODEL}_training_50_finetune.report 
