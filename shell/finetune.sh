set -x
MODEL="resnet50_360aug"

mkdir -p models/resnet/${MODEL}_finetune1
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}_finetune1  \
--batch_size 64 \
--step 14 \
--epochs 40 \
--lr 7e-3  \
--p 0.5  \
--weight_decay 1e-4  \
--momentum 0.9  \
--load models/resnet/${MODEL}_finetune/model_16* \
2>&1 | tee models/resnet/${MODEL}_finetune1/${MODEL}_training_50_finetune.report 
