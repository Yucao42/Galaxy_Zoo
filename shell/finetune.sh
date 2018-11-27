set -x
MODEL="resnet50_1111"

mkdir -p models/resnet/${MODEL}_finetune_lockBN
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}_finetune_lockBN  \
--batch_size 64 \
--step 10 \
--epochs 40 \
--lr 7e-3  \
--p 0.5  \
--weight_decay 1e-4  \
--momentum 0.9  \
--lock_bn \
--load models/resnet/${MODEL}/model_60* \
2>&1 | tee models/resnet/${MODEL}_finetune_lockBN/${MODEL}_training_50_finetune.report 
