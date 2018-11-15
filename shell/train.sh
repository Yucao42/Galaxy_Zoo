set -x
MODEL="resnet50"

mkdir -p models/resnet/${MODEL}
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}  \
--batch_size 32  \
--step 15 \
--epochs 40 \
--lr 1e-3  \
--p 0.25  \
--weight_decay 1e-3  \
--momentum 0.9  \
2>&1 | tee models/resnet/${MODEL}/${MODEL}_training_50_finetune.report 
