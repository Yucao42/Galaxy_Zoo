set -x
MODEL="resnet18_DUAL"

mkdir -p models/resnet/${MODEL}
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}  \
--batch_size 64 \
--step 40 \
--epochs 150 \
--lr 3e-2 \
--p 0.2 \
--weight_decay 5e-4  \
--optimized \
--momentum 0.9  \
2>&1 | tee models/resnet/${MODEL}/${MODEL}_training_50_finetune.report 
