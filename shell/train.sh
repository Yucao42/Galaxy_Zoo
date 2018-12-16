set -x
MODEL="resnet18_l1"

mkdir -p models/resnet/${MODEL}
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}  \
--batch_size 32 \
--step 18 \
--epochs 80 \
--lr 4e-2 \
--p 0.1  \
--weight_decay 5e-4  \
--optimized \
--momentum 0.9  \
2>&1 | tee models/resnet/${MODEL}/${MODEL}_training_50_finetune.report 
