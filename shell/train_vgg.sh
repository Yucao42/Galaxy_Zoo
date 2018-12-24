set -x
MODEL="vgg16_bn"

mkdir -p models/resnet/${MODEL}
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}  \
--batch_size 64 \
--step 10 \
--epochs 40 \
--lr 1e-2  \
--p 0.5  \
--weight_decay 5e-4  \
--momentum 0.9  \
2>&1 | tee models/resnet/${MODEL}/${MODEL}_training_50_finetune.report 
