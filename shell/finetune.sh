set -x
MODEL="resnet18_64_fine"

mkdir -p models/resnet/${MODEL}_32
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}_32  \
--batch_size 128 \
--step 10 \
--epochs 70 \
--lr 3e-2 \
--load models/resnet/${MODEL}_32/model_best.pth \
--p 0.3  \
--weight_decay 5e-4  \
--optimized \
--momentum 0.9  \
2>&1 | tee models/resnet/${MODEL}_32/${MODEL}_training_50_finetune.report 
