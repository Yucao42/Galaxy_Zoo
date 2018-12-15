set -x
MODEL="resnet18_64_fine"

mkdir -p models/resnet/${MODEL}_32
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}_32  \
--batch_size 64 \
--step 20 \
--epochs 70 \
--lr 8e-4 \
--load models/resnet/${MODEL}_32/model_best.pth \
--p 0.3  \
--weight_decay 1e-4  \
--optimized \
--momentum 0.9  \
2>&1 | tee models/resnet/${MODEL}_32/${MODEL}_training_50_finetune.report 
