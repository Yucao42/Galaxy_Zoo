set -x
MODEL="resnet50_normalized_1111"

mkdir -p models/resnet/${MODEL}_finetune5111
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}_finetune5111 \
--batch_size 64 \
--step 15 \
--epochs 40 \
--lr 1e-3  \
--p 0.5  \
--weight_decay 5e-4  \
--optimized \
--momentum 0.9  \
--load models/resnet/${MODEL}/model_best.pth \
2>&1 | tee models/resnet/${MODEL}_finetune5111/${MODEL}_training_50_finetune.report 
