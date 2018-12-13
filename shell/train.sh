set -x
MODEL="resnet34_64"

mkdir -p models/resnet/${MODEL}_fine
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}_fine  \
--batch_size 64 \
--step 13 \
--epochs 45 \
--lr 1e-2 \
--load models/resnet/${MODEL}/model_best.pth \
--p 0.1  \
--weight_decay 5e-4  \
--optimized \
--momentum 0.9  \
2>&1 | tee models/resnet/${MODEL}_fine/${MODEL}_training_50_finetune.report 
