set -x
MODEL="resnet50"

mkdir -p models/resnet/${MODEL}_fine
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}_fine  \
--batch_size 64 \
--step 25 \
--epochs 80 \
--lr 6e-4 \
--p 0.2 \
--weight_decay 5e-4  \
--load models/resnet/${MODEL}/model_best.pth \
--optimized \
--momentum 0.9  \
2>&1 | tee models/resnet/${MODEL}_fine/${MODEL}_training_50_finetune.report 
