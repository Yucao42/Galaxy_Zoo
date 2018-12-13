set -x
MODEL="resnet18_64_fine"

mkdir -p models/resnet/${MODEL}_long_sf
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}_long_sf  \
--batch_size 64 \
--step 20 \
--epochs 70 \
--lr 2e-3 \
--load models/resnet/${MODEL}/model_best.pth \
--p 0.1  \
--weight_decay 5e-4  \
--optimized \
--momentum 0.9  \
2>&1 | tee models/resnet/${MODEL}_long_sf/${MODEL}_training_50_finetune.report 
