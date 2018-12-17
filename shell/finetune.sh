set -x
MODEL="resnet18_64_fine_discrete_rotation_sc"

mkdir -p models/resnet/${MODEL}_focal
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}_focal  \
--batch_size 64 \
--step 25 \
--epochs 70 \
--lr 5e-3 \
--load models/resnet/${MODEL}/model_best.pth \
--p 0.3  \
--weight_decay 5e-4  \
--optimized \
--momentum 0.9  \
2>&1 | tee models/resnet/${MODEL}_focal/${MODEL}_training_50_finetune.report 
