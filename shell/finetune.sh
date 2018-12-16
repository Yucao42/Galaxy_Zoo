set -x
MODEL="resnet18_64_fine"

mkdir -p models/resnet/${MODEL}_discrete_rotation
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}_discrete_rotation  \
--batch_size 64 \
--step 20 \
--epochs 70 \
--lr 4e-3 \
--load models/resnet/${MODEL}/model_best.pth \
--p 0.3  \
--weight_decay 5e-4  \
--optimized \
--momentum 0.9  \
2>&1 | tee models/resnet/${MODEL}_discrete_rotation/${MODEL}_training_50_finetune.report 
