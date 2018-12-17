set -x
MODEL="resnet18_64_fine_discrete_rotation_sc_3"
mkdir -p models/resnet/${MODEL}_0.1dp #cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}_0.1dp  \
--batch_size 64 \
--step 15 \
--epochs 70 \
--lr 5e-4 \
--load models/resnet/${MODEL}/model_best.pth \
--p 0.1  \
--weight_decay 5e-4  \
--optimized \
--momentum 0.9  \
2>&1 | tee models/resnet/${MODEL}_0.1dp/${MODEL}_training_50_finetune.report 
