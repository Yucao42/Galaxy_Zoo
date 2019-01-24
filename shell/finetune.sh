set -x
MODEL="resnet18_KL"

mkdir -p models/resnet/${MODEL}_fine_kl_custom
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}_fine_kl_custom  \
--batch_size 64 \
--step 40 \
--epochs 250 \
--lr 1e-2 \
--p 0.3 \
--weight_decay 0  \
--lock_bn  \
--load models/resnet/${MODEL}/model_best.pth \
--optimized \
--momentum 0.9  \
2>&1 | tee models/resnet/${MODEL}_fine_kl_custom/${MODEL}_training_50_finetune.report 
