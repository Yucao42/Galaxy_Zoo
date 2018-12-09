set -x
MODEL="resnet50_normalized_1111"
POSIX="finetune_dpdp"

mkdir -p models/resnet/${MODEL}${POSIX}_finetune4111
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}${POSIX}_finetune4111 \
--batch_size 64 \
--step 10 \
--epochs 35 \
--lr 2e-4  \
--p 0.5  \
--weight_decay 8e-4  \
--optimized \
--momentum 0.9  \
--load models/resnet/${MODEL}${POSIX}/model_best.pth \
2>&1 | tee models/resnet/${MODEL}${POSIX}_finetune4111/${MODEL}${POSIX}_training_50_finetune.report 
