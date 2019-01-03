set -x
MODEL="resnet18_grpy_2"

mkdir -p models/resnet/${MODEL}_fine_rotate_v100_2
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}_fine_rotate_v100_2  \
--batch_size 64 \
--step 29 \
--epochs 100 \
--lr 6e-4 \
--p 0.0 \
--weight_decay 5e-4  \
--load models/resnet/${MODEL}/model_best.pth \
--optimized \
--momentum 0.9  \
2>&1 | tee models/resnet/${MODEL}_fine_rotate_v100_2/${MODEL}_training_50_finetune.report 
