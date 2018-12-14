set -x
MODEL="resnet18_sigmoid_KL"

mkdir -p models/resnet/${MODEL}
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}  \
--batch_size 64 \
--step 15 \
--epochs 80 \
--lr 2e-2 \
--p 0.1  \
--load models/resnet/${MODEL}/model_best.pth \
--weight_decay 5e-4  \
--optimized \
--sigmoid \
--momentum 0.9  \
2>&1 | tee models/resnet/${MODEL}/${MODEL}_training_50_finetune.report 
