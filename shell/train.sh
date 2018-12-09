set -x
MODEL="resnet50_normalized_1111_dpdp"

mkdir -p models/resnet/${MODEL}
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}  \
--batch_size 32 \
--step 18 \
--epochs 70 \
--lr 2e-2  \
--p 0.5  \
--weight_decay 5e-4  \
--optimized \
--momentum 0.9  \
2>&1 | tee models/resnet/${MODEL}/${MODEL}_training_50_finetune.report 
