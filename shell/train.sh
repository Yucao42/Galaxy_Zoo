set -x
MODEL="stn2_360"

mkdir -p models/${MODEL}
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}  \
--batch_size 64  \
--step 14 \
--epochs 60 \
--lr 1e-3  \
--p 0.5  \
--weight_decay 1e-3  \
--momentum 0.9  \
2>&1 | tee models/${MODEL}/${MODEL}_training_50_finetune.report 
