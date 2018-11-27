set -x
MODEL="resnet50_1111_finetune1"

python3 eval.py  \
--name ${MODEL}  \
--load  models/resnet/${MODEL}/model_36*
