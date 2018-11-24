set -x
MODEL="resnet50_360aug_finetune"

python3 eval.py  \
--name ${MODEL}  \
--load  models/resnet/${MODEL}/model_16*
