set -x
MODEL="resnet50_1111_finetune_KL_real"

python3 eval.py  \
--name ${MODEL}  \
--load  models/resnet/${MODEL}/model_20*
