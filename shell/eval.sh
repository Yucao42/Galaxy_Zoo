set -x
MODEL="resnet50"

python3 eval.py  \
--name ${MODEL}  \
--load  models/resnet/${MODEL}/model_26*
