set -x
MODEL="resnet50_180"

python3 eval.py  \
--name ${MODEL}  \
--load  models/resnet/${MODEL}/model_39*
