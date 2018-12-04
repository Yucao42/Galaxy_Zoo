set -x
MODEL="resnet50_normalized_1111"

python3 eval.py  \
--name ${MODEL}  \
--load  models/resnet/${MODEL}/model_best
