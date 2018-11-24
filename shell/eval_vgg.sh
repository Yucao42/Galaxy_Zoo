set -x
MODEL="vgg16_bn"

python3 eval.py  \
--name ${MODEL}  \
--load  models/resnet/${MODEL}/model_35*
