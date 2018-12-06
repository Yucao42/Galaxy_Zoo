set -x
MODEL="resnet50_normalized_1111_finetune"

python3 eval.py  \
--name ${MODEL}_1  \
--load  models/resnet/${MODEL}/model_best.pth \
--optimized
