set -x
#MODEL="resnet50_normalized_1111_dpdp"
MODEL="resnet18_64_fine"

python3 eval.py  \
--name ${MODEL}_254  \
--load  models/resnet/${MODEL}/model_best.pth \
--optimized
