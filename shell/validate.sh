set -x
#MODEL="resnet50_normalized_1111_dpdp"
MODEL="resnet18_64_fine"

python3 validate.py  \
--name ${MODEL}_VALIDATION  \
--load  models/resnet/${MODEL}/model_best.pth \
--optimized
