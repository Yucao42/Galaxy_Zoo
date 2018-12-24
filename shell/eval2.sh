set -x
#MODEL="resnet50_normalized_1111_dpdp"
#MODEL="resnet18_64_fine_long_sf"

#MODEL="resnet50_normalized_1111finetune_dpdp_finetune4111"
MODEL="resnet18_64_fine"


python3 eval.py  \
--name ${MODEL}  \
--load  models/resnet/${MODEL}/model_best.pth \
--optimized \
--degree 0 \

python3 eval.py  \
--name ${MODEL}  \
--load  models/resnet/${MODEL}/model_best.pth \
--optimized \
--degree 90 \

python3 eval.py  \
--name ${MODEL}  \
--load  models/resnet/${MODEL}/model_best.pth \
--optimized \
--degree 180 \

python3 eval.py  \
--name ${MODEL}  \
--load  models/resnet/${MODEL}/model_best.pth \
--optimized \
--degree 270 \
