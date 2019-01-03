set -x
#MODEL="resnet50_normalized_1111_dpdp"
#MODEL="resnet18_64_fine_long_sf"
MODEL="resnet18_grpy_2_fine_rotate_v100_2"

python3 eval1.py  \
--name ${MODEL}  \
--load  models/resnet/${MODEL}/model_best.pth \
--optimized \
