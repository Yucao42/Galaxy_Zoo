set -x
#MODEL="resnet50_normalized_1111_dpdp"
#MODEL="resnet18_64_fine_long_sf"
MODEL="resnet18_64_fine_discrete_rotation_225_fine"

python3 eval_seq.py  \
--name ${MODEL}  \
--load  models/resnet/${MODEL}/model_best.pth \
--optimized \
