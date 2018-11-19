set -x
MODEL="stn2_360"

python3 eval.py  \
--name ${MODEL}_6  \
--load  models/${MODEL}/model_42*
