export CUDA_VISIBLE_DEVICES=3

python reidattack/main.py --config-file configs/muap.yml \
    MODULE.AGENT_MODELS.NAMES "['inceptionv3_bot',]"  \
    DATA.DATASET.TEST_NAME Market1501 \
    MODULE.AGENT_MODELS.WEIGHTS "['model_weights/reid_models/ReidStrongBaseline/inceptionv3/market1501_inceptionv3_bot_map=0.77.pth',]"
echo "\033[46;37m================================================================================\033[0m"

python reidattack/main.py --config-file configs/muap.yml \
    MODULE.AGENT_MODELS.NAMES "['inceptionv3_bot',]"\
    DATA.DATASET.TEST_NAME DukeMTMCreID \
    MODULE.AGENT_MODELS.WEIGHTS "['model_weights/reid_models/ReidStrongBaseline/inceptionv3/dukemtmc_inceptionv3_bot_map=0.68.pth',]"
echo "\033[46;37m================================================================================\033[0m"

