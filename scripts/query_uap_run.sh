export CUDA_VISIBLE_DEVICES=0

for loop in 1 2 3
do
    python reidattack/main.py --config-file configs/query_uap.yml \
        MODULE.TARGET_MODEL.NAME resnet50_bot  \
        DATA.DATASET.TEST_NAME Market1501 \
        MODULE.TARGET_MODEL.WEIGHT model_weights/reid_models/ReidStrongBaseline/resnet50/market1501_resnet50_bot_map=0.86.pth
    echo "\033[46;37m================================================================================\033[0m"
    python reidattack/main.py --config-file configs/query_uap.yml \
        MODULE.TARGET_MODEL.NAME inceptionv3_bot  \
        DATA.DATASET.TEST_NAME Market1501 \
        MODULE.TARGET_MODEL.WEIGHT model_weights/reid_models/ReidStrongBaseline/inceptionv3/market1501_inceptionv3_bot_map=0.77.pth
    echo "\033[46;37m================================================================================\033[0m"
    python reidattack/main.py --config-file configs/query_uap.yml \
        MODULE.TARGET_MODEL.NAME resnet50_ibn_a_bot  \
        DATA.DATASET.TEST_NAME Market1501 \
        MODULE.TARGET_MODEL.WEIGHT model_weights/reid_models/ReidStrongBaseline/resnet50_ibn_a/market1501_resnet50_ibn_a_bot_map=0.87.pth
    echo "\033[46;37m================================================================================\033[0m"
    python reidattack/main.py --config-file configs/query_uap.yml \
        MODULE.TARGET_MODEL.NAME se_resnext50_bot  \
        DATA.DATASET.TEST_NAME Market1501 \
        MODULE.TARGET_MODEL.WEIGHT model_weights/reid_models/ReidStrongBaseline/se_resnext50/market1501_se_resnext50_bot_map=0.87.pth
    echo "\033[46;37m================================================================================\033[0m"
    python reidattack/main.py --config-file configs/query_uap.yml \
        MODULE.TARGET_MODEL.NAME vit_base  \
        DATA.DATASET.TEST_NAME Market1501 \
        MODULE.TARGET_MODEL.WEIGHT model_weights/reid_models/TransReID/market1501_vit_base_map=0.87.pth \
        DATA.TRANSFORM.NORM.PIXEL_MEAN "[0.5, 0.5, 0.5]" DATA.TRANSFORM.NORM.PIXEL_STD  "[0.5, 0.5, 0.5]"
    echo "\033[46;37m================================================================================\033[0m"
    echo "\033[46;37m================================================================================\033[0m"
    echo "\033[46;37m================================================================================\033[0m"

    python reidattack/main.py --config-file configs/query_uap.yml \
        MODULE.TARGET_MODEL.NAME resnet50_bot  \
        DATA.DATASET.TEST_NAME DukeMTMCreID \
        MODULE.TARGET_MODEL.WEIGHT model_weights/reid_models/ReidStrongBaseline/resnet50/dukemtmc_resnet50_bot_map=0.76.pth
    echo "\033[46;37m================================================================================\033[0m"
    python reidattack/main.py --config-file configs/query_uap.yml \
        MODULE.TARGET_MODEL.NAME inceptionv3_bot  \
        DATA.DATASET.TEST_NAME DukeMTMCreID \
        MODULE.TARGET_MODEL.WEIGHT model_weights/reid_models/ReidStrongBaseline/inceptionv3/dukemtmc_inceptionv3_bot_map=0.68.pth
    echo "\033[46;37m================================================================================\033[0m"
    python reidattack/main.py --config-file configs/query_uap.yml \
        MODULE.TARGET_MODEL.NAME resnet50_ibn_a_bot  \
        DATA.DATASET.TEST_NAME DukeMTMCreID \
        MODULE.TARGET_MODEL.WEIGHT model_weights/reid_models/ReidStrongBaseline/resnet50_ibn_a/dukemtmc_resnet50_ibn_a_bot_map=0.79.pth
    echo "\033[46;37m================================================================================\033[0m"
    python reidattack/main.py --config-file configs/query_uap.yml \
        MODULE.TARGET_MODEL.NAME se_resnext50_bot  \
        DATA.DATASET.TEST_NAME DukeMTMCreID \
        MODULE.TARGET_MODEL.WEIGHT model_weights/reid_models/ReidStrongBaseline/se_resnext50/dukemtmc_se_resnext50_bot_map=0.78.pth
    echo "\033[46;37m================================================================================\033[0m"
    python reidattack/main.py --config-file configs/query_uap.yml \
        MODULE.TARGET_MODEL.NAME vit_base  \
        DATA.DATASET.TEST_NAME DukeMTMCreID \
        MODULE.TARGET_MODEL.WEIGHT model_weights/reid_models/TransReID/dukemtmc_vit_base_map=0.79.pth \
        DATA.TRANSFORM.NORM.PIXEL_MEAN "[0.5, 0.5, 0.5]" DATA.TRANSFORM.NORM.PIXEL_STD  "[0.5, 0.5, 0.5]"
    echo "\033[46;37m================================================================================\033[0m"
    echo "\033[46;37m================================================================================\033[0m"
    echo "\033[46;37m================================================================================\033[0m"

    python reidattack/main.py --config-file configs/query_uap.yml \
        MODULE.TARGET_MODEL.NAME resnet50_abd  \
        DATA.DATASET.TEST_NAME Market1501 \
        MODULE.TARGET_MODEL.WEIGHT model_weights/reid_models/ABD/market1501_resnet50_abd_map=0.87.pth
    echo "\033[46;37m================================================================================\033[0m"
    python reidattack/main.py --config-file configs/query_uap.yml \
        MODULE.TARGET_MODEL.NAME resnet50_agw  \
        DATA.DATASET.TEST_NAME Market1501 \
        MODULE.TARGET_MODEL.WEIGHT model_weights/reid_models/AGW/market1501_resnet50_agw_map=0.88.pth
    echo "\033[46;37m================================================================================\033[0m"
    python reidattack/main.py --config-file configs/query_uap.yml \
        MODULE.TARGET_MODEL.NAME resnet50_ap  \
        DATA.DATASET.TEST_NAME Market1501 \
        MODULE.TARGET_MODEL.WEIGHT model_weights/reid_models/AP/market1501_resnet50_ap_map=0.89.pth
    echo "\033[46;37m================================================================================\033[0m"
    python reidattack/main.py --config-file configs/query_uap.yml \
        MODULE.TARGET_MODEL.NAME osnet_x1_0_dpr  \
        DATA.DATASET.TEST_NAME Market1501 \
        MODULE.TARGET_MODEL.WEIGHT model_weights/reid_models/DeepPersonReid/market1501_osnet_x1_0_dpr_map=0.83.pth
    echo "\033[46;37m================================================================================\033[0m"
    python reidattack/main.py --config-file configs/query_uap.yml \
        MODULE.TARGET_MODEL.NAME vit_transreid  \
        DATA.DATASET.TEST_NAME Market1501 \
        MODULE.TARGET_MODEL.WEIGHT model_weights/reid_models/TransReID/market1501_vit_transreid_map=0.89.pth \
        DATA.TRANSFORM.NORM.PIXEL_MEAN "[0.5, 0.5, 0.5]" DATA.TRANSFORM.NORM.PIXEL_STD  "[0.5, 0.5, 0.5]"
    echo "\033[46;37m================================================================================\033[0m"
    echo "\033[46;37m================================================================================\033[0m"
    echo "\033[46;37m================================================================================\033[0m"

    python reidattack/main.py --config-file configs/query_uap.yml \
        MODULE.TARGET_MODEL.NAME resnet50_abd  \
        DATA.DATASET.TEST_NAME DukeMTMCreID \
        MODULE.TARGET_MODEL.WEIGHT model_weights/reid_models/ABD/dukemtmc_resnet50_abd_map=0.77.pth
    echo "\033[46;37m================================================================================\033[0m"
    python reidattack/main.py --config-file configs/query_uap.yml \
        MODULE.TARGET_MODEL.NAME resnet50_agw  \
        DATA.DATASET.TEST_NAME DukeMTMCreID \
        MODULE.TARGET_MODEL.WEIGHT model_weights/reid_models/AGW/dukemtmc_resnet50_agw_map=0.79.pth
    echo "\033[46;37m================================================================================\033[0m"
    python reidattack/main.py --config-file configs/query_uap.yml \
        MODULE.TARGET_MODEL.NAME resnet50_ap  \
        DATA.DATASET.TEST_NAME DukeMTMCreID \
        MODULE.TARGET_MODEL.WEIGHT model_weights/reid_models/AP/dukemtmc_resnet50_ap_map=0.80.pth
    echo "\033[46;37m================================================================================\033[0m"
    python reidattack/main.py --config-file configs/query_uap.yml \
        MODULE.TARGET_MODEL.NAME osnet_x1_0_dpr  \
        DATA.DATASET.TEST_NAME DukeMTMCreID \
        MODULE.TARGET_MODEL.WEIGHT model_weights/reid_models/DeepPersonReid/dukemtmc_osnet_x1_0_dpr_map=0.72.pth
    echo "\033[46;37m================================================================================\033[0m"
    python reidattack/main.py --config-file configs/query_uap.yml \
        MODULE.TARGET_MODEL.NAME vit_transreid  \
        DATA.DATASET.TEST_NAME DukeMTMCreID \
        MODULE.TARGET_MODEL.WEIGHT model_weights/reid_models/TransReID/dukemtmc_vit_transreid_map=0.82.pth \
        DATA.TRANSFORM.NORM.PIXEL_MEAN "[0.5, 0.5, 0.5]" DATA.TRANSFORM.NORM.PIXEL_STD  "[0.5, 0.5, 0.5]"
    echo "\033[46;37m================================================================================\033[0m"
    echo "\033[46;37m================================================================================\033[0m"
    echo "\033[46;37m================================================================================\033[0m"
    echo "\033[46;37m================================================================================\033[0m"
    echo "\033[46;37m================================================================================\033[0m"
    echo "\033[46;37m================================================================================\033[0m"
done
