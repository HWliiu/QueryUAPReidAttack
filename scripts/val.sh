export PYTHONPATH=.reidattack

readonly dataset=Market1501
# readonly dataset=DukeMTMCreID
# readonly dataset=MSMT17
readonly nproc=1

# ABD:
resnet50_abd[1]=model_weights/reid_models/ABD/market1501_resnet50_abd_map=0.87.pth # market1501 weight path
resnet50_abd[2]=model_weights/reid_models/ABD/dukemtmc_resnet50_abd_map=0.77.pth # dukemtmc weight path
resnet50_abd[3]=model_weights/reid_models/ABD/msmt17_resnet50_abd_map=0.58.pth # msmt17 weight path

densenet121_abd[1]=model_weights/reid_models/ABD/market1501_densenet121_abd_map=0.87.pth
densenet121_abd[2]=model_weights/reid_models/ABD/dukemtmc_densenet121_abd_map=0.77.pth
densenet121_abd[3]=model_weights/reid_models/ABD/msmt17_densenet121_abd_map=0.59.pth

# AGW
resnet50_agw[1]=model_weights/reid_models/AGW/market1501_resnet50_agw_map=0.88.pth
resnet50_agw[2]=model_weights/reid_models/AGW/dukemtmc_resnet50_agw_map=0.79.pth
resnet50_agw[3]=model_weights/reid_models/AGW/msmt17_resnet50_agw_map=0.55.pth

# AP
resnet50_ap[1]=model_weights/reid_models/AP/market1501_resnet50_ap_map=0.89.pth
resnet50_ap[2]=model_weights/reid_models/AP/dukemtmc_resnet50_ap_map=0.80.pth
resnet50_ap[3]=model_weights/reid_models/AP/msmt17_resnet50_ap_map=0.57.pth

# DeepPersonReid:
mlfn_dpr[1]=model_weights/reid_models/DeepPersonReid/market1501_mlfn_dpr_map=0.74.pth
mlfn_dpr[2]=model_weights/reid_models/DeepPersonReid/dukemtmc_mlfn_dpr_map=0.64.pth
mlfn_dpr[3]=model_weights/reid_models/DeepPersonReid/msmt17_mlfn_dpr_map=0.39.pth

osnet_x1_0_dpr[1]=model_weights/reid_models/DeepPersonReid/market1501_osnet_x1_0_dpr_map=0.83.pth
osnet_x1_0_dpr[2]=model_weights/reid_models/DeepPersonReid/dukemtmc_osnet_x1_0_dpr_map=0.72.pth
osnet_x1_0_dpr[3]=model_weights/reid_models/DeepPersonReid/msmt17_osnet_x1_0_dpr_map=0.48.pth

# ReidStrongBaseline:
convnext_tiny_bot[1]=model_weights/reid_models/ReidStrongBaseline/convnext_tiny/market1501_convnext_tiny_bot_map=0.82.pth
convnext_tiny_bot[2]=model_weights/reid_models/ReidStrongBaseline/convnext_tiny/dukemtmc_convnext_tiny_bot_map=0.74.pth
convnext_tiny_bot[3]=model_weights/reid_models/ReidStrongBaseline/convnext_tiny/msmt17_convnext_tiny_bot_map=0.46.pth

densnet121_bot[1]=model_weights/reid_models/ReidStrongBaseline/densnet121/market1501_densnet121_bot_map=0.82.pth
densnet121_bot[2]=model_weights/reid_models/ReidStrongBaseline/densnet121/dukemtmc_densnet121_bot_map=0.74.pth
densnet121_bot[3]=model_weights/reid_models/ReidStrongBaseline/densnet121/msmt17_densnet121_bot_map=0.50.pth

efficientnet_b0_bot[1]=model_weights/reid_models/ReidStrongBaseline/efficientnet_b0/market1501_efficientnet_b0_bot_map=0.80.pth
efficientnet_b0_bot[2]=model_weights/reid_models/ReidStrongBaseline/efficientnet_b0/dukemtmc_efficientnet_b0_bot_map=0.70.pth
efficientnet_b0_bot[3]=model_weights/reid_models/ReidStrongBaseline/efficientnet_b0/msmt17_efficientnet_b0_bot_map=0.41.pth

inceptionv3_bot[1]=model_weights/reid_models/ReidStrongBaseline/inceptionv3/market1501_inceptionv3_bot_map=0.77.pth
inceptionv3_bot[2]=model_weights/reid_models/ReidStrongBaseline/inceptionv3/dukemtmc_inceptionv3_bot_map=0.68.pth
inceptionv3_bot[3]=model_weights/reid_models/ReidStrongBaseline/inceptionv3/msmt17_inceptionv3_bot_map=0.38.pth

mobilenetv2_bot[1]=model_weights/reid_models/ReidStrongBaseline/mobilenetv2/market1501_mobilenetv2_bot_map=0.78.pth
mobilenetv2_bot[2]=model_weights/reid_models/ReidStrongBaseline/mobilenetv2/dukemtmc_mobilenetv2_bot_map=0.68.pth
mobilenetv2_bot[3]=model_weights/reid_models/ReidStrongBaseline/mobilenetv2/msmt17_mobilenetv2_bot_map=0.40.pth

regnet_x_1_6gf_bot[1]=model_weights/reid_models/ReidStrongBaseline/regnet_x_1_6gf/market1501_regnet_x_1_6gf_bot_map=0.83.pth
regnet_x_1_6gf_bot[2]=model_weights/reid_models/ReidStrongBaseline/regnet_x_1_6gf/dukemtmc_regnet_x_1_6gf_bot_map=0.75.pth
regnet_x_1_6gf_bot[3]=model_weights/reid_models/ReidStrongBaseline/regnet_x_1_6gf/msmt17_regnet_x_1_6gf_bot_map=0.49.pth

resnet50_bot[1]=model_weights/reid_models/ReidStrongBaseline/resnet50/market1501_resnet50_bot_map=0.86.pth
resnet50_bot[2]=model_weights/reid_models/ReidStrongBaseline/resnet50/dukemtmc_resnet50_bot_map=0.76.pth
resnet50_bot[3]=model_weights/reid_models/ReidStrongBaseline/resnet50/msmt17_resnet50_bot_map=0.50.pth

resnet50_ibn_a_bot[1]=model_weights/reid_models/ReidStrongBaseline/resnet50_ibn_a/market1501_resnet50_ibn_a_bot_map=0.87.pth
resnet50_ibn_a_bot[2]=model_weights/reid_models/ReidStrongBaseline/resnet50_ibn_a/dukemtmc_resnet50_ibn_a_bot_map=0.79.pth
resnet50_ibn_a_bot[3]=model_weights/reid_models/ReidStrongBaseline/resnet50_ibn_a/msmt17_resnet50_ibn_a_bot_map=0.56.pth

se_resnet50_bot[1]=model_weights/reid_models/ReidStrongBaseline/se_resnet50/market1501_se_resnet50_bot_map=0.86.pth
se_resnet50_bot[2]=model_weights/reid_models/ReidStrongBaseline/se_resnet50/dukemtmc_se_resnet50_bot_map=0.77.pth
se_resnet50_bot[3]=model_weights/reid_models/ReidStrongBaseline/se_resnet50/msmt17_se_resnet50_bot_map=0.46.pth

se_resnext50_bot[1]=model_weights/reid_models/ReidStrongBaseline/se_resnext50/market1501_se_resnext50_bot_map=0.87.pth
se_resnext50_bot[2]=model_weights/reid_models/ReidStrongBaseline/se_resnext50/dukemtmc_se_resnext50_bot_map=0.78.pth
se_resnext50_bot[3]=model_weights/reid_models/ReidStrongBaseline/se_resnext50/msmt17_se_resnext50_bot_map=0.55.pth

senet154_bot[1]=model_weights/reid_models/ReidStrongBaseline/senet154/market1501_senet154_bot_map=0.86.pth
senet154_bot[2]=model_weights/reid_models/ReidStrongBaseline/senet154/dukemtmc_senet154_bot_map=0.78.pth
senet154_bot[3]=model_weights/reid_models/ReidStrongBaseline/senet154/msmt17_senet154_bot_map=0.58.pth

shufflenetv2_bot[1]=model_weights/reid_models/ReidStrongBaseline/shufflenetv2/market1501_shufflenetv2_bot_map=0.75.pth
shufflenetv2_bot[2]=model_weights/reid_models/ReidStrongBaseline/shufflenetv2/dukemtmc_shufflenetv2_bot_map=0.67.pth
shufflenetv2_bot[3]=model_weights/reid_models/ReidStrongBaseline/shufflenetv2/msmt17_shufflenetv2_bot_map=0.34.pth

vgg19_bot[1]=model_weights/reid_models/ReidStrongBaseline/vgg19/market1501_vgg19_bot_map=0.73.pth
vgg19_bot[2]=model_weights/reid_models/ReidStrongBaseline/vgg19/dukemtmc_vgg19_bot_map=0.62.pth
vgg19_bot[3]=model_weights/reid_models/ReidStrongBaseline/vgg19/msmt17_vgg19_bot_map=0.34.pth

# Transreid:
vit_base[1]=model_weights/reid_models/TransReID/market1501_vit_base_map=0.87.pth
vit_base[2]=model_weights/reid_models/TransReID/dukemtmc_vit_base_map=0.79.pth
vit_base[3]=model_weights/reid_models/TransReID/msmt17_vit_base_map=0.62.pth

vit_transreid[1]=model_weights/reid_models/TransReID/market1501_vit_transreid_map=0.89.pth
vit_transreid[2]=model_weights/reid_models/TransReID/dukemtmc_vit_transreid_map=0.82.pth
vit_transreid[3]=model_weights/reid_models/TransReID/msmt17_vit_transreid_map=0.68.pth

deit_transreid[1]=model_weights/reid_models/TransReID/market1501_deit_transreid_map=0.88.pth
deit_transreid[2]=model_weights/reid_models/TransReID/dukemtmc_deit_transreid_map=0.82.pth
deit_transreid[3]=model_weights/reid_models/TransReID/msmt17_deit_transreid_map=0.66.pth

# models=(resnet50_abd densenet121_abd resnet50_agw resnet50_ap mlfn_dpr osnet_x1_0_dpr convnext_tiny_bot \
#         densnet121_bot efficientnet_b0_bot inceptionv3_bot mobilenetv2_bot regnet_x_1_6gf_bot resnet50_bot\
#         resnet50_ibn_a_bot se_resnet50_bot se_resnext50_bot senet154_bot shufflenetv2_bot vgg19_bot vit_base\
#         vit_transreid deit_transreid)
models=(resnet50_bot inceptionv3_bot resnet50_ibn_a_bot se_resnext50_bot vit_base \
        resnet50_abd resnet50_agw resnet50_ap osnet_x1_0_dpr vit_transreid)

function run(){
    model_name=$1
    weight_path=$2
    # if model_name ends with "transreid"
    if [[ $model_name =~ ^.*transreid$ || $model_name =~ ^vit.*$ ]]
    then
        torchrun --nproc_per_node $nproc reidattack/main.py --config-file configs/attack_eval.yml DATA.DATASET.TEST_NAME $dataset \
            MODULE.TARGET_MODEL.NAME $model_name  MODULE.TARGET_MODEL.WEIGHT $weight_path \
            DATA.TRANSFORM.NORM.PIXEL_MEAN "[0.5, 0.5, 0.5]" DATA.TRANSFORM.NORM.PIXEL_STD  "[0.5, 0.5, 0.5]"
    else 
        torchrun --nproc_per_node $nproc reidattack/main.py --config-file configs/attack_eval.yml DATA.DATASET.TEST_NAME $dataset \
            MODULE.TARGET_MODEL.NAME $model_name  MODULE.TARGET_MODEL.WEIGHT $weight_path
    fi
    echo "\033[46;37m================================================================================\033[0m"
    return 0
}

for m in ${models[@]}
do
    model_name=$m
    case $dataset in 
        Market1501)
            weight_path=`eval echo '$'$m'[1]'`
            ;;
        DukeMTMCreID)
            weight_path=`eval echo '$'$m'[2]'`
            ;;
        MSMT17)
            weight_path=`eval echo '$'$m'[3]'`
            ;;
        *)
            echo "Unknown dataset: $dataset"
            exit 1
            ;;
    esac
    run $model_name $weight_path
done
