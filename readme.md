## Query UAP ReID Attack
### Prerequisites
accelerate\=\=0.12

einops\=\=0.4.1

kornia\=\=0.6.4

pytorch\=\=1.11.0

torchvision\=\=0.12.0

yacs\=\=0.1.8

tqdm\=\=4.64.0 

scipy\=\=1.9.1

gdown\=\=4.5.1


### Prepare data
Download [Market1501](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html) and [DukeMTMC](https://github.com/layumi/DukeMTMC-reID_evaluation#download-dataset) put in [datasets/market1501](datasets/market1501) and [datasets/dukemtmc-reid](datasets/dukemtmc-reid) respectively.
### Prepare pretrained models
Download the pretrained weights from [there](https://drive.google.com/drive/folders/1FLdKsg4i0fTGKIYe6TWkSWj0yE2KqK9E?usp=sharing) and put it in [model_weights](model_weights) according to the directory structure.
### Run
run Query UAP attack: `. ./scripts/query_uap_run.sh`

run DITIM attack: `. ./scripts/ditim_run.sh`

run MUAP attack: `. ./scripts/muap_run.sh`

run Bandits attack: `. ./scripts/bandits_run.sh`

run Bandits UAP attack: `. ./scripts/bandits_uap_run.sh`  
