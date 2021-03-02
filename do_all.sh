rootpath=$HOME/VisualSearch
collection=$1
space=$2
visual_feature=$3
overwrite=0


# training
gpu=2
CUDA_VISIBLE_DEVICES=$gpu python trainer.py --rootpath $rootpath --overwrite $overwrite --max_violation --text_norm --visual_norm \
                                            --collection $collection --visual_feature $visual_feature --space $space