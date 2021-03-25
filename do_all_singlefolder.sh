collectionStrt=single
collection=$1
space=$2
visual_feature=$3
rootpath=$4
overwrite=0


# training
gpu=0
CUDA_VISIBLE_DEVICES=$gpu python trainer.py --rootpath $rootpath --overwrite $overwrite --max_violation --text_norm --visual_norm \
                                            --collectionStrt $collectionStrt --collection $collection --visual_feature $visual_feature --space $space
