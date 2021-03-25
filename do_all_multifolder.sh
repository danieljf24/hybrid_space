collectionStrt=multiple
trainCollection=$1
valCollection=$2
testCollection=$3
space=$4
visual_feature=$5
rootpath=$6
overwrite=0


# training
gpu=0
CUDA_VISIBLE_DEVICES=$gpu python trainer.py --rootpath $rootpath --overwrite $overwrite --max_violation --text_norm --visual_norm \
                                            --collectionStrt $collectionStrt --trainCollection $trainCollection --valCollection $valCollection --testCollection $testCollection\
                                            --visual_feature $visual_feature --space $space
