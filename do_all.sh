rootpath=$HOME/VisualSearch
collection=$1
space=$2
visual_feature=$3
overwrite=0

# Generate a vocabulary on the training set
# ./util/do_get_vocab.sh $collection $rootpath

# Generate concepts accoridng to video captions
# ./util/do_get_tags.sh $collection $rootpath


# training
gpu=2
CUDA_VISIBLE_DEVICES=$gpu python trainer.py --rootpath $rootpath --overwrite $overwrite --max_violation --text_norm --visual_norm \
                                            --collection $collection --visual_feature $visual_feature --space $space
