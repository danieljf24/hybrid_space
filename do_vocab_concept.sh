rootpath=$HOME/VisualSearch
collection=$1
overwrite=$2  # set to 1 if you would like to generate vocabulary and concepts by yourself

# Generate a vocabulary on the training set
./util/do_get_vocab.sh $collection $rootpath $overwrite

# Generate concepts according to video captions
./util/do_get_tags.sh $collection $rootpath $overwrite

