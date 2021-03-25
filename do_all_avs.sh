trainCollection=tgif-msrvtt10k
valCollection=tv2016train
testCollection=iacc.3
collectionStrt=multiple
visual_feature=resnext101-resnet152

rootpath=$1
overwrite=0

model=dual_encoding
space=hybrid

gpu=1
CUDA_VISIBLE_DEVICES=$gpu python trainer.py --collectionStrt $collectionStrt  --trainCollection $trainCollection --valCollection $valCollection --testCollection $testCollection \
                   --max_violation --text_norm --visual_norm --rootpath $rootpath   --overwrite $overwrite\
                   --visual_feature $visual_feature --model $model --space $space \
  
