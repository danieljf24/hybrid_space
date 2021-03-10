rootpath=$HOME/VisualSearch
overwrite=0

featname_1=pyresnet-152_imagenet11k,flatten0_output,os
featname_2=pyresnext-101_rbps13k,flatten0_output,os

collection=tgif-msrvtt10k
sub_collections=tgif@msrvtt10k

python util/combine_features.py $collection $featname_1 $featname_2 \
    --overwrite $overwrite --rootpath $rootpath --sub_collections ${sub_collections}


for collection in tgif msrvtt10k tv2016train
do
	python util/combine_features.py $collection $featname_1 $featname_2 \
    --overwrite $overwrite --rootpath $rootpath --sub_collections ${collection}
done