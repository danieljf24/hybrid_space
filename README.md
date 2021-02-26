# Dual Encoding for Video Retrieval by Text

Source code of our TPAMI'21  paper [Dual Encoding for Video Retrieval by Text](https://arxiv.org/abs/2009.05381) and CVPR'19 paper [Dual Encoding for Zero-Example Video Retrieval](https://openaccess.thecvf.com/content_CVPR_2019/html/Dong_Dual_Encoding_for_Zero-Example_Video_Retrieval_CVPR_2019_paper.html).

![image](dual_encoding.jpg)

## Requirements

#### Environments
* **Ubuntu** 16.04
* **CUDA** 10.1
* **Python** 3.8
* **PyTorch** 1.5.1

We used Anaconda to setup a deep learning workspace that supports PyTorch.
Run the following script to install the required packages.
```shell
conda create --name ws_dual_py3 python=3.8
conda activate ws_dual_py3
git clone https://github.com/danieljf24/hybrid_space.git
cd hybrid_space
pip install -r requirements.txt
conda deactivate
```

## Dual Encoding on MSRVTT10K
### Required Data
Run `do_get_dataset.sh` or the following script to download and extract MSR-VTT ([msrvtt10k-resnext101_resnet152.tar.gz(4.3G)](xx)) dataset and a pre-trained word2vec ([vec500flickr30m.tar.gz(3.0G)](http://lixirong.net/data/w2vv-tmm2018/word2vec.tar.gz). The data can also be downloaded from Baidu pan ([url](https://pan.baidu.com/s/1lg23K93lVwgdYs5qnTuMFg), password:p3p0) or Google drive ([url](https://drive.google.com/drive/folders/1TEIjErztZNQAi6AyNu9cK5STwo74oI8I?usp=sharing)).
The extracted data is placed in `$HOME/VisualSearch/`.
```shell
ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH && cd $ROOTPATH

# download and extract dataset
wget xx
tar zxf xx

# download and extract pre-trained word2vec
wget http://lixirong.net/data/w2vv-tmm2018/word2vec.tar.gz
tar zxf word2vec.tar.gz
```

### Model Training and Evaluation
Run the following script to train and evaluate `Dual Encoding` network with hybrid space on the `official` partition of MSR-VTT. The video features are the concatenation of ResNeXt-101 and ResNet-152 features.
```shell
conda activate ws_dual_py3
./do_all.sh msrvtt10k hybrid resnext101-resnet152
conda deactive
```
Running the script will do the following things:
1. Train `Dual Encoding` network with hybrid space and select a checkpoint that performs best on the validation set as the final model. Notice that we only save the best-performing checkpoint on the validation set to save disk space.
2. Evaluate the final model on the test set.

If you would like to train `Dual Encoding` network with latent space (Conference Version), please run the following scrip:
```shell
conda activate ws_dual_py3
./do_all.sh msrvtt10k latent resnext101-resnet152
conda deactive
```

To train the model on the `Test1k-Miech` partition and `Test1k-Yu` partition of MSR-VTT, please run the following scrip:
```shell
conda activate ws_dual_py3
./do_all.sh msrvtt10kmiech hybrid resnext101-resnet152
./do_all.sh msrvtt10kyu hybrid resnext101-resnet152
conda deactive
```

### Expected Performance (TODO)
Run the following script to evaluate our trained [model(xxM)](xx) on MSR-VTT.
```shell
source ~/ws_dual/bin/activate
MODELDIR=$HOME/VisualSearch/msrvtt10ktrain/cvpr_2019
mkdir -p $MODELDIR
wget -P $MODELDIR http://lixirong.net/data/cvpr2019/model_best.pth.tar
CUDA_VISIBLE_DEVICES=0 python tester.py msrvtt10ktest --logger_name $MODELDIR
deactive
```

The expected performance of Dual Encoding on MSR-VTT is as follows. Notice that due to random factors in SGD based training, the numbers differ slightly from those reported in the paper.

|  | R@1 | R@5 | R@10 | Med r |	mAP |
| ------------- | ------------- | ------------- | ------------- |  ------------- | ------------- |
| Text-to-Video | 7.6  | 22.4 | 31.8 | 33 | 0.155 |
| Video-to-Text | 12.8 | 30.3 | 42.4 | 16 | 0.065 |




## Dual Encoding on VATEX

### Required Data
Download VATEX dataset ([vatex-i3d.tar.gz(3.0G)](xx)) and a pre-trained word2vec ([vec500flickr30m.tar.gz(3.0G)](http://lixirong.net/data/w2vv-tmm2018/word2vec.tar.gz)). The data can also be downloaded from Baidu pan ([url](xx), password:xx) or Google drive ([url](https://drive.google.com/drive/folders/1TEIjErztZNQAi6AyNu9cK5STwo74oI8I?usp=sharing)).
Please extract data into `$HOME/VisualSearch/`.

### Model Training and Evaluation
Run the following script to train and evaluate `Dual Encoding` network with hybrid space on VATEX.
```shell
conda activate ws_dual_py3
./do_all.sh vatex hybrid
conda deactive
```


## Dual Encoding on Ad-hoc Video Search (AVS) (TODO)

### Data

The following three datasets are used for training, validation and testing: tgif-msrvtt10k, tv2016train and iacc.3. For more information about these datasets, please refer to https://github.com/li-xirong/avs.

Run the following scripts to download and extract these datasets. The extracted data is placed in `$HOME/VisualSearch/`.

#### Sentence data
* Sentences: [tgif-msrvtt10k](http://lixirong.net/data/mm2019/tgif-msrvtt10k-sent.tar.gz), [tv2016train](http://lixirong.net/data/mm2019/tv2016train-sent.tar.gz)
* TRECVID 2016 / 2017 / 2018 AVS topics and ground truth:  [iacc.3](http://lixirong.net/data/mm2019/iacc.3-avs-topics.tar.gz)

#### Frame-level feature data
* 2048-dim ResNeXt-101: [tgif](http://39.104.114.128/avs/tgif_ResNext-101.tar.gz)(7G), [msrvtt10k](http://39.104.114.128/avs/msrvtt10k_ResNext-101.tar.gz)(2G), [tv2016train](http://39.104.114.128/avs/tv2016train_ResNext-101.tar.gz)(42M), [iacc.3](http://39.104.114.128/avs/iacc.3_ResNext-101.tar.gz)(27G)

```shell
ROOTPATH=$HOME/VisualSearch
cd $ROOTPATH

# download and extract dataset
wget http://39.104.114.128/avs/tgif_ResNext-101.tar.gz
tar zxf tgif_ResNext-101.tar.gz

wget http://39.104.114.128/avs/msrvtt10k_ResNext-101.tar.gz
tar zvf msrvtt10k_ResNext-101.tar

wget http://39.104.114.128/avs/tv2016train_ResNext-101.tar.gz
tar zvf tv2016train_ResNext-101.tar.gz

wget http://39.104.114.128/avs/iacc.3_ResNext-101.tar.gz
tar zvf iacc.3_ResNext-101.tar.gz

# combine feature of tgif and msrvtt10k
./do_combine_features.sh

```

### Train Dual Encoding model from scratch

```shell
source ~/ws_dual/bin/activate

trainCollection=tgif-msrvtt10k
visual_feature=pyresnext-101_rbps13k,flatten0_output,os

# Generate a vocabulary on the training set
./do_get_vocab.sh $trainCollection

# Generate video frame info
#./do_get_frameInfo.sh $trainCollection $visual_feature


# training and testing
./do_all_avs.sh 

deactive
```

## How to run Dual Encoding on another datasets?

Store the training, validation and test subset into three folders in the following structure respectively.
```shell
${subset_name}
├── FeatureData
│   └── ${feature_name}
│       ├── feature.bin
│       ├── shape.txt
│       └── id.txt
├── ImageSets
│   └── ${subset_name}.txt
└── TextData
    └── ${subset_name}.caption.txt

```

* `FeatureData`: video frame features. Using [txt2bin.py](https://github.com/danieljf24/simpleknn/blob/master/txt2bin.py) to convert video frame feature in the required binary format.
* `${subset_name}.txt`: all video IDs in the specific subset, one video ID per line.
* `${dsubset_name}.caption.txt`: caption data. The file structure is as follows, in which the video and sent in the same line are relevant.
```
video_id_1#1 sentence_1
video_id_1#2 sentence_2
...
video_id_n#1 sentence_k
...
```

You can run the following script to check whether the data is ready:
```shell
./do_format_check.sh ${train_set} ${val_set} ${test_set} ${rootpath} ${feature_name}
```
where `train_set`, `val_set` and `test_set` indicate the name of training, validation and test set, respectively, ${rootpath} denotes the path where datasets are saved and `feature_name` is the video frame feature name.


If you pass the format check, use the following script to train and evaluate Dual Encoding on your own dataset:
```shell
source ~/ws_dual/bin/activate
./do_all_own_data.sh ${train_set} ${val_set} ${test_set} ${rootpath} ${feature_name} ${caption_num} full
deactive
```

If training data of your task is relatively limited, we suggest dual encoding with level 2 and 3. Compared to the full edition, this version gives nearly comparable performance on MSR-VTT, but with less trainable parameters.
```shell
source ~/ws_dual/bin/activate
./do_all_own_data.sh ${train_set} ${val_set} ${test_set} ${rootpath} ${feature_name} ${caption_num} reduced
deactive
```


## References
If you find the package useful, please consider citing our TPAMI'21 or CVPR'19 paper:
```
@article{dong2021dual,
  title={Dual Encoding for Video Retrieval by Text},
  author={Dong, Jianfeng and Li, Xirong and Xu, Chaoxi and Yang, Xun and Yang, Gang and Wang, Xun and Wang, Meng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  doi = {10.1109/TPAMI.2021.3059295},
  year={2021}
}
```


```
@inproceedings{cvpr2019-dual-dong,
title = {Dual Encoding for Zero-Example Video Retrieval},
author = {Jianfeng Dong and Xirong Li and Chaoxi Xu and Shouling Ji and Yuan He and Gang Yang and Xun Wang},
booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2019},
}
```