# The overwrite of datasets

## Table of Contents
- [MSR-VTT](#msr-vtt)
- [VATEX](#vatex)
- [TGIF](#tgif)
- [TRECVID AVS](#trecvid-avs)
- [References](#references)

## MSR-VTT

The MSR-VTT dataset [1], originally developed for video captioning, consists of 10k web video clips and 200k natural sentences describing the visual content of the clips. The number of sentences per clip is 20. Recently, MSR-VTT has become a popular benchmark for video-text retrieval tasks. For this dataset, we notice there are three distinct editions of data partition in the literature [1], [2], [3].


|  Partitions     | Train |  Val | Test |
| ------------  | ------------ | ------------ |------------ |
| Official [1]       |  6,513 clips, 130,260 sentences  |  497 clips, 9,940 sentences  |  2,990 clips, 59,800 sentences  |
| Test1k-Miech [2]   |  6,656 clips, 133,120 sentences  |  1,000 clips, 1,000 sentences |  1,000 clips, 1,000 sentences |
| Test1k-Yu [3]      |  7,010 clips, 140,200 sentences  |  1,000 clips, 1,000 sentences |  1,000 clips, 1,000 sentences |

Note that as the last two data partitions provide no validation set, we build a validation set by randomly sample 1,000 clips from MSR-VTT with [2], [3] excluded, respectively.



## VATEX

VATEX [4] a large-scale multilingual video description dataset. Each video, collected for YouTube, has a duration of 10 seconds. Per video there are 10 English sentences and 10 Chinese sentences to describe the corresponding video content. Here, we only use the English sentences. We adopt the dataset partition provided by [5].

|       | Train |  Val | Test |
| ------------  | ------------ | ------------ |------------ |
| VATEX       |  25,991 clips, 259,910 sentences   |  1,500 clips, 15,000 sentences  |  1,500 clips, 15,000 sentences  |



## TGIF

TGIF [9] is originally developed for the captioning task, while now also popular for the text-to-video retrieval task. It contains 100K animated GIFs collected from Tumblr, and 120K natural language sentences annotated via crowdsourcing. Each GIF is associated with 1-3 sentences. For  this  dataset, we  notice  there  are  two  distinct  editions  of  data  split  in the  literature [10],  [11].
|       | Train |  Val | Test |
| ------------  | ------------ | ------------ |------------ |
| TGIF-Li [10]      |  78,799 clips, 79,632 sentences   |  10,705 clips, 10,828 sentences  |  11,351 clips, 34,074 sentences  |
| TGIF-Chen [11]      |  79,451 clips, 80,295 sentences   |  10,651 clips, 10,774 sentences  |  11,310 clips, 33,951 sentences  |




## TRECVID AVS

IACC.3 dataset is the largest test bed for video retrieval by text to this date, which developed for TRECVID (Ad-hoc Video Search) AVS 2016, 2017 and 2018 task [6-8]. Given an ad-hoc query, e.g., `Find shots of military personnel interacting with protesters`, the task is to return for the query a list of 1,000 shots from the test collection ranked according to their likelihood of containing the given query. Per year TRECVID specifies 30 distinct queries of varied complexity. As TRECVID does not specify training data for the AVS task, we train the dual encoding network using the joint collection of MSR-VTT [1] and TGIF [9], use the `tv2016train` as the validation set.

| Split      |  Datasets     |  #Sentences    | #Videos Clips |
| ---------- | ------------  |  ------------  | ------------ |
| Train      | MSR-VTT       |     200,000    |   10,000    |
| Train      | TGIF          |     124,534    |   100,855   |
| Val        | tv2016train   |      400       |    200      |
| Test       | TRECVID AVS 2016      |   30 test queries  |   -   |
| Test       | TRECVID AVS 2017      |   30 test queries  |   -   |
| Test       | TRECVID AVS 2018      |   30 test queries  |   -   |
| Test       | IACC.3        |   -            |   335,944   |




## References

[1] J. Xu, T. Mei, T. Yao, and Y. Rui, “MSR-VTT: A large video description dataset for bridging video and language,” in CVPR, 2016.

[2] A. Miech, I. Laptev, and J. Sivic, “Learning a text-video embed- ding from incomplete and heterogeneous data,” arXiv preprint arXiv:1804.02516, 2018.

[3] Y. Yu, J. Kim, and G. Kim, “A joint sequence fusion model for video question answering and retrieval,” in ECCV, 2018.

[4] X. Wang, J. Wu, J. Chen, L. Li, Y.-F. Wang, and W. Y. Wang, “VATEX: A large-scale, high-quality multilingual dataset for video- and-language research,” in ICCV, 2019.

[5] S. Chen, Y. Zhao, Q. Jin, and Q. Wu, “Fine-grained video-text retrieval with hierarchical graph reasoning,” in CVPR, 2020.

[6] G. Awad, et al., “TRECVID 2016: Evaluating video search, video event detection, localization, and hyperlinking,” in TRECVID Workshop, 2016.

[7] G. Awad, et al., “TRECVID 2017: Evaluating ad-hoc and instance video search, events detection, video captioning and hyperlinking,” in TRECVID Workshop, 2017.

[8] G. Awad, et al., “TRECVID 2018: Benchmarking video activity detection, video captioning and matching, video storytelling linking and video search,” in TRECVID Workshop, 2018.

[9] Y. Li, Y. Song, L. Cao, J. Tetreault, L. Goldberg, A. Jaimes, and J. Luo, “TGIF: A new dataset and benchmark on animated GIF description,” in CVPR, 2016.

[10] Li,  F.  Zhou,  C.  Xu,  J.  Ji,  and  G.  Yang,  “Sea:  Sentence  encoderassembly for video retrieval by textual queries,”IEEE Transactions on Multimedia, 2020.

[11]  S. Chen, Y. Zhao, Q. Jin, and Q. Wu, “Fine-grained video-text retrievalwith  hierarchical  graph  reasoning,”  in CVPR,  2020.