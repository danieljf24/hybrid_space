rootpath=/media/daniel/disk2/daniel/VisualSearch
testCollection=iacc.3
topic_set=tv18
overwrite=1

score_file=/media/daniel/disk2/daniel/VisualSearch/iacc.3/results/tv18.avs.txt/tgif-msrvtt10k/tv2016train/hybrid_hist_dual_encoding_concate_full_dp_0.2_measure_cosine_hist_tag_vocab_size_512_tag_weighted/vocab_word_vocab_5_word_dim_500_text_rnn_size_512_text_norm_True_kernel_sizes_2-3-4_num_512/visual_feature_resnext-101_resnet152-13k_visual_rnn_size_512_visual_norm_True_kernel_sizes_2-3-4-5_num_512/mapping_text_0-1536_img_0-1536/loss_func_both_mrl_tag_margin_0.2_0.2_direction_all_max_violation_True_cost_style_sum_w1_1.00_w2_1.00/optimizer_adam_lr_0.0001_decay_0.99_grad_clip_2.0_val_metric_recall/runs_pami_0/model_best.pth.tar/id.sent.score_0.8.txt

bash do_txt2xml.sh $testCollection $score_file $topic_set $overwrite $rootpath
python trec_eval.py ${score_file}.xml --rootpath $rootpath --collection $testCollection --edition $topic_set --overwrite $overwrite


