import os
import sys
import json
import time
import torch
import pickle
import logging
import argparse
import numpy as np

import evaluation
from model import get_model
from validate import norm_score

import util.data_provider as data
from util.vocab import Vocabulary
from util.text2vec import get_text_encoder


from basic.util import read_dict
from basic.constant import ROOT_PATH
from basic.bigfile import BigFile
from basic.common import makedirsforfile, checkToSkip
from basic.generic_utils import Progbar

def parse_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH, help='path to datasets. (default: %s)'%ROOT_PATH)
    parser.add_argument('--testCollection', type=str, help='test collection')
    parser.add_argument('--collectionStrt', type=str, default='multiple', help='collection structure (single|multiple)')
    parser.add_argument('--split', default='test', type=str, help='split, only for single-folder collection structure (val|test)')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1],  help='overwrite existed file. (default: 0)')
    parser.add_argument('--batch_size', default=128, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=5, type=int, help='Number of data loader workers.')
    parser.add_argument('--logger_name', default='runs', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--checkpoint_name', default='model_best.pth.tar', type=str, help='name of checkpoint (default: model_best.pth.tar)')
    parser.add_argument('--query_sets', type=str, default='tv16.avs.txt',  help='test query sets,  tv16.avs.txt,tv17.avs.txt,tv18.avs.txt for TRECVID 16/17/18.')

    args = parser.parse_args()
    return args

def eval_avs(t2v_matrix, query_ids, video_ids, pred_result_file, rootpath, testCollection, query_set):

    inds = np.argsort(t2v_matrix, axis=1)
    with open(pred_result_file, 'w') as fout:
       for index in range(inds.shape[0]):
           ind = inds[index][::-1]
           fout.write(query_ids[index]+' '+' '.join([video_ids[i]+' %s'%t2v_matrix[index][i]
               for i in ind])+'\n')

    templete = ''.join(open( 'tv-avs-eval/TEMPLATE_do_eval.sh').readlines())
    striptStr = templete.replace('@@@rootpath@@@', rootpath)
    striptStr = striptStr.replace('@@@testCollection@@@', testCollection)
    striptStr = striptStr.replace('@@@topic_set@@@', query_set.split('.')[0])
    striptStr = striptStr.replace('@@@overwrite@@@', str(1))
    striptStr = striptStr.replace('@@@score_file@@@', pred_result_file)

    runfile = 'do_eval_%s.sh' % testCollection
    open(os.path.join('tv-avs-eval', runfile), 'w').write(striptStr + '\n')
    os.system('cd tv-avs-eval; chmod +x %s; bash %s; cd -' % (runfile, runfile))


def main():
    opt = parse_args()
    logging.info(json.dumps(vars(opt), indent=2))

    rootpath = opt.rootpath
    testCollection = opt.testCollection
    assert collectionStrt == "multiple"
    resume = os.path.join(opt.logger_name, opt.checkpoint_name)

    if not os.path.exists(resume):
        logging.info(resume + ' not exists.')
        sys.exit(0)

    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    best_rsum = checkpoint['best_rsum']
    logging.info("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
          .format(resume, start_epoch, best_rsum))
    options = checkpoint['opt']

    trainCollection = options.trainCollection
    valCollection = options.valCollection

    visual_feat_file = BigFile(os.path.join(rootpath, testCollection, 'FeatureData', options.visual_feature))
    assert options.visual_feat_dim == visual_feat_file.ndims
    video2frame = read_dict(os.path.join(rootpath, testCollection, 'FeatureData', options.visual_feature, 'video2frames.txt'))
    vid_data_loader = data.get_vis_data_loader(visual_feat_file, opt.batch_size, opt.workers, video2frame)
    vis_embs = None

    # set bow vocabulary and encoding
    bow_vocab_file = os.path.join(rootpath, options.trainCollection, 'TextData', 'vocabulary', 'bow', options.vocab+'.pkl')
    bow_vocab = pickle.load(open(bow_vocab_file, 'rb'))
    bow2vec = get_text_encoder('bow')(bow_vocab)
    options.bow_vocab_size = len(bow_vocab)

    # set rnn vocabulary
    rnn_vocab_file = os.path.join(rootpath, options.trainCollection, 'TextData', 'vocabulary', 'rnn', options.vocab+'.pkl')
    rnn_vocab = pickle.load(open(rnn_vocab_file, 'rb'))
    options.vocab_size = len(rnn_vocab)

    model = get_model(options.model)(options)
    model.load_state_dict(checkpoint['model'])
    model.val_start()

    output_dir = resume.replace(trainCollection, testCollection)
    for query_set in opt.query_sets.strip().split(','):
        output_dir_tmp = output_dir.replace(valCollection, '%s/%s/%s' % (query_set, trainCollection, valCollection))
        output_dir_tmp = output_dir_tmp.replace('/%s/' % options.cv_name, '/results/')
        pred_result_file = os.path.join(output_dir_tmp, 'id.sent.score.txt')
        logging.info(pred_result_file)
        if checkToSkip(pred_result_file, opt.overwrite):
            sys.exit(0)
        makedirsforfile(pred_result_file)

        # query data loader
        query_file = os.path.join(rootpath, testCollection, 'TextData', query_set)
        query_loader = data.get_txt_data_loader(query_file, rnn_vocab, bow2vec, opt.batch_size, opt.workers)
        
        # encode videos
        if vis_embs is None:
            start = time.time()
            if options.space == 'hybrid':
                video_embs, video_tag_probs, video_ids = evaluation.encode_text_or_vid_tag_hist_prob(model.embed_vis, vid_data_loader)
            else:
                video_embs, video_ids = evaluation.encode_text_or_vid(model.embed_vis, vid_data_loader)
            logging.info("encode video time: %.3f s" % (time.time()-start))
        
        # encode text
        start = time.time()
        if options.space == 'hybrid':
            query_embs, query_tag_probs, query_ids = evaluation.encode_text_or_vid_tag_hist_prob(model.embed_txt, query_loader)
        else:
            query_embs, query_ids = evaluation.encode_text_or_vid(model.embed_txt, query_loader)
        logging.info("encode text time: %.3f s" % (time.time()-start))
        

        if options.space == 'hybrid':
            t2v_matrix_1 = evaluation.cal_simi(query_embs, video_embs)
            # eval_avs(t2v_matrix_1, query_ids, video_ids, pred_result_file, rootpath, testCollection, query_set)

            t2v_matrix_2 = evaluation.cal_simi(query_tag_probs, video_tag_probs)
            # pred_result_file = os.path.join(output_dir_tmp, 'id.sent.score_2.txt')
            # eval_avs(t2v_matrix_2, query_ids, video_ids, pred_result_file, rootpath, testCollection, query_set)

            t2v_matrix_1 = norm_score(t2v_matrix_1)
            t2v_matrix_2 = norm_score(t2v_matrix_2)
            for w in [0.8]:
                print("\n")
                t2v_matrix = w * t2v_matrix_1 + (1-w) * t2v_matrix_2
                pred_result_file = os.path.join(output_dir_tmp, 'id.sent.score_%.1f.txt' % w)
                eval_avs(t2v_matrix, query_ids, video_ids, pred_result_file, rootpath, testCollection, query_set)
        else:
            t2v_matrix_1 = evaluation.cal_simi(query_embs, video_embs)
            eval_avs(t2v_matrix_1, query_ids, video_ids, pred_result_file, rootpath, testCollection, query_set)


if __name__ == '__main__':
    main()