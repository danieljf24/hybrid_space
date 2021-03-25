import os
import sys
import time
import json
import torch
import shutil
import pickle
import logging
import argparse

import validate
import tensorboard_logger as tb_logger
from model import get_model, get_we_parameter

import util.tag_data_provider as data
from util.vocab import Vocabulary
from util.text2vec import get_text_encoder

from basic.constant import ROOT_PATH
from basic.bigfile import BigFile
from basic.common import makedirsforfile, checkToSkip
from basic.util import read_dict, AverageMeter, LogCollector, log_config
from basic.generic_utils import Progbar


def parse_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH,
                        help='path to datasets. (default: %s)'%ROOT_PATH)
    parser.add_argument('--collectionStrt', type=str, default='single', help='collection structure (single|multiple)')
    parser.add_argument('--collection', type=str,  help='dataset name')
    parser.add_argument('--trainCollection', type=str, help='train collection')
    parser.add_argument('--valCollection', type=str,  help='validation collection')
    parser.add_argument('--testCollection', type=str,  help='test collection')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1], help='overwrite existed file. (default: 0)')
    # model
    parser.add_argument('--model', type=str, default='dual_encoding', help='model name. (default: dual_encoding)')
    parser.add_argument('--space', type=str, default='hybrid', help='which concept? hybrid, latent, concept')
    parser.add_argument('--concate', type=str, default='full', help='feature concatenation style. (full|reduced) full=level 1+2+3; reduced=level 2+3')
    parser.add_argument('--measure', type=str, default='cosine', help='measure method. (default: cosine)')
    parser.add_argument('--measure_2', type=str, default='jaccard', help='measure method. (default: cosine)')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate (default: 0.2)')
    # text-side multi-level encoding
    parser.add_argument('--vocab', type=str, default='word_vocab_5', help='word vocabulary. (default: word_vocab_5)')
    parser.add_argument('--word_dim', type=int, default=500, help='word embedding dimension')
    parser.add_argument('--text_rnn_size', type=int, default=512, help='text rnn encoder size. (default: 1024)')
    parser.add_argument('--text_kernel_num', type=int, default=512, help='number of each kind of text kernel')
    parser.add_argument('--text_kernel_sizes', type=str, default='2-3-4', help='dash-separated kernel size to use for text convolution')
    parser.add_argument('--text_norm', action='store_true', help='normalize the text embeddings at last layer')
    # video-side multi-level encoding
    parser.add_argument('--visual_feature', type=str, default='resnet-152-img1k-flatten0_outputos', help='visual feature.')
    parser.add_argument('--visual_rnn_size', type=int, default=512, help='visual rnn encoder size')
    parser.add_argument('--visual_kernel_num', type=int, default=512, help='number of each kind of visual kernel')
    parser.add_argument('--visual_kernel_sizes', type=str, default='2-3-4-5', help='dash-separated kernel size to use for visual convolution')
    parser.add_argument('--visual_norm', action='store_true', help='normalize the visual embeddings at last layer')
    parser.add_argument('--gru_pool', type=str, default='mean', help='pooling on output of gru (mean|max)')
    # common space learning
    parser.add_argument('--text_mapping_layers', type=str, default='0-1536', help='text fully connected layers for common space learning. (default: 0-2048)')
    parser.add_argument('--visual_mapping_layers', type=str, default='0-1536', help='visual fully connected layers  for common space learning. (default: 0-2048)')
    # loss
    parser.add_argument('--loss_fun', type=str, default='mrl', help='loss function')
    parser.add_argument('--margin', type=float, default=0.2, help='rank loss margin')
    parser.add_argument('--margin_2', type=float, default=0.2, help='rank loss margin')
    parser.add_argument('--direction', type=str, default='all', help='retrieval direction (all|t2v|v2t)')
    parser.add_argument('--max_violation', action='store_true', help='use max instead of sum in the rank loss')
    parser.add_argument('--cost_style', type=str, default='sum', help='cost style (sum, mean). (default: sum)')
    # optimizer
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer. (default: rmsprop)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.99, help='learning rate decay rate. (default: 0.99)')
    parser.add_argument('--grad_clip', type=float, default=2, help='gradient clipping threshold')
    parser.add_argument('--resume', type=str, default='', metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--val_metric', type=str, default='recall', help='performance metric for validation (mir|recall)')
    # misc
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=128, help='Size of a training mini-batch.')
    parser.add_argument('--workers', type=int, default=5, help='Number of data loader workers.')
    parser.add_argument('--postfix', type=str, default='runs_0', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--log_step', type=int, default=10, help='Number of steps to print and record the log.')
    parser.add_argument('--cv_name', type=str, default='cv_tpami_2021', help='')
    #tag
    parser.add_argument('--tag_vocab_size', type=int, default=512, help='what the size of tag vocab will you use')

    args = parser.parse_args()
    return args


def main():
    opt = parse_args()

    rootpath = opt.rootpath
    collectionStrt = opt.collectionStrt
    collection = opt.collection

    if collectionStrt == 'single': # train,val data are in one directory
        opt.trainCollection = '%strain' % collection
        opt.valCollection = '%sval' % collection
        opt.testCollection = '%stest' % collection
        collections_pathname = {'train': collection, 'val': collection, 'test': collection}
    elif collectionStrt == 'multiple': # train,val data are separated in multiple directories
        collections_pathname = {'train': opt.trainCollection, 'val': opt.valCollection, 'test': opt.testCollection}
    else:
        raise NotImplementedError('collection structure %s not implemented' % collectionStrt)


    cap_file = {'train': '%s.caption.txt' % opt.trainCollection, 
                'val': '%s.caption.txt' % opt.valCollection}
                
    opt.collections_pathname = collections_pathname
    opt.cap_file = cap_file

    if opt.loss_fun == "mrl" and opt.measure == "cosine":
        assert opt.text_norm is True
        assert opt.visual_norm is True

    # checkpoint path
    opt.model = opt.model + "_" + opt.space
    model_info = '%s_concate_%s_dp_%.1f_measure_%s_%s' %  (opt.model, opt.concate, opt.dropout, opt.measure, opt.measure_2)
    # text-side multi-level encoding info
    text_encode_info = 'vocab_%s_word_dim_%s_text_rnn_size_%s_text_norm_%s' % \
            (opt.vocab, opt.word_dim, opt.text_rnn_size, opt.text_norm)
    text_encode_info += "_kernel_sizes_%s_num_%s" % (opt.text_kernel_sizes, opt.text_kernel_num)
    # video-side multi-level encoding info
    visual_encode_info = 'visual_feature_%s_visual_rnn_size_%d_visual_norm_%s' % \
            (opt.visual_feature, opt.visual_rnn_size, opt.visual_norm)
    visual_encode_info += "_kernel_sizes_%s_num_%s" % (opt.visual_kernel_sizes, opt.visual_kernel_num)
    # common space learning info
    mapping_info = "mapping_text_%s_img_%s_tag_vocab_size_%d" % (opt.text_mapping_layers, opt.visual_mapping_layers, opt.tag_vocab_size)

    if opt.gru_pool == 'max':
        mapping_info += '_gru_pool_%s' % opt.gru_pool

    loss_info = 'loss_func_%s_margin_%s_%s_direction_%s_max_violation_%s_cost_style_%s' % \
                        (opt.loss_fun, opt.margin, opt.margin_2, opt.direction, opt.max_violation, opt.cost_style)
    optimizer_info = 'optimizer_%s_lr_%s_decay_%.2f_grad_clip_%.1f_val_metric_%s' % \
                    (opt.optimizer, opt.learning_rate, opt.lr_decay_rate, opt.grad_clip, opt.val_metric)


    opt.logger_name = os.path.join(rootpath, collections_pathname['train'], opt.cv_name, collections_pathname['val'], model_info, text_encode_info,
                            visual_encode_info, mapping_info, loss_info, optimizer_info, opt.postfix)
    logging.info(opt.logger_name)

    if checkToSkip(os.path.join(opt.logger_name, 'model_best.pth.tar'), opt.overwrite):
        sys.exit(0)
    if checkToSkip(os.path.join(opt.logger_name, 'val_metric.txt'), opt.overwrite):
        sys.exit(0)
    makedirsforfile(os.path.join(opt.logger_name, 'val_metric.txt'))

    # logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    log_config(opt.logger_name)
    tb_logger.configure(opt.logger_name, flush_secs=5)
    logging.info(json.dumps(vars(opt), indent = 2))


    opt.text_kernel_sizes = list(map(int, opt.text_kernel_sizes.split('-')))
    opt.visual_kernel_sizes = list(map(int, opt.visual_kernel_sizes.split('-')))

    # caption
    caption_files = { x: os.path.join(rootpath, collections_pathname[x], 'TextData', cap_file[x])
                        for x in cap_file }
    # Load visual features
    visual_feat_path = {x: os.path.join(rootpath, collections_pathname[x], 'FeatureData', opt.visual_feature)
                        for x in cap_file }
    visual_feats = {x: BigFile(visual_feat_path[x]) for x in visual_feat_path}
    opt.visual_feat_dim = visual_feats['train'].ndims

    # Load tag vocabulary
    tag_vocab_size = opt.tag_vocab_size
    tag_vocab_path = os.path.join(rootpath, collections_pathname['train'], 'TextData', 'tags', 'video_label_th_1', 'tag_vocab_%d.json' % tag_vocab_size)
    tag_path = os.path.join(rootpath, collections_pathname['train'], 'TextData', 'tags', 'video_label_th_1.txt')
       
    # set bow vocabulary and encoding
    bow_vocab_file = os.path.join(rootpath, collections_pathname['train'], 'TextData', 'vocabulary', 'bow', opt.vocab+'.pkl')
    bow_vocab = pickle.load(open(bow_vocab_file, 'rb'))
    bow2vec = get_text_encoder('bow')(bow_vocab)
    opt.bow_vocab_size = len(bow_vocab)

    # set rnn vocabulary 
    rnn_vocab_file = os.path.join(rootpath, collections_pathname['train'], 'TextData', 'vocabulary', 'rnn', opt.vocab+'.pkl')
    rnn_vocab = pickle.load(open(rnn_vocab_file, 'rb'))
    opt.vocab_size = len(rnn_vocab)
        
    # initialize word embedding
    opt.we_parameter = None
    if opt.word_dim == 500:
        w2v_data_path = os.path.join(rootpath, "word2vec", 'flickr', 'vec500flickr30m')
        opt.we_parameter = get_we_parameter(rnn_vocab, w2v_data_path)

    # mapping layer structure
    opt.text_mapping_layers = list(map(int, opt.text_mapping_layers.split('-')))
    opt.visual_mapping_layers = list(map(int, opt.visual_mapping_layers.split('-')))
    if opt.concate == 'full':
        opt.text_mapping_layers[0] = opt.bow_vocab_size + opt.text_rnn_size*2 + opt.text_kernel_num * len(opt.text_kernel_sizes) 
        opt.visual_mapping_layers[0] = opt.visual_feat_dim + opt.visual_rnn_size*2 + opt.visual_kernel_num * len(opt.visual_kernel_sizes)
    elif opt.concate == 'reduced':
        opt.text_mapping_layers[0] = opt.text_rnn_size*2 + opt.text_kernel_num * len(opt.text_kernel_sizes) 
        opt.visual_mapping_layers[0] = opt.visual_rnn_size*2 + opt.visual_kernel_num * len(opt.visual_kernel_sizes)
    else:
        raise NotImplementedError('Model %s not implemented' % opt.model)


    # set data loader
    video2frames = {x: read_dict(os.path.join(rootpath, collections_pathname[x], 'FeatureData', opt.visual_feature, 'video2frames.txt'))
                    for x in cap_file }
    data_loaders = data.get_train_data_loaders(
        caption_files, visual_feats, tag_path, tag_vocab_path, rnn_vocab, bow2vec, opt.batch_size, opt.workers, video2frames=video2frames)
    val_video_ids_list = data.read_video_ids(caption_files['val'])
    val_vid_data_loader = data.get_vis_data_loader(visual_feats['val'], opt.batch_size, opt.workers, video2frames['val'], video_ids=val_video_ids_list)
    val_text_data_loader = data.get_txt_data_loader(caption_files['val'], rnn_vocab, bow2vec, opt.batch_size, opt.workers)

    # Construct the model
    model = get_model(opt.model)(opt)
    opt.we_parameter = None
    
    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            logging.info("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            logging.info("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            validate.validate(opt, tb_logger, data_loaders['val'], model, measure=opt.measure)
        else:
            logging.info("=> no checkpoint found at '{}'".format(opt.resume))


    # Train the Model
    best_rsum = 0
    no_impr_counter = 0
    lr_counter = 0
    best_epoch = None
    fout_val_metric_hist = open(os.path.join(opt.logger_name, 'val_metric_hist.txt'), 'w')
    for epoch in range(opt.num_epochs):
        logging.info('Epoch[{0} / {1}] LR: {2}'.format(epoch, opt.num_epochs, get_learning_rate(model.optimizer)[0]))
        logging.info('-'*10)
        # train for one epoch
        train(opt, data_loaders['train'], model, epoch)

        if opt.space == 'hybrid':
            rsum = validate.validate_hybrid(opt, tb_logger, val_vid_data_loader, val_text_data_loader, model, measure=opt.measure, measure_2=opt.measure_2)
        elif opt.space == 'latent':
            rsum = validate.validate(opt, tb_logger, val_vid_data_loader, val_text_data_loader, model, measure=opt.measure)
        
        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        logging.info(' * Current perf: {}'.format(rsum))
        logging.info(' * Best perf: {}'.format(best_rsum))
        logging.info('')
        fout_val_metric_hist.write('epoch_%d: %f\n' % (epoch, rsum))
        fout_val_metric_hist.flush()
        
        if is_best:
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.Eiters,
            }, is_best, filename='checkpoint_epoch_%s.pth.tar'%epoch, prefix=opt.logger_name + '/', best_epoch=best_epoch)
            best_epoch = epoch

    
        lr_counter += 1
        decay_learning_rate(opt, model.optimizer, opt.lr_decay_rate)
        if not is_best:
            # When the validation performance decreased after an epoch,
            # we divide the learning rate by 2 and continue training;
            # but we use each learning rate for at least 3 epochs.
            if lr_counter > 2:
                decay_learning_rate(opt, model.optimizer, 0.5)
                lr_counter = 0

        # Early stop occurs if the validation performance does not improve in ten consecutive epochs
        if not is_best:
            no_impr_counter += 1
        else:
            no_impr_counter = 0
        if no_impr_counter > 10:
            logging.info('Early stopping happended.\n')
            break

    fout_val_metric_hist.close()

    logging.info('best performance on validation: {}\n'.format(best_rsum))
    with open(os.path.join(opt.logger_name, 'val_metric.txt'), 'w') as fout:
        fout.write('best performance on validation: ' + str(best_rsum))

    # generate evaluation shell script
    if opt.testCollection == 'iacc.3':
        striptStr = ''.join(open( 'util/TEMPLATE_do_test_avs.sh').readlines())
        striptStr = striptStr.replace('@@@query_sets@@@', 'tv16.avs.txt,tv17.avs.txt,tv18.avs.txt')
    else:
        striptStr = ''.join(open( 'util/TEMPLATE_do_test.sh').readlines())
    striptStr = striptStr.replace('@@@rootpath@@@', rootpath)
    striptStr = striptStr.replace('@@@collectionStrt@@@', collectionStrt)
    striptStr = striptStr.replace('@@@testCollection@@@', collections_pathname['test'])
    striptStr = striptStr.replace('@@@logger_name@@@', opt.logger_name)
    striptStr = striptStr.replace('@@@overwrite@@@', str(opt.overwrite))

    # perform evaluation on test set
    runfile = 'do_test_%s_%s.sh' % (opt.model, collections_pathname['test'])
    open(runfile, 'w').write(striptStr + '\n')
    os.system('chmod +x %s' % runfile)
    os.system('./'+runfile)


def train(opt, train_loader, model, epoch):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    # switch to train mode
    model.train_start()

    progbar = Progbar(len(train_loader.dataset))
    end = time.time()
    for i, train_data in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger
        # Update the model
        b_size, loss = model.train_emb(*train_data)
        progbar.add(b_size, values=[('loss', loss)])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix='', best_epoch=None):
    """save checkpoint at specific path"""
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
    if best_epoch is not None:
        os.remove(prefix + 'checkpoint_epoch_%s.pth.tar'%best_epoch)


def decay_learning_rate(opt, optimizer, decay):
    """decay learning rate to the last LR"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*decay


def get_learning_rate(optimizer):
    """Return learning rate"""
    lr_list = []
    for param_group in optimizer.param_groups:
        lr_list.append(param_group['lr'])
    return lr_list


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
