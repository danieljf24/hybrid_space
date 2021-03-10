'''
join several sub collections' features into one large collection feature and concatenate multiple features into one feature
'''

import os
import sys
import logging
from basic.constant import ROOT_PATH
from basic.generic_utils import Progbar
from basic.bigfile import BigFile
from txt2bin import process as txt2bin

logger = logging.getLogger(__file__)
logging.basicConfig(
        format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
        datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

def process(options, collection, featname_1, featname_2, sub_collections):
    rootpath = options.rootpath
    target_feat_dir = os.path.join(rootpath, collection, 'FeatureData', featname_1+"_"+featname_2)

    if os.path.exists(target_feat_dir):
        if options.overwrite:
            logger.info('%s exists! overwrite.', target_feat_dir)
        else:
            logger.info('%s exists! quit.', target_feat_dir)
            sys.exit(0)
    else:
        os.makedirs(target_feat_dir)

    target_feat_file = os.path.join(target_feat_dir, 'id.feature.txt')
    sub_collections = sub_collections.split('@')

    with open(target_feat_file, 'w') as fw_feat:
        for collect in sub_collections:
            feat_dir_1 = os.path.join(rootpath, collect, 'FeatureData', featname_1)
            feat_dir_2 = os.path.join(rootpath, collect, 'FeatureData', featname_2)
            featfile_1 = BigFile(feat_dir_1)
            featfile_2 = BigFile(feat_dir_2)

            print(">>> Process %s" % collect)
            progbar = Progbar(len(featfile_1.names))
            for name in featfile_1.names:
                feat_1 = featfile_1.read_one(name)
                feat_2 = featfile_2.read_one(name)
                fw_feat.write('%s %s\n' % (name, ' '.join(['%g'%x for x in feat_1+feat_2])))
                progbar.add(1)

    # transform txt to bin format
    txt2bin(len(feat_1)+len(feat_2), target_feat_file, target_feat_dir, options.overwrite)

    ln_target_feat_dir = os.path.join(rootpath, collection, 'FeatureData', 'resnext101-resnet152')
    os.system('ln -s %s %s' % (target_feat_dir, ln_target_feat_dir))
    

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection featname_1 featname_2""")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--sub_collections", default="", type="str", help="sub collections")

    (options, args) = parser.parse_args(argv)
    if len(args) < 4:
        parser.print_help()
        return 1

    return process(options, args[0], args[1], args[2], args[3])

if __name__ == '__main__':
    sys.exit(main())

