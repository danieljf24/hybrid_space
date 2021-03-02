import os
import re
import sys
import nltk
import json
import argparse
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from collections import Counter

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    return string.strip().lower()

ENGLISH_STOP_WORDS = set(stopwords.words('english'))
'''
get the labels of each caption on train&val&test  actually only use train labels in train in test/val will use
the labels that creat by retriveed captions
'''
def checkToSkip(filename, overwrite):
    if os.path.exists(filename):
        print ("%s exists." % filename),
        if overwrite:
            print ("overwrite")
            return 0
        else:
            print ("skip")
            return 1
    return 0

def makedirsforfile(filename):
    try:
        os.makedirs(os.path.split(filename)[0])
    except:
        pass

def get_wordnet_pos(tag):
    '''
    This function will get each word speech
    :param tag: have tagged speech of words
    :return:the part of speech of word
    '''
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    else:
        return None


def fromtext(opt, txt):
    vid2words = {}
    sid2words = {}
    with open(txt, 'r') as t:
        for line in tqdm(t.readlines(), desc='Reading and processing captions'):
            sid, cap = line.strip().split(" ", 1)
            cap = clean_str(cap)
            vid, num = sid.strip().split("#", 1)
            num = num.strip().split("#")[-1]
            sid2words[sid] = []
            if int(num) == 0:
                vid2words[vid] = []
            if opt.use_lemma:
                tokens = nltk.word_tokenize(cap)
                tagged_sent = pos_tag(tokens)
                wnl = WordNetLemmatizer()
                for tag in tagged_sent:
                    wordnet_pos = get_wordnet_pos(tag[1])
                    # if tag[1] == 'CD':
                    #     wordnet_pos = 'n'
                    if wordnet_pos is None:
                        continue
                    try:
                        w = wnl.lemmatize(tag[0], pos=wordnet_pos)
                    except UnicodeDecodeError:
                        print ('%s encoding error' % tag[0])
                        continue
                    if w in ENGLISH_STOP_WORDS:
                        continue
                    vid2words[vid].append(w)
                    sid2words[sid].append(w)
            else:
                vid2words[vid].extend(cap.strip().split(" "))
                sid2words[sid].extend(cap.strip().split(" "))

    return vid2words, sid2words  # {vid:[word,word,word....],....}

def get_tags(id2words, threshold, output_file, str_name):
    fout = open(output_file, 'w')
    for vid in tqdm(id2words.keys(), desc='Processing {0} labels'.format(str_name)):
        word2counter = {}
        fout.write('%s\t' % vid)

        if vid.startswith('tgif'):
            new_threshold = 0
        else:
            new_threshold = threshold

        for w in id2words[vid]:
            word2counter[w] = word2counter.get(w, 0) + 1  # count word frequency
        word2counter = sorted(word2counter.items(), key=lambda a: a[1],
                              reverse=True)  # return a list like [(word:frequency)]

        for (word, value) in word2counter:
            if value > new_threshold:
                fout.write('%s:%d ' % (word, value))  # (word , frequency)
        fout.write('\n')
    fout.close()

def main(opt):
    rootpath = opt.rootpath
    collection = opt.collection
    th4vl = opt.th4vl
    th4sl = opt.th4sl
    use_lemma = opt.use_lemma
    tag_vocab_size = opt.tag_vocab_size

    output_video_labels = os.path.join(rootpath, collection, "TextData", 'tags', 'video_label_th_%d.txt' % (th4vl))
    # output_sent_labels = os.path.join(rootpath, collection, "TextData", 'tags',  'sentence_label_th_%d.txt' % (th4sl))

    if checkToSkip(output_video_labels, opt.overwrite):
        sys.exit(0)
    makedirsforfile(output_video_labels)

    tag_vocab_dir = os.path.join(rootpath, collection, "TextData", 'tags', 'video_label_th_%d' % (th4vl))
    all_words_file = os.path.join(tag_vocab_dir, 'all_words.txt')
    if checkToSkip(all_words_file, opt.overwrite):
        sys.exit(0)
    makedirsforfile(all_words_file)

    cap_file = os.path.join(rootpath, collection, 'TextData', '%s.caption.txt'%collection)
    if not os.path.exists(cap_file):
        cap_file = os.path.join(rootpath, collection, 'TextData', '%strain.caption.txt'%collection)
    vid2words, sid2words = fromtext(opt, cap_file)
    
    if collection.startswith('MPII-MD') or collection.startswith('TGIF'):
        th4vl = 0
        th4sl = 0
    get_tags(vid2words, th4vl, output_video_labels, 'video')
    # get_tags(sid2words, th4sl, output_sent_labels, 'sentence')

    print ('The video labels have saved to %s' % (output_video_labels))
    # print ('The sentence labels have saved to %s' % (output_sent_labels))

    
    # generate tag vocabulary
    lines = map(str.strip, open(output_video_labels))
    cnt = Counter()

    for line in lines:
        elems = line.split()
        del elems[0]
        # assert(len(elems)>0)
        for x in elems:
            tag,c = x.split(':')
            cnt[tag] += int(c)


    print(len(cnt))
    taglist = cnt.most_common()
    fw = open(all_words_file, 'w')
    fw.write('\n'.join(['%s %d' % (x[0], x[1]) for x in taglist]))
    fw.close()

    top_tag_list = [x[0] for x in taglist]

    # save tag vocabulary
    output_json_file = os.path.join(tag_vocab_dir, 'tag_vocab_%d.json' % tag_vocab_size)
    output_txt_file = os.path.join(tag_vocab_dir, 'tag_vocab_%d.txt' % tag_vocab_size)

    with open(output_json_file, 'w') as jsonFile:
        jsonFile.write(json.dumps(top_tag_list[:tag_vocab_size]))

    with open(output_txt_file, 'w') as txtFile:
        txtFile.write('\n'.join(top_tag_list[:tag_vocab_size]))

    # open(output_file, 'w').write('\n'.join(output_vocab))
    print('Save words to %s and %s' % (output_json_file, output_txt_file))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--rootpath', default='/home/jd/VisualSearch/test', help='rootpath of the data')
    parser.add_argument('--collection', default="msrvtt10k", type=str, help='collection')
    parser.add_argument('--th4vl', type=int, default=1, help='minimum word count threshold for video labels')
    parser.add_argument('--th4sl', type=int, default=0, help='minimum word count threshold for sentence labels')
    parser.add_argument('--overwrite', default=1, type=int, help='overwrite existing file (default=0)')
    parser.add_argument('--use_lemma', action="store_true", help='whether use lemmatization')
    parser.add_argument('--tag_vocab_size', default=512, type=int, help='vocabulary size of concepts tags')
    args = parser.parse_args()
    # opt = opts.parse_opt()
    main(args)