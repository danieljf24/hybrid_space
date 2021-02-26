import numpy as np
from basic.metric import getScorer

# recall@k, Med r, Mean r for Text-to-Video Retrieval (need to know the number of captions for each videos)
def t2v(c2i, vis_details=False, n_caption=5):
    """
    Text->Videos (Text-to-Video Retrieval)
    c2i: (5N, N) matrix of caption to video errors
    vis_details: if true, return a dictionary for ROC visualization purposes
    """
    # print("errors matrix shape: ", c2i.shape)
    assert c2i.shape[0] // c2i.shape[1] == n_caption, c2i.shape
    ranks = np.zeros(c2i.shape[0])

    for i in range(len(ranks)):
        d_i = c2i[i]
        inds = np.argsort(d_i)

        rank = np.where(inds == i//n_caption)[0][0]
        ranks[i] = rank

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return map(float, [r1, r5, r10, medr, meanr])



# recall@k, Med r, Mean r for Video-to-Text Retrieval
def v2t(c2i, n_caption=5):
    """
    Videos->Text (Video-to-Text Retrieval)
    c2i: (5N, N) matrix of caption to video errors
    """
    #remove duplicate videos
    # print("errors matrix shape: ", c2i.shape)
    assert c2i.shape[0] // c2i.shape[1] == n_caption, c2i.shape
    ranks = np.zeros(c2i.shape[1])

    for i in range(len(ranks)):
        d_i = c2i[:, i]
        inds = np.argsort(d_i)

        rank = np.where(inds//n_caption == i)[0][0]
        ranks[i] = rank

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return map(float, [r1, r5, r10, medr, meanr])


# mAP for Text-to-Video Retrieval
def t2v_map(c2i, t2v_gts):
    """
    Text->Videos (Text-to-Video Retrieval)
    c2i: (5N, N) matrix of caption to video errors
    """
    scorer = getScorer('AP')
    perf_list = []
    for i in range(c2i.shape[0]):
        d_i = c2i[i, :]
        labels = [0]*len(d_i)

        x = t2v_gts[i][0]
        labels[x] = 1

        sorted_labels = [labels[x] for x in np.argsort(d_i)]

        current_score = scorer.score(sorted_labels)
        perf_list.append(current_score)
    return np.mean(perf_list)


# mAP for Video-to-Text Retrieval
def v2t_map(c2i, v2t_gts):
    """
    Videos->Text (Video-to-Text Retrieval)
    c2i: (5N, N) matrix of caption to video errors
    """
    scorer = getScorer('AP')
    perf_list = []
    for i in range(c2i.shape[1]):
        d_i = c2i[:, i]

        labels = [0]*len(d_i)
        # labels[i*n_caption:(i+1)*n_caption] = [1]*n_caption
        for x in v2t_gts[i]:
            labels[x] = 1
        sorted_labels = [labels[x] for x in np.argsort(d_i)]

        current_score = scorer.score(sorted_labels)
        perf_list.append(current_score)

    return np.mean(perf_list)



def get_gt(video_ids, caption_ids):
    v2t_gt = []
    for vid_id in video_ids:
        v2t_gt.append([])
        for i, cap_id in enumerate(caption_ids):
            if cap_id.split('#', 1)[0] == vid_id:
                v2t_gt[-1].append(i)

    t2v_gt = {}
    for i, t_gts in enumerate(v2t_gt):
        for t_gt in t_gts:
            t2v_gt.setdefault(t_gt, [])
            t2v_gt[t_gt].append(i)

    return v2t_gt, t2v_gt



def eval_q2m(scores, q2m_gts):
    '''
    Image -> Text / Text -> Image
    Args:
      scores: (n_query, n_memory) matrix of similarity scores
      q2m_gts: list, each item is the positive memory ids of the query id
    Returns:
      scores: (recall@1, 5, 10, median rank, mean rank)
      gt_ranks: the best ranking of ground-truth memories
    '''
    n_q, n_m = scores.shape
    gt_ranks = np.zeros((n_q,), np.int32)
    aps = np.zeros(n_q)
    for i in range(n_q):
        s = scores[i]
        sorted_idxs = np.argsort(s)
        rank = n_m + 1
        tmp_set = []
        for k in q2m_gts[i]:
            tmp = np.where(sorted_idxs == k)[0][0] + 1
            if tmp < rank:
                rank = tmp

        gt_ranks[i] = rank

    # compute metrics
    r1 = 100.0 * len(np.where(gt_ranks <= 1)[0]) / n_q
    r5 = 100.0 * len(np.where(gt_ranks <= 5)[0]) / n_q
    r10 = 100.0 * len(np.where(gt_ranks <= 10)[0]) / n_q
    medr = np.median(gt_ranks)
    meanr = gt_ranks.mean()
    # mAP = aps.mean()

    return (r1, r5, r10, medr, meanr)



def t2v_inv_rank(c2i, n_caption=1):
    """
    Text->Videos (Text-to-Video Retrieval)
    c2i: (5N, N) matrix of caption to video errors
    n_caption: number of captions of each image/video
    """
    assert c2i.shape[0] // c2i.shape[1] == n_caption, c2i.shape
    inv_ranks = np.zeros(c2i.shape[0])

    for i in range(len(inv_ranks)):
        d_i = c2i[i,:]
        inds = np.argsort(d_i)

        rank = np.where(inds == i//n_caption)[0]
        inv_ranks[i] = sum(1.0 / (rank +1 ))

    return np.mean(inv_ranks)



def v2t_inv_rank(c2i, n_caption=1):
    """
    Videos->Text (Video-to-Text Retrieval)
    c2i: (5N, N) matrix of caption to video errors
    n_caption: number of captions of each image/video
    """
    assert c2i.shape[0] // c2i.shape[1] == n_caption, c2i.shape
    inv_ranks = np.zeros(c2i.shape[1])

    for i in range(len(inv_ranks)):
        d_i = c2i[:, i]
        inds = np.argsort(d_i)

        rank = np.where(inds//n_caption == i)[0]
        inv_ranks[i] = sum(1.0 / (rank +1 ))

    return np.mean(inv_ranks)




def v2t_inv_rank_multi(c2i, n_caption=2):
    """
    Text->videos (Image Search)
    c2i: (5N, N) matrix of caption to image errors
    n_caption: number of captions of each image/video
    """
    # print("errors matrix shape: ", c2i.shape)
    assert c2i.shape[0] // c2i.shape[1] == n_caption, c2i.shape
    inv_ranks = np.zeros(c2i.shape[1])

    result = []
    for i in range(n_caption):
        idx = range(i, c2i.shape[0], n_caption)
        sub_c2i = c2i[idx, :]
        score = v2t_inv_rank(sub_c2i, n_caption=1)
        result.append(score)
    return result