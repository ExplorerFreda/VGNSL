import os
import pickle

import numpy
from data import get_eval_loader
import time
import numpy as np
from vocab import Vocabulary 
import torch
from model import VGNSL
from collections import OrderedDict

from utils import generate_tree, clean_tree
from IPython import embed

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print, vocab=None, stage='dev'):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()
    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    logged = False
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger
        lengths = torch.Tensor(lengths).long()
        if torch.cuda.is_available():
            lengths = lengths.cuda()

        # compute the embeddings
        model_output = model.forward_emb(images, captions, lengths, volatile=True)
        img_emb, cap_span_features, left_span_features, right_span_features, word_embs, tree_indices, all_probs, \
        span_bounds = model_output[:8]

        # output sampled trees
        if (not logged) or (stage == 'test'):
            logged = True
            if stage == 'dev':
                sample_num = 5
            for j in range(sample_num):
                logging(generate_tree(captions, tree_indices, j, vocab))

        cap_emb = torch.cat([cap_span_features[l-2][i].reshape(1, -1) for i, l in enumerate(lengths)], dim=0)

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        del images, captions

    return img_embs, cap_embs


def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
        # print(npts)
    index_list = []

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        d = numpy.dot(im, captions.T).flatten()
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
        # print(npts)
    ims = numpy.array([images[i] for i in range(0, len(images), 5)])

    ranks = numpy.zeros(5 * npts)
    top1 = numpy.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]

        # compute scores
        d = numpy.dot(queries, ims.T)
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[5 * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def test_trees(model_path):
    """ use the trained model to generate parse trees for text """
    # load model and options
    checkpoint = torch.load(model_path, map_location='cpu')
    opt = checkpoint['opt']

    # load vocabulary used by the model
    vocab = pickle.load(open(os.path.join(opt.data_path, 'vocab.pkl'), 'rb'))
    opt.vocab_size = len(vocab)

    # construct model
    model = VGNSL(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_eval_loader(
        opt.data_path, 'test', vocab, opt.batch_size, opt.workers, 
        load_img=False, img_dim=opt.img_dim
    )

    cap_embs = None
    logged = False
    trees = list()
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = print
        lengths = torch.Tensor(lengths).long()
        if torch.cuda.is_available():
            lengths = lengths.cuda()

        # compute the embeddings
        model_output = model.forward_emb(images, captions, lengths, volatile=True)
        img_emb, cap_span_features, left_span_features, right_span_features, word_embs, tree_indices, all_probs, \
        span_bounds = model_output[:8]

        candidate_trees = list()
        for j in range(len(ids)):
            candidate_trees.append(generate_tree(captions, tree_indices, j, vocab))
        appended_trees = ['' for _ in range(len(ids))]
        for j in range(len(ids)):
            appended_trees[ids[j] - min(ids)] = clean_tree(candidate_trees[j])
        trees.extend(appended_trees)
        cap_emb = torch.cat([cap_span_features[l-2][i].reshape(1, -1) for i, l in enumerate(lengths)], dim=0)
        del images, captions, img_emb, cap_emb

    ground_truth = [line.strip() for line in open(
        os.path.join(opt.data_path, 'test_ground-truth.txt'))]
    return trees, ground_truth
