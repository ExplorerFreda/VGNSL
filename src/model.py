import numpy as np
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.nn import functional
from torch.nn import functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import make_embeddings, l2norm, cosine_sim, sequence_mask, \
    index_mask, index_one_hot_ellipsis
import utils


class EncoderImagePrecomp(nn.Module):
    """ image encoder """
    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm

        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """ Xavier initialization for the fully connected layer """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """ extract image feature vectors """
        # assuming that the precomputed features are already l2-normalized
        features = self.fc(images.float())

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)
        
        return features

    def load_state_dict(self, state_dict):
        """ copies parameters, overwritting the default one to
            accept state_dict from Full model """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


class EncoderText(nn.Module):
    """ text encoder """
    def __init__(self, opt, vocab_size, semantics_dim):
        super(EncoderText, self).__init__()
        opt.syntax_dim = semantics_dim  # syntax is tied with semantics

        self.vocab_size = vocab_size
        self.semantics_dim = semantics_dim

        self.sem_embedding = nn.Linear(semantics_dim, semantics_dim, bias=False)
        self.syn_score = nn.Sequential(
            nn.Linear(opt.syntax_dim * 2, opt.syntax_score_hidden),
            nn.ReLU(),
            nn.Linear(opt.syntax_score_hidden, 1, bias=False)
        )
        # self.reset_weights()

    def reset_weights(self):
        self.sem_embedding.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths, volatile=False, audio_masks=None):  
        """ sample a tree for each sentence """
        max_select_cnt = int(lengths.max(dim=0)[0].item()) - 1

        tree_indices = list()
        tree_probs = list()
        span_bounds = list()
        features = list()
        left_span_features = list()
        right_span_features = list()

        # closed range: [left_bounds[i], right_bounds[i]]
        left_bounds = utils.add_dim(torch.arange(
            0, max_select_cnt + 1, dtype=torch.long, device=x.device), 0, x.size(0))
        right_bounds = left_bounds

        assert audio_masks is not None
        if torch.cuda.is_available():
            audio_masks = audio_masks.cuda()
        sem_embeddings = (self.sem_embedding(x) - 1e10 * (1 - audio_masks)).max(-2)[0]
        syn_embeddings = sem_embeddings

        output_word_embeddings = sem_embeddings * \
            sequence_mask(lengths, max_length=lengths.max()).unsqueeze(-1).float()
        
        valid_bs = lengths.size(0)
        for i in range(max_select_cnt):
            seq_length = sem_embeddings.size(1)
            # set invalid positions to 0 prob
            # [0, 0, ..., 1, 1, ...]
            length_mask = 1 - sequence_mask(
                (lengths - 1 - i).clamp(min=0), max_length=seq_length - 1).float()
            # 0 = done
            undone_mask = 1 - length_mask[:, 0]

            syn_feats = torch.cat(
                (l2norm(syn_embeddings[:, 1:]), l2norm(syn_embeddings[:, :-1])), 
                dim=2
            )
            prob_logits = self.syn_score(syn_feats).squeeze(-1)

            prob_logits = prob_logits - 1e10 * length_mask
            probs = F.softmax(prob_logits, dim=1)

            if not volatile:
                sampler = Categorical(probs)
                indices = sampler.sample()
            else:
                indices = probs.max(1)[1]
            tree_indices.append(indices)
            tree_probs.append(index_one_hot_ellipsis(probs, 1, indices))

            this_spans = torch.stack([
                index_one_hot_ellipsis(left_bounds, 1, indices),
                index_one_hot_ellipsis(right_bounds, 1, indices + 1)
            ], dim=1)
            this_features = torch.add(
                index_one_hot_ellipsis(sem_embeddings, 1, indices),
                index_one_hot_ellipsis(sem_embeddings, 1, indices + 1)
            )
            this_left_features = index_one_hot_ellipsis(sem_embeddings, 1, indices)
            this_right_features = index_one_hot_ellipsis(sem_embeddings, 1, indices + 1)
            this_features = l2norm(this_features)
            this_left_features = l2norm(this_left_features)
            this_right_features = l2norm(this_right_features)

            span_bounds.append(this_spans)
            features.append(l2norm(this_features) * undone_mask.unsqueeze(-1).float())
            left_span_features.append(this_left_features * undone_mask.unsqueeze(-1).float())
            right_span_features.append(this_right_features * undone_mask.unsqueeze(-1).float())

            # update word embeddings
            left_mask = sequence_mask(indices, max_length=seq_length).float()
            right_mask = 1 - sequence_mask(indices + 2, max_length=seq_length).float()
            center_mask = index_mask(indices, max_length=seq_length).float()
            update_masks = (left_mask, right_mask, center_mask)

            this_features_syn = torch.add(
                index_one_hot_ellipsis(syn_embeddings, 1, indices),
                index_one_hot_ellipsis(syn_embeddings, 1, indices + 1)
            )
            this_features_syn = l2norm(this_features_syn)
            syn_embeddings = self.update_with_mask(syn_embeddings, syn_embeddings, this_features_syn, *update_masks)

            sem_embeddings = self.update_with_mask(sem_embeddings, sem_embeddings, this_features, *update_masks)
            left_bounds = self.update_with_mask(left_bounds, left_bounds, this_spans[:, 0], *update_masks)
            right_bounds = self.update_with_mask(right_bounds, right_bounds, this_spans[:, 1], *update_masks)

        return features, left_span_features, right_span_features, output_word_embeddings, tree_indices, \
               tree_probs, span_bounds

    @staticmethod
    def update_with_mask(lv, rv, cv, lm, rm, cm):
        if lv.dim() > lm.dim():
            lm = lm.unsqueeze(2)
            rm = rm.unsqueeze(2)
            cm = cm.unsqueeze(2)

        return (lv * lm.to(lv))[:, :-1] + (rv * rm.to(rv))[:, 1:] + (cv.unsqueeze(1) * cm.to(cv))[:, :-1]


class ContrastiveReward(nn.Module):
    """ compute contrastive reward """

    def __init__(self, margin=0):
        super(ContrastiveReward, self).__init__()
        self.margin = margin
        self.sim = cosine_sim

    def forward(self, im, s):  
        """ return the reward """
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval, given images and retrieve captions
        reward_s = (d1 - scores - self.margin).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval, given caption and retrieve images
        reward_im = (d2 - scores - self.margin).clamp(min=0)
        # clear diagonals
        I = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = I.cuda()
        reward_s = reward_s.masked_fill_(I, 0)
        reward_im = reward_im.masked_fill_(I, 0)

        # sum up the reward
        reward_s = reward_s.mean(1)
        reward_im = reward_im.mean(0)

        return reward_s + reward_im


class ContrastiveLoss(nn.Module):
    """ compute contrastive loss for VSE """

    def __init__(self, margin=0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sim = cosine_sim

    def forward(self, im, s):
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        loss_s = (self.margin + scores - d1).clamp(min=0)
        loss_im = (self.margin + scores - d2).clamp(min=0)
        I = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = I.cuda()
        loss_s = loss_s.masked_fill_(I, 0)
        loss_im = loss_im.masked_fill_(I, 0)

        loss_s = loss_s.mean(1)
        loss_im = loss_im.mean(0)

        return loss_s + loss_im


class VGNSL(object):
    """ the main VGNSL model """

    def __init__(self, opt):
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImagePrecomp(
            opt.img_dim, opt.embed_size, opt.no_imgnorm
        )
        self.txt_enc = EncoderText(opt, opt.vocab_size, opt.mfcc_dim)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # loss, reward and optimizer
        self.reward_criterion = ContrastiveReward(margin=opt.margin)
        self.loss_criterion = ContrastiveLoss(margin=opt.margin)
        self.vse_reward_alpha = opt.vse_reward_alpha
        self.vse_loss_alpha = opt.vse_loss_alpha
        self.lambda_hi = opt.lambda_hi

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())
        self.params = params

        self.optimizer = getattr(torch.optim, opt.optimizer)(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(), self.optimizer.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        if len(state_dict) >= 3:
            self.optimizer.load_state_dict(state_dict[2])

    def train_start(self):
        """ switch to train mode """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """ switch to evaluate mode """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, volatile=False, audio_masks=None):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
        with torch.set_grad_enabled(not volatile):
            img_emb = self.img_enc(images)
            txt_outputs= self.txt_enc(captions, lengths, volatile, audio_masks=audio_masks)
        return (img_emb, ) + txt_outputs

    def forward_reward(self, base_img_emb, cap_span_features, left_span_features, right_span_features,
                       word_embs, lengths, span_bounds, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        reward_matrix = torch.zeros(base_img_emb.size(0), lengths.max(0)[0]-1).float()
        left_reg_matrix = torch.zeros(base_img_emb.size(0), lengths.max(0)[0]-1).float()
        right_reg_matrix = torch.zeros(base_img_emb.size(0), lengths.max(0)[0]-1).float()
        if torch.cuda.is_available():
            reward_matrix = reward_matrix.cuda()
            right_reg_matrix = right_reg_matrix.cuda()
            left_reg_matrix = left_reg_matrix.cuda()

        matching_loss = 0
        for i in range(lengths.max(0)[0] - 1):
            curr_imgs = list()
            curr_caps = list()
            curr_left_caps = list()
            curr_right_caps = list()
            indices = list()
            for j in range(base_img_emb.size(0)):
                if i < lengths[j] - 1:
                    curr_imgs.append(base_img_emb[j].reshape(1, -1))
                    curr_caps.append(cap_span_features[lengths[j] - 2 - i][j].reshape(1, -1))
                    curr_left_caps.append(left_span_features[lengths[j] - 2 - i][j].reshape(1, -1))
                    curr_right_caps.append(right_span_features[lengths[j] - 2 - i][j].reshape(1, -1))
                    indices.append(j)

            img_emb = torch.cat(curr_imgs, dim=0)
            cap_emb = torch.cat(curr_caps, dim=0)
            left_cap_emb = torch.cat(curr_left_caps, dim=0)
            right_cap_emb = torch.cat(curr_right_caps, dim=0)
            reward = self.reward_criterion(img_emb, cap_emb)
            left_reg = self.loss_criterion(img_emb, left_cap_emb)
            right_reg = self.loss_criterion(img_emb, right_cap_emb)
            for idx, j in enumerate(indices):
                reward_matrix[j][lengths[j] - 2 - i] = reward[idx]
                left_reg_matrix[j][lengths[j] - 2 - i] = left_reg[idx]
                right_reg_matrix[j][lengths[j] - 2 - i] = right_reg[idx]

            this_matching_loss = self.loss_criterion(img_emb, cap_emb)
            matching_loss += this_matching_loss.sum() + left_reg.sum() + right_reg.sum()
        reward_matrix = reward_matrix / (self.lambda_hi * right_reg_matrix + 1.0)
        reward_matrix = self.vse_reward_alpha * reward_matrix

        return reward_matrix, matching_loss

    def train_emb(self, images, captions, audios, audio_masks, lengths, ids=None, epoch=None, *args):
        """ one training step given images and captions """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
        lengths = torch.Tensor(lengths).long()
        if torch.cuda.is_available():
            lengths = lengths.cuda()

        # compute the embeddings
        img_emb, cap_span_features, left_span_features, right_span_features, word_embs, tree_indices, probs, \
            span_bounds = self.forward_emb(images, audios, lengths, audio_masks=audio_masks)

        # measure accuracy and record loss
        cum_reward, matching_loss = self.forward_reward(
            img_emb, cap_span_features, left_span_features, right_span_features, word_embs, lengths,
            span_bounds
        )
        probs = torch.cat(probs, dim=0).reshape(-1, lengths.size(0)).transpose(0, 1)
        masks = sequence_mask(lengths - 1, lengths.max(0)[0] - 1).float()
        rl_loss = torch.sum(-masks * torch.log(probs) * cum_reward.detach())
        
        loss = rl_loss + matching_loss * self.vse_loss_alpha
        loss = loss / cum_reward.shape[0]
        self.logger.update('Loss', float(loss), img_emb.size(0))
        self.logger.update('MatchLoss', float(matching_loss / cum_reward.shape[0]), img_emb.size(0))
        self.logger.update('RL-Loss', float(rl_loss / cum_reward.shape[0]), img_emb.size(0))
        
        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

        # clean up
        if epoch > 0:
            del cum_reward
            del tree_indices
            del probs
            del cap_span_features
            del span_bounds
