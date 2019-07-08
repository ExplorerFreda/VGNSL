import nltk
import numpy as np
import os

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class PrecompDataset(data.Dataset):
    """ load precomputed captions and image features """

    def __init__(self, data_path, data_split, vocab, 
                 load_img=True, img_dim=2048):
        self.vocab = vocab

        # captions
        self.captions = list()
        with open(os.path.join(data_path, f'{data_split}_caps.txt'), 'r') as f:
            for line in f:
                self.captions.append(line.strip().lower().split())
            f.close()
        self.length = len(self.captions)

        # image features
        if load_img:
            self.images = np.load(os.path.join(data_path, f'{data_split}_ims.npy'))
        else:
            self.images = np.zeros((self.length // 5, img_dim))
        
        # each image can have 1 caption or 5 captions 
        if self.images.shape[0] != self.length:
            self.im_div = 5
            assert self.images.shape[0] * 5 == self.length
        else:
            self.im_div = 1

    def __getitem__(self, index):
        # image
        img_id = index  // self.im_div
        image = torch.tensor(self.images[img_id])
        # caption
        caption = [self.vocab(token) 
                   for token in ['<start>'] + self.captions[index] + ['<end>']]
        caption = torch.tensor(caption)
        return image, caption, index, img_id

    def __len__(self):
        return self.length


def collate_fn(data):
    """ build mini-batch tensors from a list of (image, caption) tuples """
    # sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    zipped_data = list(zip(*data))
    images, captions, ids, img_ids = zipped_data
    images = torch.stack(images, 0)
    targets = torch.zeros(len(captions), len(captions[0])).long()
    lengths = [len(cap) for cap in captions]
    for i, cap in enumerate(captions):
        end = len(cap)
        targets[i, :end] = cap[:end]
    return images, targets, lengths, ids


def get_precomp_loader(data_path, data_split, vocab, batch_size=128,
                       shuffle=True, num_workers=2, load_img=True, 
                       img_dim=2048):
    dset = PrecompDataset(data_path, data_split, vocab, load_img, img_dim)
    data_loader = torch.utils.data.DataLoader(
        dataset=dset, batch_size=batch_size, shuffle=shuffle,
        pin_memory=True, 
        collate_fn=collate_fn
    )
    return data_loader


def get_train_loaders(data_path, vocab, batch_size, workers):
    train_loader = get_precomp_loader(
        data_path, 'train', vocab, batch_size, True, workers
    )
    val_loader = get_precomp_loader(
        data_path, 'dev', vocab, batch_size, False, workers
    )
    return train_loader, val_loader


def get_eval_loader(data_path, split_name, vocab, batch_size, workers, 
                    load_img=False, img_dim=2048):
    eval_loader = get_precomp_loader(
        data_path, split_name, vocab, batch_size, False, workers, 
        load_img=load_img, img_dim=img_dim
    )
    return eval_loader
