# -*- coding: utf-8 -*-
"""DataLoader_fns

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1t2vWLuHjJ9f2XaKZ75UHjFK3AWDiJv2M
"""

import torch
from torchtext.data.utils import get_tokenizer
word_tokenizer = get_tokenizer('basic_english')

class Collate:
    def __init__(self, vocab, device):
        self.vocab = vocab
        self.device = device
        self.count = 0

    def pad_review(self, review, max_len):
        num_sents = len(review)
        review_pos_indices = [i+1 for i in range(num_sents)]
        for i in range(max_len - num_sents):
            review.append('<pad>')
            review_pos_indices.append(0)
        return review, review_pos_indices

    def get_indices(self, sentence, max_sent_len):
        tokens = word_tokenizer(sentence)
        indices = [self.vocab[token] for token in tokens]
        diff = max_sent_len - len(tokens)
        positional_indices = [i + 1 for i in range(len(tokens))]
        for i in range(diff):
            indices.append(self.vocab['<pad>'])  # padding idx=1
            positional_indices.append(0)
        return indices, positional_indices

    def yelp_collate(self, batch):
        max_num_sents = 0
        max_sent_len = 0
        review_len = []
        for i, sample in enumerate(batch):
            sent_len = []
            review = sample['text']
            for sent in review:
                l = len(word_tokenizer(sent))
                sent_len.append(l)
                if l > max_sent_len:
                    max_sent_len = l
            num_sents = len(review)
            if num_sents > max_num_sents:
                max_num_sents = num_sents
            review_len.append(sent_len)

        for sample in batch:
            review = sample['text']
            pad_review, sample['review_pos_indices'] = self.pad_review(review, max_num_sents)
            sample['indices'] = []
            sample['word_pos_indices'] = []
            for sent in pad_review:
                indices, positional_indices = self.get_indices(sent, max_sent_len)
                sample['indices'].append(indices)
                sample['word_pos_indices'].append(positional_indices)

        batch_dict = {'text': [], 'indices': [], 'category': [], 'review_pos_indices': [], 'word_pos_indices': []}
        for sample in batch:
            batch_dict['text'].append(sample['text'])
            batch_dict['indices'].append(sample['indices'])
            batch_dict['category'].append(sample['category'])
            batch_dict['review_pos_indices'].append(sample['review_pos_indices'])
            batch_dict['word_pos_indices'].append(sample['word_pos_indices'])

        batch_dict['indices'] = torch.tensor(batch_dict['indices'], device=self.device)
        batch_dict['review_pos_indices'] = torch.tensor(batch_dict['review_pos_indices'], device=self.device)
        batch_dict['word_pos_indices'] = torch.tensor(batch_dict['word_pos_indices'], device=self.device)
        batch_dict['category'] = torch.tensor(batch_dict['category'], device=self.device)
        batch_dict['lens'] = review_len

        return batch_dict
