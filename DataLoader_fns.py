
import torch
from torchtext.data.utils import get_tokenizer
from collections import defaultdict
word_tokenizer = get_tokenizer('basic_english')


class Collate:
    def __init__(self, vocab, device):
        self.vocab = vocab
        self.device = device
        self.count = 0

    def pad_trans(self, trans, max_len):
        num_sents = len(trans)
        sent_pos_indices = [i + 1 for i in range(num_sents+1)]
        for i in range(max_len - num_sents):
            trans.append('<pad>')
            sent_pos_indices.append(0)
        return trans, sent_pos_indices

    def get_indices(self, sentence, max_sent_len):
        tokens = word_tokenizer(sentence)
        tokens.insert(0, "<cls>")
        indices = [self.vocab[token] for token in tokens]
        diff = max_sent_len - len(tokens) + 1
        positional_indices = [i + 1 if t !='<pad>' else 0 for i, t in enumerate(tokens)]
        for i in range(diff):
            indices.append(self.vocab['<pad>'])  # padding idx=1
            positional_indices.append(0)
        return indices, positional_indices

    def yelp_collate(self, batch):
        max_num_sents = 0
        max_sent_len = 0
        trans_len = []
        for i, sample in enumerate(batch):
            sent_len = []
            trans = sample['text']
            for sent in trans:
                l = len(word_tokenizer(sent))
                sent_len.append(l)
                if l > max_sent_len:
                    max_sent_len = l
            num_sents = len(trans)
            if num_sents > max_num_sents:
                max_num_sents = num_sents
            trans_len.append(sent_len)

        for sample in batch:
            trans = sample['text']
            pad_trans, sample['sent_pos_indices'] = self.pad_trans(trans, max_num_sents)
            sample['indices'] = []
            sample['word_pos_indices'] = []
            for sent in pad_trans:
                indices, positional_indices = self.get_indices(sent, max_sent_len)
                sample['indices'].append(indices)
                sample['word_pos_indices'].append(positional_indices)

        batch_dict = defaultdict(list)
        for sample in batch:
            for key in sample:
                batch_dict[key].append(sample[key])

        batch_dict['indices'] = torch.tensor(batch_dict['indices'], device=self.device)
        batch_dict['sent_pos_indices'] = torch.tensor(batch_dict['sent_pos_indices'], device=self.device)
        batch_dict['word_pos_indices'] = torch.tensor(batch_dict['word_pos_indices'], device=self.device)
        batch_dict['lens'] = trans_len
        batch_dict['category'] = torch.tensor(batch_dict['category'], device=self.device)

        return batch_dict
