# -*- coding: utf-8 -*-
"""TrainModel

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JkW85J7ZhjzUcKk7sSf8mSfFK6svNtND
"""

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from Models import *
from Inference_fns import get_metrics
from sklearn.utils import class_weight

class TrainYelpModel():
    def __init__(self, dataloader_train, dataloader_dev, vocab_size, vec_size, weights_matrix,
                 args, max_review_len, max_sent_len):
        self.dataloader_train = dataloader_train
        self.dataloader_dev = dataloader_dev
        self.vocab_size = vocab_size
        self.vec_size = vec_size
        self.weights_matrix = weights_matrix
        self.max_sent_len = max_sent_len
        self.max_review_len = max_review_len
        self.args = args

    def get_model(self):

        if self.args.model == 'baseline':
            print("running baseline")
            encoder = EncoderRNN(self.vocab_size, self.vec_size, self.args.model_size, self.weights_matrix,
                                 self.args.dropout)
        elif self.args.model == 'gru_attention':
            print("running gru+attention")
            encoder = GRUAttention(self.vocab_size, self.vec_size, self.args.model_size, self.weights_matrix,
                                   self.args.dropout)
        elif self.args.model == 'han':
            print("running Hierarchical Attention Network")
            encoder = HAN(self.vocab_size, self.vec_size, self.args.model_size, self.weights_matrix, self.args.dropout)
        elif self.args.model == 'hsan':
            print("running Hierarchical Self Attention Network")
            encoder = HSAN(self.vocab_size, self.vec_size, self.args.model_size, self.weights_matrix,
                           self.max_review_len, self.max_sent_len, self.args.sent_nh, self.args.dropout,
                           self.args.num_layers)
        elif self.args.model == 'hs2an':
            print("running Hierarchical Self-Self Attention Network")
            encoder = HS2AN(self.vocab_size, self.vec_size, self.args.model_size, self.weights_matrix,
                            self.max_review_len, self.max_sent_len, self.args.word_nh, self.args.sent_nh,
                            self.args.dropout, self.args.num_layers,
                            self.args.word_nlayers)
        elif self.args.model == 'lstm':
            print("running LSTM Self Attention Network")
            encoder = LSTMAttention(self.vocab_size, self.vec_size, self.args.model_size, self.weights_matrix,
                                    self.args.dropout)
        elif self.args.model == 'wtsan':
            print("running Word Transformer Self Attn model")
            encoder = WordTransformerAttention(self.vocab_size, self.vec_size, self.args.model_size, self.weights_matrix, self.max_sent_len,
                                               self.args.word_nh, self.args.dropout, self.args.word_nlayers)
        elif self.args.model == 'stsan':
            print("running Sent Transformer Self Attn Model")
            encoder = SentTransformerAttention(self.vocab_size, self.vec_size, self.args.model_size, self.args.sent_nh, self.max_review_len,
                                               self.args.dropout, self.args.num_layers)
        fcn = FCN(self.args.model_size, self.args.dropout)
        model = EncoderFCN(encoder, fcn)
        return model

    def train(self):
        model = self.get_model()
        encoder_optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        print(model)
        criterion = nn.CrossEntropyLoss()
        self.train_model(self.args.epochs, model, criterion, encoder_optimizer)
        return model

    def train_model(self, epochs, encoder, criterion, encoder_optimizer):
        train_acc = []
        dev_acc = []
        loss_arr = []
        encoder = encoder.to(self.args.device)
        encoder.train()
        for n in range(epochs):
            epoch_loss = 0
            for batch in tqdm(self.dataloader_train):
                loss = 0
                output, scores = encoder(batch)
                target = batch['category']
                loss += criterion(output, target)
                encoder_optimizer.zero_grad()
                epoch_loss += loss.detach().item()
                loss.backward()
                encoder_optimizer.step()
            avg_epoch_loss = epoch_loss / len(self.dataloader_train)
            print("Average loss at epoch {}: {}".format(n, avg_epoch_loss))
            loss_arr.append(avg_epoch_loss)
            if n % 5 == 4:
                print("Training metric at end of epoch {}:".format(n))
                train_metrics, _ = get_metrics(self.dataloader_train, encoder)
                print("Dev metric at end of epoch {}:".format(n))
                dev_metrics, _ = get_metrics(self.dataloader_dev, encoder)
                train_acc.append(train_metrics)
                dev_acc.append(dev_metrics)
        plt.plot(loss_arr)
        plt.show()
        plt.savefig(self.args.save_path+'loss.png')
        print("Training Evaluation Metrics: ", train_acc)
        print("Dev Evaluation Metrics: ", dev_acc)
