# -*- coding: utf-8 -*-
"""MainYelp

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dFLbHKGo2FvujqwKvJuMA2ff5xpM0ixl
"""
import os
import argparse

from DatasetClasses import YelpDataset
from torch.utils.data import DataLoader
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors
import numpy as np
import torch
from TrainModel import TrainYelpModel
from DataLoader_fns import Collate
from Inference_fns import get_metrics, plot_roc

np.random.seed(0)
torch.manual_seed(0)

word_embedding_pt = dict(glove='../word_embeddings/glove_word_vectors',
                         w2v='..word_embeddings/custom_w2v_100d',
                         fasttext='../word_embeddings/fasttext_300d.bin')


def _parse_args():
    """
    Command-line arguments to the system.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='MainYelp.py')

    # General system running and configuration options
    parser.add_argument('--model', type=str, default='baseline', help='model to run')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('--epochs', type=int, default=1, help='epochs to run')
    parser.add_argument('--word_embedding', type=str, default='glove', help='word embedding to use')
    parser.add_argument('--train_path', type=str, default='dataset_train.json', help='path to train set')
    parser.add_argument('--dev_path', type=str, default='dataset_dev.json', help='path to dev set')
    parser.add_argument('--test_path', type=str, default='dataset_test.json', help='path to test set')
    parser.add_argument('--word_nh', type=int, default=1, help='number of attention heads for word attn')
    parser.add_argument('--sent_nh', type=int, default=1, help='number of attention heads for sent attn')    
    parser.add_argument('--model_size', type=int, default=64, help='model_size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--save_path', type=str, default='test', help='path to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda', help='device to train')
    parser.add_argument("--num_layers", default=1, type=int, help="num of layers of sentence level self attention")
    parser.add_argument("--word_nlayers", default=1, type=int, help="num of layers of word level self attention")
    args = parser.parse_args()
    return args


def predict_reviews(trainer, dataloader_transcripts_test):
    if args.model == 'baseline':
        print("running baseline")
        model = trainer.train_gru_model()
    elif args.model == 'gru_attention':
        print("running gru+attention")
        model = trainer.train_gru_attention()
    elif args.model == 'han':
        print("running Hierarchical Attention Network")
        model = trainer.train_HAN()
    elif args.model == 'hsan':
        print("running Hierarchical Self Attention Network")
        model = trainer.train_HSAN()
    elif args.model == 'hs2an':
        print("running Hierarchical Self-Self Attention Network")
        model = trainer.train_HS2AN()
    elif args.model == 'lstm':
        print("running LSTM Self Attention Network")
        model = trainer.train_lstm()

    ##save state dict
    torch.save(model.state_dict(), args.save_path + "yelp_{}.model".format(args.model))
    print('Test Metrics for Yelp dataset is:')
    metrics, pred_df = get_metrics(dataloader_transcripts_test, model)
    plot_roc(pred_df, args.save_path + "yelp_{}_auc".format(args.model))
    pred_df.to_pickle(args.save_path + "pred_test_{}.p".format(args.model))
    return metrics


def get_max_len(df):
    def fun(sent):
        return len(sent.split())

    max_trans_len = np.max(df.text.apply(lambda x: len(x.split("\n"))))
    max_sent_len = np.max(df.text.apply(lambda x: max(map(fun, x.split('\n')))))
    return max_trans_len, max_sent_len


def run_yelp_model():
    print("Creating Yelp Dataset")
    dataset_train = YelpDataset(args.train_path)
    dataset_dev = YelpDataset(args.dev_path)
    dataset_test = YelpDataset(args.test_path)
    max_review_len,  max_sent_len = 256, 512
    vocab = dataset_train.get_vocab()
    dataset_train.save_vocab('vocab')
    c = Collate(vocab, args.device)
    dataloader_transcripts_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                              num_workers=0, collate_fn=c.yelp_collate)
    dataloader_transcripts_dev = DataLoader(dataset_dev, batch_size=args.batch_size, shuffle=False,
                                            num_workers=0, collate_fn=c.yelp_collate)
    dataloader_transcripts_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                             num_workers=0, collate_fn=c.yelp_collate)

    print("Start loading glove vectors")
    model = KeyedVectors.load(word_embedding_pt[args.word_embedding], mmap='r')
    print("Finished loading glove vectors")
    vec_size = model.vector_size
    vocab_size = len(vocab)
    weights_matrix = np.zeros((vocab_size, vec_size))
    i = 2
    for word in vocab.get_itos()[2:]:
        try:
            weights_matrix[i] = model[word]  # model.wv[word] for trained word2vec
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(vec_size,))
        i += 1
    weights_matrix[0] = np.mean(weights_matrix, axis=0)
    weights_matrix = torch.tensor(weights_matrix)
    trainer = TrainYelpModel(dataloader_transcripts_train, dataloader_transcripts_dev, vocab_size, vec_size,
                             weights_matrix, args, max_review_len, max_sent_len)
    metrics = predict_reviews(trainer, dataloader_transcripts_test)
    return [metrics]


if __name__ == "__main__":
    args = _parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Arguments:", args)
    os.makedirs(args.save_path)
    results = run_yelp_model()
    avg_tuple = [sum(y) / len(y) for y in zip(*results)]
    print("Overall accuracy={} Overall F1 score={}".format(avg_tuple[0], avg_tuple[1]))
