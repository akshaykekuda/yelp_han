# -*- coding: utf-8 -*-
"""Inference_fns

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gtEyI0DrfZVDxgsIGqETSfGBeIPuvqU4
"""
import pandas as pd
from tqdm import tqdm
import torch
from torchtext.data.utils import get_tokenizer
from DataLoader_fns import get_indices
from sklearn.metrics import classification_report, f1_score, mean_squared_error

def get_accuracy(dataloader, model):
  total_correct = 0
  with torch.no_grad():

    for batch in tqdm(dataloader):

      batch_size = len(batch['indices'])

      output, att_scores = model(batch['indices'])

      for i in range(batch_size):

        classification = torch.argmax(output[i]).item()
        target = batch['category'][i]
        if target == classification:
          total_correct+=1

  acc = total_correct/(len(dataloader) * batch_size)
  print("Accuracy: {}".format(acc))
  print(classification_report())
  return acc