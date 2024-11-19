#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import json
import random
import time
import datetime
import random
import re
import numpy as np
import emoji
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from sklearn.metrics import *

# If there's a GPU available...
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name())
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# In[2]:
def fix_the_random(seed_val = 42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


# In[3]:
fix_the_random(2021)


# In[4]:
def evalMetric(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    mf1Score = f1_score(y_true, y_pred, average='macro')
    f1Score  = f1_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    area_under_c = auc(fpr, tpr)
    recallScore = recall_score(y_true, y_pred)
    precisionScore = precision_score(y_true, y_pred)
    return {"accuracy": accuracy, 'mF1Score': mf1Score, 'f1Score': f1Score, 'auc': area_under_c, 
            'precision': precisionScore, 'recall': recallScore}


def getFeaturesandLabel(X, y, text_features):
    X_text_data = []
    for i in X:
        X_text_data.append(text_features[i])
    X_text_data = torch.tensor(X_text_data)
    y_data = torch.tensor(y)
    return X_text_data, y_data


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# In[5]:
import torch.nn as nn
import torch.nn.functional as F

class Uni_Model(nn.Module):
    def __init__(self, input_size, fc1_hidden, fc2_hidden, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, fc1_hidden),
            nn.ReLU(),
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Linear(fc2_hidden, output_size),
        )
    
    def forward(self, xb):
        return self.network(xb)


# In[6]:
import torch
import torch.nn as nn


# In[7]:
import numpy as np
def getProb(temp):
    t = np.exp(temp)
    return t[1] / (sum(t))


# In[8]:
import pandas as pd
def getPerformanceOfLoader(model, test_dataloader, loadType):
    model.eval()
    # Tracking variables 
    predictions, true_labels = [], []
    # Predict 
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
  
        # Unpack the inputs from our dataloader
        b_text_ids, b_labels = batch
  
        with torch.no_grad():
            outputs = model(b_text_ids)
        
        logits = outputs.max(1, keepdim=True)[1]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend(logits)
        true_labels.extend(label_ids)

    print('DONE.')

    pred = [i[0] for i in predictions]
    df = pd.DataFrame()
    if loadType == 'val':
        df['Ids'] = val_list
    else:
        df['Ids'] = test_list
    df['true'] = true_labels
    df['target'] = pred
    return df


# In[9]:
def trainModel(model, train_dataloader, validation_dataloader, test_dataloader):
    model.cuda()

    bestValAcc = 0
    bestValMF1 = 0
    besttest_df = None
    bestEpochs = -1
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 30
    total_steps = len(train_dataloader) * epochs
    loss_values = []

    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_text_ids = batch[0].to(device)
            b_labels = batch[1].to(device)
            model.zero_grad()        

            outputs = model(b_text_ids)
            y_preds = torch.max(outputs, 1)[1]

            loss = F.cross_entropy(outputs, b_labels, weight=torch.FloatTensor([0.374, 0.626]).to(device))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

        print("")
        print("Running Validation...")
        t0 = time.time()

        val_df = getPerformanceOfLoader(model, validation_dataloader, "val")
        origValValue, preValValue = list(val_df['true']), list(val_df['target'])
        valMf1Score = evalMetric(origValValue, preValValue)['mF1Score']
        tempValAcc = evalMetric(origValValue, preValValue)['accuracy']
        if (valMf1Score > bestValMF1):
            bestEpochs = epoch_i
            bestValMF1 = valMf1Score
            bestValAcc = tempValAcc
            besttest_df = getPerformanceOfLoader(model, test_dataloader, "test")

        print("  Accuracy: {0:.2f}".format(tempValAcc))
        print("  Macro F1: {0:.2f}".format(valMf1Score))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

    print("")
    print(bestEpochs)
    print("Training complete!")
    return besttest_df


# In[10]:
import pickle
with open('./FoldWiseDetailBengaliAbusiveMeme.p', 'rb') as fp:
    allDataAnnotation = pickle.load(fp)


# In[11]:
FOLDER_NAME = "./"

modelNameMapping = {
    "mBERTEmbedding": FOLDER_NAME + 'AllFeatures/mEmbedding_bn_memes.p',
    "MuRILEmbedding": FOLDER_NAME + 'AllFeatures/MuRILEmbedding_bn_memes.p',
    "PhoBERTEmb": FOLDER_NAME + 'AllFeatures/PhoBERTEmbedding.p',
    "XLMREmb": FOLDER_NAME + 'AllFeatures/xlmBERTEmbedding_bn_memes.p',
    "vgg16": FOLDER_NAME + 'AllFeatures/vgg16.p',
    "resNet152_new": FOLDER_NAME + 'AllFeatures/resNet152_newFeatures_224.p',
    "vit_new_wOReize": FOLDER_NAME + 'AllFeatures/vit_newFeatures_wOResize.p',
    'van': FOLDER_NAME + "AllFeatures/van_newFeatures.p",
}


# In[12]:
metricType = ['accuracy', 'mF1Score', 'f1Score', 'auc', 'precision', 'recall']


# In[13]:
k = 2
epochs = 30
batch_size = 32
learning_rate = 1e-4
log_interval = 1
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

allF = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']

import pickle

outputFp = open("unimodalFoldWiseLessParam.txt", "w")

for foldName in allF:

    with open(modelNameMapping["mBERTEmbedding"], 'rb') as fp:
        text_features = pickle.load(fp)

    # X and y for training
    X = allDataAnnotation[foldName]['X']
    y = allDataAnnotation[foldName]['y']
    val_list = allDataAnnotation[foldName]['val']
    test_list = allDataAnnotation[foldName]['test']

    # Chia dữ liệu thành 3 phần: train, val, test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tiếp tục chia X_train thành X_train và X_val cho tập xác nhận
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # Chuyển đổi thành dữ liệu PyTorch Tensor
    X_train_data, y_train_data = getFeaturesandLabel(X_train, y_train, text_features)
    X_val_data, y_val_data = getFeaturesandLabel(X_val, y_val, text_features)
    X_test_data, y_test_data = getFeaturesandLabel(X_test, y_test, text_features)

    # Tạo DataLoader cho các tập dữ liệu
    train_data = TensorDataset(X_train_data, y_train_data)
    val_data = TensorDataset(X_val_data, y_val_data)
    test_data = TensorDataset(X_test_data, y_test_data)

    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
    val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=batch_size)
    test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=batch_size)

    # Tạo model và huấn luyện
    model = Uni_Model(input_size=768, fc1_hidden=256, fc2_hidden=128, output_size=2)
    test_df = trainModel(model, train_dataloader, val_dataloader, test_dataloader)
    
    outputFp.write(str(test_df))
