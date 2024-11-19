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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from sklearn.metrics import *


# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name())

# If not...
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



def getFeaturesandLabel(X,y, text_features, image_features):
    X_text_data = []
    X_image_data = []
    for i in X:
        try:
            X_text_data.append(text_features[i]['text'])
        except:
            X_text_data.append(text_features[i])
        try:
            X_image_data.append(image_features[i]['image'])
        except:
            X_image_data.append(image_features[i])
    X_text_data = torch.tensor(X_text_data).float()
    X_image_data = torch.tensor(X_image_data).float()
    y_data = torch.tensor(y)
    return X_text_data, X_image_data, y_data

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
        self.network=nn.Sequential(
            nn.Linear(input_size,fc1_hidden),
            nn.ReLU(),
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Linear(fc2_hidden, output_size),
        )
    def forward(self, xb):
        return self.network(xb)


    
class Combined_model(nn.Module):
    def __init__(self, text_model, image_model, num_classes):
        super().__init__()
        self.text_model = text_model
        self.image_model = image_model
        self.fc_output   = nn.Linear(2*64, num_classes)
    def forward(self, x_text, x_vid):
        tex_out = self.text_model(x_text)
        vid_out = self.image_model(x_vid)
        inp = torch.cat((tex_out, vid_out), dim = 1)
        out = self.fc_output(inp)
        return out


# In[6]:


import torch
import torch.nn as nn


# In[7]:


import numpy as np
def getProb(temp):
    t = np.exp(temp)
    return t[1]/(sum(t))


# In[8]:


import pandas as pd
def getPerformanceOfLoader(model,test_dataloader, loadType):
    model.eval()
    # Tracking variables 
    predictions , true_labels = [], []
    # Predict 
    for batch in test_dataloader:
        #print(batch)
    # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
  
        # Unpack the inputs from our dataloader
        b_text_ids, b_image_ids, b_labels = batch
  
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_text_ids, b_image_ids)
        
        logits = outputs.max(1, keepdim=True)[1]
        #print(logits)
        #print(logits.shape)
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # Store predictions and true labels
        predictions.extend(logits)
        true_labels.extend(label_ids)

    print('DONE.')

    pred = [i[0] for i in predictions]
    df = pd.DataFrame()
    if loadType =='val':
        df['Ids']=val_list
    else:
        df['Ids']=test_list
    df['true']=true_labels
    df['target'] = pred
    #df['score'] = proba
    return df


# In[9]:


# Tell pytorch to run this model on the GPU.
def trainModel(model, train_dataloader, validation_dataloader, test_dataloader, model_name):
    model.cuda()

    bestValAcc  = 0
    bestValMF1  = 0
    besttest_df  = None
    bestEpochs = -1
    # Get all of the model's parameters as a list of tuples.
    learning_rate = 1e-4
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)   # optimize all cnn parameters

    epochs = 30
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler.

    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_text_ids = batch[0].to(device)
            b_image_ids = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()        

            outputs = model(b_text_ids, b_image_ids)
            y_preds = torch.max(outputs, 1)[1]  # y_pred != output

            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple.
            loss = F.cross_entropy(outputs, b_labels, weight=torch.FloatTensor([0.374, 0.626]).to(device))

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()


        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader) 

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()



        # Tracking variables 
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        val_df = getPerformanceOfLoader(model,validation_dataloader, "val")
        origValValue, preValValue = list(val_df['true']), list(val_df['target'])
        # Report the final accuracy for this validation run.
        valMf1Score = evalMetric(origValValue, preValValue)['mF1Score']
        tempValAcc  = evalMetric(origValValue, preValValue)['accuracy']
        if (valMf1Score > bestValMF1):
            bestEpochs = epoch_i
            bestValMF1 = valMf1Score
            bestValAcc  = tempValAcc
            besttest_df = getPerformanceOfLoader(model,test_dataloader, "test")
            torch.save(model, "./SavedModel/"+model_name)
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


FOLDER_NAME="./"


# In[12]:


modelNameMapping = {
    "mBERTEmbedding": FOLDER_NAME+'AllFeatures/mEmbedding_bn_memes.p',
    "MuRILEmbedding" : FOLDER_NAME+'AllFeatures/MuRILEmbedding_bn_memes.p',
    "banglaBERTEmb" : FOLDER_NAME+'AllFeatures/banglaBERTEmbedding.p', 
    "XLMREmb" : FOLDER_NAME+'AllFeatures/xlmBERTEmbedding_bn_memes.p',
    
    "vgg16" : FOLDER_NAME+'AllFeatures/vgg16.p',
    "resNet152_new" : FOLDER_NAME+'AllFeatures/resNet152_newFeatures_224.p',
    "vit_new_wOReize" : FOLDER_NAME+'AllFeatures/vit_newFeatures_wOResize.p',
    'CLIP': FOLDER_NAME+'AllFeatures/CLIP_newFeatures.p',
}


# In[13]:


metricType = ['accuracy', 'mF1Score', 'f1Score', 'auc', 'precision', 'recall']


# In[14]:


# training parameters
k = 2            # number of target category
epochs = 30
batch_size = 32
learning_rate = 1e-4
log_interval = 1
import numpy as np

# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

allF = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']

import pickle

outputFp = open("MMFoldWiseLateFusion_LessParam.txt", 'a')


image_models =["resNet152_new", "vit_new_wOReize"]
text_models =["MuRILEmbedding", "XLMREmb"]


# image_models =["CLIP"]
# text_models =["CLIP"]



image_models =["CLIP"]
text_models =["CLIP"]

for text_mod in text_models:
    for image_mod in image_models:
        modelName = "lt_"+text_mod+"_"+image_mod

        with open(modelNameMapping[text_mod],'rb') as fp:
            inputTextFeatures = pickle.load(fp)

        with open(modelNameMapping[image_mod],'rb') as fp:
            inputImageFeatures = pickle.load(fp)

        # Audio parameters
        if text_mod == "CLIP":
            input_size_text, input_size_image = 512, 512
        else:
            input_size_text = len(inputTextFeatures['image_759.jpg'])
            input_size_image = len(inputImageFeatures['image_759.jpg'])


        fc1_hidden_audio, fc2_hidden_audio = 256, 256

        finalOutputAccrossFold ={}

        for fold in allF:
            # train, test split
            train_list, train_label= allDataAnnotation[fold]['train']
            val_list, val_label  =  allDataAnnotation[fold]['val']
            test_list, test_label  =  allDataAnnotation[fold]['test']

            X_train_text, X_train_image, y_train = getFeaturesandLabel(train_list, train_label, inputTextFeatures, inputImageFeatures)
            X_val_text, X_val_image, y_val = getFeaturesandLabel(val_list, val_label, inputTextFeatures, inputImageFeatures)
            X_test_text, X_test_image, y_test = getFeaturesandLabel(test_list, test_label, inputTextFeatures, inputImageFeatures)


            BATCH_SIZE = 32
            #Dataset wrapping tensors.
            train_data = TensorDataset(X_train_text, X_train_image, y_train)
            val_data = TensorDataset(X_val_text, X_val_image, y_val)
            test_data = TensorDataset(X_test_text, X_test_image, y_test)

            #Samples elements randomly. If without replacement(default), then sample from a shuffled dataset.
            train_sampler = RandomSampler(train_data)
            val_sampler = SequentialSampler(val_data)
            test_sampler = SequentialSampler(test_data)

            #represents a Python iterable over a dataset
            train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = BATCH_SIZE)
            validation_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size = BATCH_SIZE)
            test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size = BATCH_SIZE)


            tex = Uni_Model(input_size_text, fc1_hidden_audio, fc2_hidden_audio, 64).to(device)
            img = Uni_Model(input_size_image, fc1_hidden_audio, fc2_hidden_audio,  64).to(device)
            comb = Combined_model(tex, img, 2).to(device)

            besttest_df = trainModel(comb, train_dataloader, validation_dataloader, test_dataloader, text_mod+"_"+fold)


            finalOutputAccrossFold[fold] = besttest_df

        with open("ResultFolder/new_"+modelName+"_foldWise_res_LessParam.p", 'wb') as fp:
            pickle.dump(finalOutputAccrossFold,fp)

        allValueDict ={}
        for fold in allF:
            evalObject = evalMetric(finalOutputAccrossFold[fold]['true'], finalOutputAccrossFold[fold]['target'])
            for metType in metricType:
                try:
                    allValueDict[metType].append(evalObject[metType])
                except:
                    allValueDict[metType]=[evalObject[metType]]

        print(modelName)
        for i in allValueDict:
            print(f"{i} : Mean {np.mean(allValueDict[i])}  STD: {np.std(allValueDict[i])}")

        outputFp.write(modelName)
        for i in allValueDict:
            outputFp.write(f"{i} : Mean {np.mean(allValueDict[i])}  STD: {np.std(allValueDict[i])}")
            outputFp.write("\n")
        outputFp.write("====================\n")

        print("==============================")

outputFp.close()

