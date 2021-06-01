#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 20:22:26 2020

@author: krishna
"""

import torch
import numpy as np
from torch.utils.data import DataLoader   
from dataset import CustomDataset
import torch.nn as nn
import os
from torch import optim
import argparse
from models.Emo_Raw_TDNN_StatPool import Emo_Raw_TDNN
from sklearn.metrics import accuracy_score
from utils.utils_wav import speech_collate
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

########## Argument parser
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--wav_files_collect_path', type=str, default='wav_collect_files.pkl')

parser.add_argument('-input_dim', action="store_true", default=1)
parser.add_argument('-num_classes', action="store_true", default=4)
parser.add_argument('-lamda_val', action="store_true", default=0.1)
parser.add_argument('-batch_size', action="store_true", default=64)
parser.add_argument('-use_gpu', action="store_true", default=True)
parser.add_argument('-num_epochs', action="store_true", default=1000)
args = parser.parse_args()

### Data related
dataset_train = CustomDataset(args.wav_files_collect_path, mode='train', test_sess=5)
#dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=speech_collate) 
dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)  
dataset_test = CustomDataset(args.wav_files_collect_path, mode='test', test_sess=5)
#dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, collate_fn=speech_collate) 
dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)  
## Model related
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Emo_Raw_TDNN(args.num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)
loss_fun = nn.CrossEntropyLoss()



def train(train_loader,epoch):
    train_loss_list=[]
    full_preds=[]
    full_gts=[]
    model.train()
    train_loader = tqdm(train_loader)
    for step, sample_batched in enumerate(train_loader):
        
        features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[0]])).float()
        print(sample_batched[0].shape)
        print(features.shpae)
        labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
        print(sample_batched[1].shape)
        print(labels.shape)
        features, labels = features.to(device),labels.to(device)
        features.requires_grad = True
        optimizer.zero_grad()
        pred_logits = model(features)
        #### CE loss
        loss = loss_fun(pred_logits,labels)
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.item())
        #train_acc_list.append(accuracy)
        if step % 10 == 0:
            train_loader.desc = "[epoch {} step {}] mean loss {}".format(epoch, step, np.mean(np.asarray(train_loss_list)))
        
        predictions = np.argmax(pred_logits.detach().cpu().numpy(),axis=1)
        for pred in predictions:
            full_preds.append(pred)
        for lab in labels.detach().cpu().numpy():
            full_gts.append(lab)
            
    mean_acc = accuracy_score(full_gts,full_preds)
    mean_loss = np.mean(np.asarray(train_loss_list))
    print('[epoch {}] train loss {} train accuracy {}'.format(epoch, mean_loss, mean_acc))
    


def test(test_loader,epoch, best_acc):
    model.eval()
    with torch.no_grad():
        val_loss_list=[]
        full_preds=[]
        full_gts=[]
        for i_batch, sample_batched in enumerate(test_loader):
            features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[0]])).float()
            labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
            features, labels = features.to(device),labels.to(device)
            pred_logits = model(features)
            #### CE loss
            loss = loss_fun(pred_logits,labels)
            val_loss_list.append(loss.item())
            #train_acc_list.append(accuracy)
            predictions = np.argmax(pred_logits.detach().cpu().numpy(),axis=1)
            for pred in predictions:
                full_preds.append(pred)
            for lab in labels.detach().cpu().numpy():
                full_gts.append(lab)
                
        mean_acc = accuracy_score(full_gts,full_preds)
        mean_loss = np.mean(np.asarray(val_loss_list))
        print('[epoch {}] test loss {} test accuracy {}\n'.format(epoch, mean_loss, mean_acc))
        if mean_acc > best_acc:
            best_acc = mean_acc
            model_save_path = os.path.join('save_model', 'best_'+str(epoch)+'_'+str(round(mean_acc, 5)))
            state_dict = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}
            torch.save(state_dict, model_save_path)
            print('-' * 20, 'best model in epoch {} '.format(epoch), '-' * 20)
    return best_acc
    
if __name__ == '__main__':
    if not os.path.isdir('save_model'):
        os.makedirs('save_model')
    best_acc = 0.0
    for epoch in range(args.num_epochs):
        train(dataloader_train,epoch)
        best_acc = test(dataloader_test,epoch, best_acc)
        
    
    
