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
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
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

class Cross_Entropy_Loss_Label_Smooth(nn.Module):
    def __init__(self, num_classes=4, epsilon=0.1):
        super(Cross_Entropy_Loss_Label_Smooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
    
    def forward(self, outputs, targets):
        N = targets.size(0)
        smoothed_labels = torch.full(size = (N, self.num_classes), fill_value = self.epsilon / (self.num_classes - 1)).cuda()
        targets = targets.data
        smoothed_labels.scatter_(dim = 1, index = torch.unsqueeze(targets, dim = 1), value = 1 - self.epsilon)
        log_prob = nn.functional.log_softmax(outputs, dim = 1)
        loss = - torch.sum(log_prob * smoothed_labels) / N
        return loss 

### Data related
dataset_train = CustomDataset(args.wav_files_collect_path, mode='train', test_sess=5)
dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=speech_collate)  

dataset_test = CustomDataset(args.wav_files_collect_path, mode='test', test_sess=5)
dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=speech_collate)  

## Model related
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('device: ', device)
model = Emo_Raw_TDNN(args.num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.000001, betas=(0.9, 0.98), eps=1e-9)
#loss_fun = nn.CrossEntropyLoss()
loss_fun = Cross_Entropy_Loss_Label_Smooth()



def train(train_loader,epoch):
    train_loss_list=[]
    full_preds=[]
    full_gts=[]
    model.train()
    train_loader = tqdm(train_loader)
    for step, (features, labels, _) in enumerate(train_loader):
        #print(features.shape)
        features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in features])).float()         
        labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in labels]))
        #print(labels.shape)
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
            
    unweighted_avg_recall = balanced_accuracy_score(full_gts,full_preds)
    mean_loss = np.mean(np.asarray(train_loss_list))
    print('[epoch {}] train loss {} train unweighted average recall {}'.format(epoch, mean_loss, unweighted_avg_recall))
    


def test(test_loader,epoch, best_acc, target_names):
    model.eval()
    with torch.no_grad():
        val_loss_list=[]
        full_preds=[]
        full_gts=[]
        wrong_durations = []
        for i_batch, sample_batched in enumerate(test_loader):
            features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[0]])).float()
            labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
            durations = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[2]]))
            features, labels = features.to(device),labels.to(device)
            durations = durations.to(device)
            pred_logits = model(features)
            #### CE loss
            loss = loss_fun(pred_logits,labels)
            val_loss_list.append(loss.item())
            #train_acc_list.append(accuracy)
            predictions = np.argmax(pred_logits.detach().cpu().numpy(),axis=1)
            for i in range(len(labels)):
                if(labels[i] != predictions[i]):
                    wrong_durations.append(durations[i])
            for pred in predictions:
                full_preds.append(pred)
            for lab in labels.detach().cpu().numpy():
                full_gts.append(lab)
                
        unweighted_avg_recall = balanced_accuracy_score(full_gts,full_preds)
        mean_loss = np.mean(np.asarray(val_loss_list))
        print('[epoch {}] test loss {} test unweighted average recall {}\n'.format(epoch, mean_loss, unweighted_avg_recall))
        if unweighted_avg_recall > best_acc:
            print('-' * 20, 'best model in epoch {} '.format(epoch), '-' * 20) 
            best_acc = unweighted_avg_recall
            print(classification_report(full_gts, full_preds, target_names=target_names))
            print(confusion_matrix(full_gts, full_preds))
            model_save_path = os.path.join('save_model', 'best_'+str(epoch)+'_'+str(round(unweighted_avg_recall, 5)))
            state_dict = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}
            torch.save(state_dict, model_save_path)
        for i in range(len(wrong_durations)):
            print(i, wrong_durations[i])
    return best_acc
    
if __name__ == '__main__':
    if not os.path.isdir('save_model'):
        os.makedirs('save_model')
    best_acc = 0.0
    target_names = ['happy', 'angry', 'sad', 'neutral'] 
    for epoch in range(args.num_epochs):
        train(dataloader_train,epoch)
        best_acc = test(dataloader_test,epoch, best_acc, target_names)
        
    
    
