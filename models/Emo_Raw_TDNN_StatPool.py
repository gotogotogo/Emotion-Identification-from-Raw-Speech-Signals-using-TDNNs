#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:59:45 2020

@author: krishna

"""

import torch.nn as nn
from models.tdnn import TDNN
import torch
import torch.nn.functional as F

class CNN_frontend(nn.Module):
    def __init__(self):
        super(CNN_frontend, self).__init__()
        # self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=7, stride=4, padding=3,
        #                        bias=False)
        self.conv1 = nn.Conv1d(1, 256, kernel_size=400, stride=160, padding=199, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.nin1 = NIN(256, 128)

        self.nin2 = NIN(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
    def forward(self, raw_input):
        # N x 1 x 160000
        out = self.conv1(raw_input)
        # N x 256 x 1000
        out = F.relu(torch.log(torch.abs(out)))
        out = self.bn1(out)
        out = self.nin1(out)
        # N x 128 x 500
        out = self.nin2(out)
        out = self.bn2(out)
        # N x 128 x 250
        return out

class NIN(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(NIN, self).__init__()
        self.fc1 = nn.Linear(inchannel * 2, inchannel * 4, bias=False)
        self.fc2 = nn.Linear(inchannel * 4, outchannel, bias=False)
    def forward(self, x):
        N, channels, time_steps = x.shape[0], x.shape[1], x.shape[2]
        x = x.permute(0, 2, 1).contiguous()
        x = x.reshape(N, time_steps // 2, channels * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.permute(0, 2, 1).contiguous()
        return x

class Emo_Raw_TDNN(nn.Module):
    def __init__(self, num_classes=4):
        super(Emo_Raw_TDNN, self).__init__()
        self.cnn_frontend = CNN_frontend()
        
        self.tdnn1 = TDNN(input_dim=128, output_dim=128, context_size=3, dilation=1,dropout_p=0.5)
        
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=64,num_layers=1,bidirectional=True,dropout=0.5,batch_first=True)     
        
        self.tdnn2 = TDNN(input_dim=128, output_dim=128, context_size=7, dilation=3,dropout_p=0.5)
        
        self.tdnn3 = TDNN(input_dim=128, output_dim=128, context_size=7, dilation=3,dropout_p=0.5)
        
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64,num_layers=1,bidirectional=True,dropout=0.5,batch_first=True)     
        
        self.tdnn4 = TDNN(input_dim=128, output_dim=128, context_size=7, dilation=3,dropout_p=0.5)
        
        self.tdnn5 = TDNN(input_dim=128, output_dim=128, context_size=7, dilation=3,dropout_p=0.5)
        
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=64,num_layers=1,bidirectional=True,dropout=0.5,batch_first=True)     
        self.fc = nn.Linear(2*128,num_classes)
        
        
        
    def forward(self, inputs):
        cnn_out = self.cnn_frontend(inputs)
        #print('frontend output shape:', cnn_out.shape) 
        print('frontend output', cnn_out)
        cnn_out = cnn_out.permute(0,2,1)
        tdnn1_out = self.tdnn1(cnn_out)
        lstm1_out, (final_hidden_state, final_cell_state) = self.lstm1(tdnn1_out)
        tdnn2_out = self.tdnn2(lstm1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        lstm2_out, (final_hidden_state, final_cell_state) = self.lstm2(tdnn3_out)
        
        tdnn4_out = self.tdnn4(lstm2_out)
        tdnn5_out = self.tdnn5(tdnn4_out)
        lstm3_out, (final_hidden_state, final_cell_state) = self.lstm3(tdnn5_out)
        ### Stat Pool
        mean = torch.mean(lstm3_out,1)
        std = torch.var(lstm3_out,1)
        stat_pooling = torch.cat((mean,std),1)
        #print('stat pooling shape:', stat_pooling.shape)
        emo_predictions= self.fc(stat_pooling)
        return emo_predictions
    
    
    
