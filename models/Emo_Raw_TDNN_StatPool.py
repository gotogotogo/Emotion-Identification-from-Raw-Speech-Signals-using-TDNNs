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
        self.nin1 = NIN(1, 64, kernel_size=7, stride=4, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool1 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.nin2 = NIN(64, 128, kernel_size=7, stride=4, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.maxpool2 = nn.MaxPool1d(kernel_size=4, stride=2)

        self.nin3 = nn.Conv1d(128, 128, kernel_size=7, stride=4)
        self.bn3 = nn.BatchNorm1d(128)
        self.maxpool3 = nn.MaxPool1d(kernel_size=4, stride=2)


    def forward(self, x):
        # N x 1 x 16000*8
        out = self.nin1(x)
        #print('1-1 ', out.shape)
        out = F.relu(self.bn1(out))
        out = self.maxpool1(out)
        #print('1-2 ', out.shape) 

        out = self.nin2(out)
        #print('2-1 ', out.shape)
        out = F.relu(self.bn2(out))
        out = self.maxpool2(out)
        #print('2-2 ', out.shape)

        out = self.nin3(out)
        #print('3-1 ', out.shape)
        out = F.relu(self.bn3(out))
        out = self.maxpool3(out)
        #print('3-2 ', out.shape)
        return out

class NIN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(NIN, self).__init__()
        self.blk = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(), 
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.blk(x)

# class CustomAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, context_size, stride=1, dilation=1, padding=0):
#         super(CustomAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.context_size = context_size
#         self.dilation = dilation
#         self.padding = padding
    
#     def forward(self, x):
#         # N x seq x channels




class Emo_Raw_TDNN(nn.Module):
    def __init__(self, num_classes=4):
        super(Emo_Raw_TDNN, self).__init__()
        self.cnn_frontend = CNN_frontend()
        
        self.tdnn1 = TDNN(input_dim=128, output_dim=128, context_size=3, dilation=1,dropout_p=0.5)
        self.Q1 = torch.nn.Parameter(torch.randn(64, 128, 128))
        self.atten1 = nn.MultiheadAttention(embed_dim=128, num_heads=8)

        self.lstm1 = nn.LSTM(input_size=128, hidden_size=64,num_layers=1,bidirectional=True,dropout=0.5,batch_first=True)     
        
        self.tdnn2 = TDNN(input_dim=128, output_dim=128, context_size=7, dilation=3,dropout_p=0.5)
        
        self.tdnn3 = TDNN(input_dim=128, output_dim=128, context_size=7, dilation=3,dropout_p=0.5)
        self.Q2 = torch.nn.Parameter(torch.randn(64, 64, 128))
        self.atten2 = nn.MultiheadAttention(embed_dim=128, num_heads=8)

        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64,num_layers=1,bidirectional=True,dropout=0.5,batch_first=True)     
        
        self.tdnn4 = TDNN(input_dim=128, output_dim=128, context_size=7, dilation=3,dropout_p=0.5)
        
        self.tdnn5 = TDNN(input_dim=128, output_dim=128, context_size=7, dilation=3,dropout_p=0.5)
        self.Q3 = torch.nn.Parameter(torch.randn(64, 32, 128))
        self.atten3 = nn.MultiheadAttention(embed_dim=128, num_heads=8)

        self.lstm3 = nn.LSTM(input_size=128, hidden_size=64,num_layers=1,bidirectional=True,dropout=0.5,batch_first=True)     
        self.multihead_attn = nn.MultiheadAttention(embed_dim=128, num_heads=8)
        self.fc = nn.Linear(2*128,num_classes)
        
        
    def forward(self, inputs):
        cnn_out = self.cnn_frontend(inputs)
        #print('frontend output shape:', cnn_out.shape) 
        # 64 x 128 x 248
        # cnn_out = cnn_out.permute(0,2,1)
        # tdnn1_out = self.tdnn1(cnn_out)
        # lstm1_out, (final_hidden_state, final_cell_state) = self.lstm1(tdnn1_out)

        # tdnn2_out = self.tdnn2(lstm1_out)
        # tdnn3_out = self.tdnn3(tdnn2_out)
        # lstm2_out, (final_hidden_state, final_cell_state) = self.lstm2(tdnn3_out)

        # tdnn4_out = self.tdnn4(lstm2_out)
        # tdnn5_out = self.tdnn5(tdnn4_out)
        # lstm3_out, (final_hidden_state, final_cell_state) = self.lstm3(tdnn5_out)

        # lstm3_out = lstm3_out.permute(1, 0, 2)
        # lstm3_out, _ = self.multihead_attn(lstm3_out, lstm3_out, lstm3_out)
        # lstm3_out = lstm3_out.permute(1, 0, 2) 
        # print('attention shape: ', lstm3_out.shape)

        cnn_out = cnn_out.permute(0, 2, 1)
        # 64 x 248 x 128
        #print(self.Q1)
        atten1_out, _ = self.atten1(self.Q1, cnn_out, cnn_out)
        lstm1_out, _ = self.lstm1(atten1_out)
        #print('lstm1 out', lstm1_out.shape)

        # 64 x 128 x 128
        atten2_out, _ = self.atten2(self.Q2, lstm1_out, lstm1_out)
        lstm2_out, _ = self.lstm2(atten2_out)
        #print('lstm2 out', lstm2_out.shape)

        # 64 x 64  x 128
        atten3_out, _ = self.atten3(self.Q3, lstm2_out, lstm2_out)
        lstm3_out, _ = self.lstm3(atten3_out)
        #print('lstm3 out', lstm3_out.shape)

        # 64 x 32 x 128
        lstm3_out, _ = self.multihead_attn(lstm3_out, lstm3_out, lstm3_out)

        ### Stat Pool
        mean = torch.mean(lstm3_out,1)
        # print('mean shape: ', mean.shape)
        std = torch.var(lstm3_out,1)
        stat_pooling = torch.cat((mean,std),1)

        emo_predictions= self.fc(stat_pooling)
        return emo_predictions
    
    
    
