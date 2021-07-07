#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:59:45 2020

@author: krishna

"""

import torch
import torch.nn as nn 
import torch.nn.functional as F
from models.tdnn import TDNN 

class CNN_frontend(nn.Module):
    def __init__(self):
        super(CNN_frontend, self).__init__()
        self.nin1 = NIN(1, 64, kernel_size=7, stride=4, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool1 = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

        self.nin2 = NIN(64, 128, kernel_size=7, stride=4, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.maxpool2 = nn.AvgPool1d(kernel_size=4, stride=2)

        self.nin3 = nn.Conv1d(128, 128, kernel_size=7, stride=4)
        self.bn3 = nn.BatchNorm1d(128)
        self.maxpool3 = nn.AvgPool1d(kernel_size=4, stride=2)


    def forward(self, x):
        # N x 1 x 16000*8
        out = self.nin1(x)
        out = F.relu(self.bn1(out))
        out = self.maxpool1(out)

        out = self.nin2(out)
        out = F.relu(self.bn2(out))
        out = self.maxpool2(out)

        out = self.nin3(out)
        out = F.relu(self.bn3(out))
        out = self.maxpool3(out)
        return out

class NIN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(NIN, self).__init__()
        self.blk = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=2, padding=padding, bias=bias),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, dilation=2, bias=False),
            nn.ReLU(), 
            nn.Conv1d(out_channels, out_channels, kernel_size=1, dilation=2, bias=False),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.blk(x)




class Atten_Model(nn.Module):
    def __init__(self, args):
        super(Atten_Model, self).__init__()
        self.cnn_frontend = CNN_frontend()
        
        # self.tdnn1 = TDNN(input_dim=128, output_dim=128, context_size=3, dilation=1,dropout_p=0.5)
        self.Q1 = torch.nn.Parameter(torch.zeros(64, args.batch_size, 128))
        self.atten1 = nn.MultiheadAttention(embed_dim=128, num_heads=8)

        self.lstm1 = nn.LSTM(input_size=128, hidden_size=64,num_layers=1,bidirectional=True,dropout=0.5,batch_first=True)     
        
        # self.tdnn2 = TDNN(input_dim=128, output_dim=128, context_size=7, dilation=3,dropout_p=0.5)
        # self.tdnn3 = TDNN(input_dim=128, output_dim=128, context_size=7, dilation=3,dropout_p=0.5)
        self.Q2 = torch.nn.Parameter(torch.zeros(64, args.batch_size, 128))
        self.atten2 = nn.MultiheadAttention(embed_dim=128, num_heads=8)

        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64,num_layers=1,bidirectional=True,dropout=0.5,batch_first=True)     
        
        # self.tdnn4 = TDNN(input_dim=128, output_dim=128, context_size=7, dilation=3,dropout_p=0.5)
        # self.tdnn5 = TDNN(input_dim=128, output_dim=128, context_size=7, dilation=3,dropout_p=0.5)
        self.Q3 = torch.nn.Parameter(torch.zeros(64, args.batch_size, 128))
        self.atten3 = nn.MultiheadAttention(embed_dim=128, num_heads=8)

        self.lstm3 = nn.LSTM(input_size=128, hidden_size=64,num_layers=1,bidirectional=True,dropout=0.5,batch_first=True)     
        self.multi_head1 = nn.MultiheadAttention(embed_dim=128, num_heads=8)
        self.multi_head2 = nn.MultiheadAttention(embed_dim=128, num_heads=8)
        self.fc_c1 = nn.Linear(2 * 128, 64)
        self.fc_c2 = nn.Linear(64, args.num_classes)
        self.fc_d1 = nn.Linear(2 * 128, 64)
        self.fc_d2 = nn.Linear(64, 3)
        
    def forward(self, inputs):
        cnn_out = self.cnn_frontend(inputs)
        cnn_out = cnn_out.permute(0, 2, 1)

        cnn_out = cnn_out.permute(1, 0, 2)
        atten1_out, _ = self.atten1(self.Q1, cnn_out, cnn_out)
        atten1_out = atten1_out.permute(1, 0, 2)
        lstm1_out, _ = self.lstm1(atten1_out)
        #print('lstm1 out', lstm1_out.shape)

        lstm1_out = lstm1_out.permute(1, 0, 2)
        atten2_out, _ = self.atten2(self.Q2, lstm1_out, lstm1_out)
        atten2_out = atten2_out.permute(1, 0, 2)
        lstm2_out, _ = self.lstm2(atten2_out)
        #print('lstm2 out', lstm2_out.shape)


        lstm2_out = lstm2_out.permute(1, 0, 2)
        lstm2_out = torch.cat((lstm2_out, cnn_out), 0)
        atten3_out, _ = self.atten3(self.Q3, lstm2_out, lstm2_out)
        atten3_out = atten3_out.permute(1, 0, 2)
        lstm3_out, _ = self.lstm3(atten3_out)
        #print('lstm3 out', lstm3_out.shape)

        lstm3_out = lstm3_out.permute(1, 0, 2)
        lstm3_out = torch.cat((lstm3_out, lstm1_out))

        c_out, _ = self.multi_head1(lstm3_out, lstm3_out, lstm3_out)
        c_out = c_out.permute(1, 0, 2)
        c_mean = torch.mean(c_out, 1)
        c_std = torch.var(c_out, 1)
        c_pooling = torch.cat((c_mean, c_std), 1)
        c_pred = self.fc_c1(c_pooling)
        c_pred = self.fc_c2(c_pred)

        d_out, _ = self.multi_head2(lstm3_out, lstm3_out, lstm3_out)
        d_out = d_out.permute(1, 0, 2)
        d_mean = torch.mean(d_out, 1)
        d_std = torch.var(d_out, 1)
        d_pooling = torch.cat((d_mean, d_std), 1)
        d_pred= self.fc_d1(d_pooling)
        d_pred = self.fc_d2(d_pred)

        return c_pred, d_pred
    
    
    
