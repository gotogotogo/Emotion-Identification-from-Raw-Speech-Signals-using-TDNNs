import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from vad_dataset import VadDataset
from vad_model import Vad_Classify

import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')
torch.multiprocessing.set_sharing_strategy('file_system')

########## Argument parser
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-input_dim', type=int, default=1)
parser.add_argument('-num_classes', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('-num_epochs', type=int, default=1000)
parser.add_argument('--raw_wav_path', type=str, default='raw_wavs.pkl')
parser.add_argument('--lr', type=float, default=0.0001)
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



def train(train_loader,epoch, model, device, optimizer, criterion):
    train_loss_list=[]
    full_preds=[]
    full_gts=[]
    model.train()
    train_loader = tqdm(train_loader)
    for step, (vad, labels) in enumerate(train_loader):
        #print(features.shape)
        print('vad shape: ', vad.shape)
        print('labels shape: ', labels.shape)
        print(type(vad))
        labels = labels.reshape(-1).to(device)
        print(type(labels))
        vad = vad.to(device)
        vad.requires_grad = True
        optimizer.zero_grad()
        preds = model(vad)
        #### CE loss
        loss = criterion(preds,labels)
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.item())
        #train_acc_list.append(accuracy)
        if step % 10 == 0:
            train_loader.desc = "[epoch {} step {}] mean loss {}".format(epoch, step, np.mean(np.asarray(train_loss_list)))
        
        predictions = np.argmax(preds.detach().cpu().numpy(),axis=1)
        for pred in predictions:
            full_preds.append(pred)
        for lab in labels.detach().cpu().numpy():
            full_gts.append(lab)
            
    unweighted_avg_recall = round(balanced_accuracy_score(full_gts,full_preds), 5)
    mean_loss = round(np.mean(np.asarray(train_loss_list)), 5)
    print('[epoch {}] train loss {} train unweighted average recall {}'.format(epoch, mean_loss, unweighted_avg_recall))
    


def test(test_loader,epoch, model, device, criterion):
    model.eval()
    with torch.no_grad():
        val_loss_list=[]
        full_preds=[]
        full_gts=[]
        for (vad, labels) in test_loader:
            vad = vad.to(device)
            labels = labels.to(device)
            preds= model(vad)
            #### CE loss
            loss = criterion(preds,labels)
            val_loss_list.append(loss.item())
            #train_acc_list.append(accuracy)
            predictions = np.argmax(preds.detach().cpu().numpy(),axis=1)
           
            for pred in predictions:
                full_preds.append(pred)
            for lab in labels.detach().cpu().numpy():
                full_gts.append(lab)
                
        unweighted_avg_recall = round(balanced_accuracy_score(full_gts,full_preds), 5)
        mean_loss = round(np.mean(np.asarray(val_loss_list)), 5)
        print('[epoch {}] test loss {} test unweighted average recall {}\n'.format(epoch, mean_loss, unweighted_avg_recall))
    return unweighted_avg_recall, mean_loss, full_gts, full_preds

def main(args):
    print("Pytorch version:{}".format(torch.__version__))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('device: ', device)
    
    train_set = VadDataset(args.raw_wav_path, mode='train', test_sess=5)
    print('len of train set: ', len(train_set))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=False)  

    test_set = VadDataset(args.raw_wav_path, mode='test', test_sess=5)
    print('len of test set: ', len(test_set))
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=8, pin_memory=False)

    model = Vad_Classify().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.000001, betas=(0.9, 0.98), eps=1e-9)
    #loss_fun = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss().to(device)


    best_acc = 0.0
    target_names = ['happy', 'angry', 'sad', 'neutral']
    for epoch in range(args.num_epochs):
        train(train_loader, epoch, model, device, optimizer, criterion)
        unweighted_avg_recall, mean_loss, full_gts, full_preds = test(test_loader, epoch, model, device, criterion)
        if unweighted_avg_recall > best_acc:
            print('-' * 20, 'best model in epoch {} '.format(epoch), '-' * 20) 
            best_acc = unweighted_avg_recall
            print(classification_report(full_gts, full_preds, target_names=target_names))
            print(confusion_matrix(full_gts, full_preds))
            model_save_path = os.path.join('vad_save_model', 'best_'+str(epoch)+'_'+str(round(unweighted_avg_recall, 5)))
            state_dict = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}
            torch.save(state_dict, model_save_path)

if __name__ == '__main__':
    main(args)