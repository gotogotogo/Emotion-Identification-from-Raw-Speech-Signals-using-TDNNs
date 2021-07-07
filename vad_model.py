
import torch 
import torch.nn as nn  
import torch.nn.functional as F

class Vad_Classify(nn.Module):
    def __init__(self):
        super(Vad_Classify, self).__init__()
        self.fc1 = nn.Linear(3,16)
        self.fc2 = nn.Linear(16, 32)

        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 4)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out
