
import torch 
import torch.nn as nn  
import torch.nn.functional as F
from models.atten_model import CNN_frontend

class Vad_Classify(nn.Module):
    def __init__(self):
        super(Vad_Classify, self).__init__()
        self.fc1 = nn.Linear(3,16)
        self.fc2 = nn.Linear(16, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 64)

        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 4)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = F.relu(self.fc5(out))
        out = self.fc6(out)
        return out
