
import torch 
import torch.nn as nn  
import torch.nn.functional as F
from models.atten_model import CNN_frontend

class Vad_Classify(nn.Module):
    def __init__(self):
        super(Vad_Classify, self).__init__()
        self.fc1 = nn.Linear(3,64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, 64)
        self.fc7 = nn.Linear(64, 4)

    def forward(self, x):
        out1 = F.relu(self.fc1(x))
        out2 = F.relu(self.fc2(out1))
        out3 = F.relu(self.fc3(out2))
        out4 = F.relu(self.fc4(out3 + out1))
        out5 = F.relu(self.fc5(out4))
        out6 = F.relu(self.fc6(out5))
        out7 = self.fc7(out6 + out3 + out1)
        return out7
