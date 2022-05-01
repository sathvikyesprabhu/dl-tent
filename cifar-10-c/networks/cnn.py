import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Conv2d(3,16,5), # 16*24*24
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,32,5), # 32*20*20
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2), # 32*10*10
            nn.Conv2d(32,64,5), # 64*6*6
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2) #64*3*3
        )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(1024,100),
            nn.ReLU(),
            nn.Linear(100,10)
        )       
        
    def forward(self,x):
        out = self.layer(x)
        out = out.view(out.size(0),-1)
        out = self.fc_layer(out)

        return out