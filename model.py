import torch.nn as nn
from torchvision import models

class TJ_model(nn.Module):
    def __init__(self,config):
        super(TJ_model,self).__init__()
        self.device=config.device
        self.bk=models.resnet18(pretrained=True)
        self.bk.fc=nn.Linear(self.bk.fc.in_features, config.num_class)
        
        for name,param in self.bk.named_parameters():
            param.requires_grad = False
        self.bk.fc.weight.requires_grad=True    
        self.bk.fc.bias.requires_grad=True

    def forward(self,x):
        return self.bk(x)