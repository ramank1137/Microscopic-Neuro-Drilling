
import torch.nn as nn
import torch.nn.init as init
from network.base import BaseModel

class NeuralNetwork(nn.Module):
    def __init__(self, cfg):
        super(NeuralNetwork, self).__init__()
        self.model = BaseModel(cfg)

        if cfg.backbone == 'resnet18':
            hdim = 512
        elif cfg.backbone == 'resnet34':
            hdim = 512
        elif cfg.backbone == 'vgg16':
            hdim = 512
        elif cfg.backbone == 'vgg16v2':
            hdim = 512
        elif cfg.backbone == 'vgg16v2norm':
            hdim = 512
        elif cfg.backbone == 'alex':
            hdim = 256
        elif cfg.backbone == 'efficient':
            hdim = 1280
        elif cfg.backbone == 'convnext':
            hdim = 1000
        elif cfg.backbone == 'swin':
            hdim = 1024
        elif cfg.backbone == 'res2net':
            hdim = 1000
        elif cfg.backbone == 'vit':
            hdim = 768
        elif cfg.backbone == 'deit':
            hdim = 1000
        elif cfg.backbone == 'cait':
            hdim = 1000
        elif cfg.backbone == 'gcvit':
            hdim = 1000
        elif cfg.backbone == 'ghostnetv2':
            hdim = 1000
        elif cfg.backbone == 'convnextv2':
            hdim = 1024
        elif cfg.backbone == 'inceptnext':
            hdim = 1000
        elif cfg.backbone == 'fastvit':
            hdim = 1000
        elif cfg.backbone == 'maxvit':
            hdim = 1000
        
        if cfg.loss == "mse":
            self.linear_relu_stack = nn.Sequential(
            nn.Linear(hdim, 64),
            nn.ReLU(),
            nn.Linear(64,10),
            nn.ReLU(),
            nn.Linear(10, 1)
            )
        else:
            self.linear_relu_stack = nn.Sequential(
            nn.Linear(hdim, 64),
            nn.ReLU(),
            nn.Linear(64,10)
            #nn.ReLU(),
            #nn.Linear(10, 1)
            )
            

    def forward(self, x):
        x = self.model(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    def _initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode='fan_out')
            init.constant_(m.bias, 0)