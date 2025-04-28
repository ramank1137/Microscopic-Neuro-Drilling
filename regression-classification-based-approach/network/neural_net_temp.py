
import torch.nn as nn
from resnet import Resnet50
import torch.nn.init as init
from network.base_temp import BaseModel

class NeuralNetwork(nn.Module):
    def __init__(self, cfg):
        super(NeuralNetwork, self).__init__()
        self.model = BaseModel(cfg)

        if cfg.backbone == 'resnet18':
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

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(hdim, 64),
            nn.ReLU(),
            nn.Linear(64,10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.model(x)
        logits = self.linear_relu_stack(x)
        import ipdb
        ipdb.set_trace()
        return logits
    
    def _initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode='fan_out')
            init.constant_(m.bias, 0)