import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import timm



class BaseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.backbone == 'resnet18':
            backbone = models.resnet18(pretrained=True)
            backbone.fc = nn.Identity()
            self.encoder = backbone
        
        elif cfg.backbone == 'resnet34':
            backbone = models.resnet34(pretrained=True)
            backbone.fc = nn.Identity()
            self.encoder = backbone

        elif cfg.backbone == 'vgg16':
            backbone = models.vgg16_bn(pretrained=True)
            backbone.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            backbone.classifier = nn.Identity()
            self.encoder = backbone

        elif cfg.backbone == 'alex':
            backbone = models.alexnet(pretrained=True)
            backbone.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            backbone.classifier = nn.Identity()
            self.encoder = backbone

        elif cfg.backbone == 'efficient':
            backbone = models.efficientnet_v2_s(pretrained=True)
            backbone.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            backbone.classifier = nn.Identity()
            self.encoder = backbone
            
        elif cfg.backbone == 'convnext':
            backbone = models.convnext_base(pretrained=True)
            #backbone.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            #backbone.classifier = nn.Identity()
            self.encoder = backbone

        elif cfg.backbone == 'swin':
            backbone = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
            #backbone.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            backbone.head = nn.Identity()
            self.encoder = backbone
        
        elif cfg.backbone == 'vit':
            backbone = models.vit_b_16(weights=models.ViT_B_16_Weights)
            #backbone.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            backbone.heads = nn.Identity()
            self.encoder = backbone

        elif cfg.backbone == 'res2net':
            backbone = timm.create_model('res2net101_26w_4s', pretrained=True)
            #backbone.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            #backbone. = nn.Identity()
            self.encoder = backbone

        elif cfg.backbone == 'deit':
            backbone = timm.create_model('deit_small_patch16_224.fb_in1k', pretrained=True)
            #backbone.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            #backbone. = nn.Identity()
            self.encoder = backbone
            
        elif cfg.backbone == 'gcvit':
            backbone = timm.create_model('gcvit_small.in1k', pretrained=True)
            self.encoder = backbone

        elif cfg.backbone == 'cait':
            backbone = timm.create_model('cait_s24_224.fb_dist_in1k', pretrained=True)
            #backbone.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            #backbone. = nn.Identity()
            self.encoder = backbone
        
        elif cfg.backbone == 'convnextv2':
            backbone = timm.create_model('convnextv2_base.fcmae', pretrained=True)
            self.encoder = backbone

        elif cfg.backbone == 'ghostnetv2':
            backbone = timm.create_model('ghostnetv2_130.in1k', pretrained=True)
            self.encoder = backbone

        elif cfg.backbone == 'inceptnext':
            backbone = timm.create_model('inception_next_base.sail_in1k', pretrained=True)
            self.encoder = backbone

        elif cfg.backbone == 'fastvit':
            backbone = timm.create_model('fastvit_s12.apple_in1k', pretrained=True)
            self.encoder = backbone
        
        elif cfg.backbone == 'maxvit':
            backbone = timm.create_model('maxvit_rmlp_small_rw_224.sw_in1k', pretrained=True)
            self.encoder = backbone

        elif cfg.backbone == 'vgg16v2':  # no bn, relu, maxpool after last convolution
            backbone = models.vgg16_bn(pretrained=True)
            backbone.features[41] = nn.Identity()
            backbone.features[42] = nn.Identity()
            backbone.features[43] = nn.Identity()
            backbone.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            backbone.classifier = nn.Identity()
            self.encoder = backbone

        elif cfg.backbone == 'vgg16v2norm':  # no bn, relu, maxpool after last convolution
            backbone = models.vgg16_bn(pretrained=True)
            backbone.features[41] = nn.Identity()
            backbone.features[42] = nn.Identity()
            backbone.features[43] = nn.Identity()
            backbone.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            class Normalization(torch.nn.Module):
                def __init__(self, dim=-1):
                    super().__init__()
                    self.dim = dim
                def forward(self, x):
                    return nn.functional.normalize(x, dim=self.dim)
            backbone.classifier = Normalization()
            self.encoder = backbone

        elif cfg.backbone == 'vgg16fc':
            backbone = models.vgg16_bn(pretrained=True)
            backbone.classifier[5] = nn.Identity()
            backbone.classifier[6] = nn.Identity()
            self.encoder = backbone
        else:
            raise ValueError(f'Not supported backbone architecture {cfg.backbone}')

    def forward(self, x_base, x_ref=None):
        # feature extraction
        base_embs = self.encoder(x_base)
        if x_ref is not None:
            ref_embs = self.encoder(x_ref)
            out = self._forward(base_embs, ref_embs)
            return out, base_embs, ref_embs
        else:
            #out = self._forward(base_embs)
            out = base_embs
            return out

    def _forward(self, base_embs, ref_embs=None):
        raise NotImplementedError('Suppose to be implemented by subclass')