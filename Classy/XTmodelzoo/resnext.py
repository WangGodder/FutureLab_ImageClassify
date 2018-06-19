import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from .resnext_features import resnext101_64x4d_features
import torch.nn.functional as F
__all__ = ['ResNeXt101_64x4d',
           'resnext101_64x4d',
           'ResNeXt101_Cross',
           'resnext101_Cross',
           'ResNeXt101_Cross_Triplet',
           'resnext101_Cross_Triplet']

pretrained_settings = {
    'resnext101_64x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/resnext101_64x4d-e77a0586.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    }
}

class ResNeXt101_Cross(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNeXt101_Cross, self).__init__()
        self.num_classes = num_classes
        self.features = resnext101_64x4d_features
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(2048, num_classes)


    def logits(self, input):
        x = self.avg_pool(input)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return F.log_softmax(x,dim=1)


def resnext101_Cross(num_classes=1000, pretrained='imagenet'):
    model = ResNeXt101_Cross(num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['resnext101_64x4d'][pretrained]
        '''assert num_classes == settings['num_classes'], \"num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)'''
        if num_classes == settings['num_classes']:
            model.load_state_dict(model_zoo.load_url(settings['url']))
            model.input_space = settings['input_space']
            model.input_size = settings['input_size']
            model.input_range = settings['input_range']
            model.mean = settings['mean']
            model.std = settings['std']
        else:
            '''model.load_state_dict(model_zoo.load_url(settings['url']))
            model.input_space = settings['input_space']
            model.input_size = settings['input_size']
            model.input_range = settings['input_range']
            model.mean = settings['mean']
            model.std = settings['std']
            last_linear.weight last_linear.bias'''
            pretrained_dict = model_zoo.load_url(settings['url'])
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            update_dict = {}
            for k, v in pretrained_dict.items():
                if str(k) != 'last_linear.weight' and str(k) !='last_linear.bias':
                    update_dict.setdefault(k, v)

            update = {k: v for k, v in update_dict.items() if k in model_dict}
            model_dict.update(update)
            model.load_state_dict(model_dict)

            model.input_space = settings['input_space']
            model.input_size = settings['input_size']
            model.input_range = settings['input_range']
            model.mean = settings['mean']
            model.std = settings['std']

    return model




class ResNeXt101_64x4d(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNeXt101_64x4d, self).__init__()
        self.num_classes = num_classes
        self.features = resnext101_64x4d_features
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        #self.last_linear = nn.Linear(2048, num_classes)
        self.preluip1 = nn.PReLU()

        self.ip1 = nn.Linear(2048, 2)
        self.ip2 = nn.Linear(2048, num_classes)

    def logits(self, input):
        x = self.avg_pool(input)
        x = x.view(x.size(0), -1)
        #x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        #ip1 = self.preluip1(self.ip1(x))
        ip2 = self.ip2(x)
        #ip1 = self.preluip1(self.ip1(x))
        #ip2 = self.ip2(ip1)
        return ip2, F.log_softmax(ip2,dim=1)
        # return x


def resnext101_64x4d(num_classes=1000, pretrained='imagenet'):
    model = ResNeXt101_64x4d(num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['resnext101_64x4d'][pretrained]
        '''assert num_classes == settings['num_classes'], \"num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)'''
        if num_classes == settings['num_classes']:
            model.load_state_dict(model_zoo.load_url(settings['url']))
            model.input_space = settings['input_space']
            model.input_size = settings['input_size']
            model.input_range = settings['input_range']
            model.mean = settings['mean']
            model.std = settings['std']
        else:
            '''model.load_state_dict(model_zoo.load_url(settings['url']))
            model.input_space = settings['input_space']
            model.input_size = settings['input_size']
            model.input_range = settings['input_range']
            model.mean = settings['mean']
            model.std = settings['std']
            last_linear.weight last_linear.bias'''
            pretrained_dict = model_zoo.load_url(settings['url'])
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            update_dict = {}
            for k, v in pretrained_dict.items():
                if str(k) != 'last_linear.weight' and str(k) !='last_linear.bias':
                    update_dict.setdefault(k, v)

            update = {k: v for k, v in update_dict.items() if k in model_dict}
            model_dict.update(update)
            model.load_state_dict(model_dict)

            model.input_space = settings['input_space']
            model.input_size = settings['input_size']
            model.input_range = settings['input_range']
            model.mean = settings['mean']
            model.std = settings['std']

    return model




class ResNeXt101_Cross_Triplet(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNeXt101_Cross_Triplet, self).__init__()
        self.num_classes = num_classes
        self.features = resnext101_64x4d_features
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(2048, num_classes)


    def logits(self, input):
        x = self.avg_pool(input)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x, F.log_softmax(x,dim=1)


def resnext101_Cross_Triplet(num_classes=1000, pretrained='imagenet'):
    model = ResNeXt101_Cross_Triplet(num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['resnext101_64x4d'][pretrained]
        '''assert num_classes == settings['num_classes'], \"num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)'''
        if num_classes == settings['num_classes']:
            model.load_state_dict(model_zoo.load_url(settings['url']))
            model.input_space = settings['input_space']
            model.input_size = settings['input_size']
            model.input_range = settings['input_range']
            model.mean = settings['mean']
            model.std = settings['std']
        else:
            '''model.load_state_dict(model_zoo.load_url(settings['url']))
            model.input_space = settings['input_space']
            model.input_size = settings['input_size']
            model.input_range = settings['input_range']
            model.mean = settings['mean']
            model.std = settings['std']
            last_linear.weight last_linear.bias'''
            pretrained_dict = model_zoo.load_url(settings['url'])
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            update_dict = {}
            for k, v in pretrained_dict.items():
                if str(k) != 'last_linear.weight' and str(k) !='last_linear.bias':
                    update_dict.setdefault(k, v)

            update = {k: v for k, v in update_dict.items() if k in model_dict}
            model_dict.update(update)
            model.load_state_dict(model_dict)

            model.input_space = settings['input_space']
            model.input_size = settings['input_size']
            model.input_range = settings['input_range']
            model.mean = settings['mean']
            model.std = settings['std']

    return model


