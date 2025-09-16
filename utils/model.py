import os

import torch.nn as nn
import torch

from .CNN_F import image_net

class Image_Net(nn.Module):
    def __init__(self, bit, pretrain_model):
        super(Image_Net, self).__init__()
        self.bit = bit
        self.cnn_f = image_net(pretrain_model)
        self.image_module = nn.Sequential(
            nn.Linear(4096, 2048, bias=True),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(2048, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.hash_module = nn.Sequential(
            nn.Linear(1024, bit),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.cnn_f(x)
        x = x.contiguous().view(x.size(0), 4096)

        hid = self.image_module(x)
        code = self.hash_module(hid)

        return code, hid

    def load(self, path, use_gpu=False):
        if not use_gpu:
            self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(path))

    def save(self, name=None, path='./checkpoints/models', cuda_device=None):
        if not os.path.exists(path):
            os.makedirs(path)
        if cuda_device == 'cpu':
            torch.save(self.state_dict(), os.path.join(path, name))
        else:
            with torch.cuda.device(cuda_device):
                torch.save(self.state_dict(), os.path.join(path, name))
        return name


class Txt_Net(nn.Module):
    def __init__(self, txt_feat_len, bit):
        super(Txt_Net, self).__init__()
        self.bit = bit
        self.text_module = nn.Sequential(
            nn.Linear(txt_feat_len, 8192, bias=True),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(8192, 4096, bias=True),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(4096, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.hash_module = nn.Sequential(
            nn.Linear(1024, bit, bias=True),
            nn.Tanh()
        )

        for layer in self.hash_module:
            if isinstance(layer, nn.Linear):
                torch.nn.init.normal_(layer.weight, mean=0.0, std=0.3)

    def forward(self, x):
        hid = self.text_module(x)
        code = self.hash_module(hid)

        return code, hid

    def load(self, path, use_gpu=False):
        if not use_gpu:
            self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(path))

    def save(self, name=None, path='./checkpoints/models', cuda_device=None):
        if not os.path.exists(path):
            os.makedirs(path)
        if cuda_device == 'cpu':
            torch.save(self.state_dict(), os.path.join(path, name))
        else:
            with torch.cuda.device(cuda_device):
                torch.save(self.state_dict(), os.path.join(path, name))
        return name


class LabelNet(nn.Module):
    def __init__(self, code_len):
        super(LabelNet, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc_encode = nn.Linear(1024, code_len)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU(inplace=True)
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=0.2)
        torch.nn.init.normal_(self.fc_encode.weight, mean=0.0, std=0.2)

    def forward(self, x):
        feat = self.relu(self.fc1(x))
        hid = self.fc_encode(self.dropout(feat))
        code = torch.tanh(hid)

        return code, feat

    def load(self, path, use_gpu=False):
        if not use_gpu:
            self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(path))

    def save(self, name=None, path='./checkpoints/models', cuda_device=None):
        if not os.path.exists(path):
            os.makedirs(path)
        if cuda_device == 'cpu':
            torch.save(self.state_dict(), os.path.join(path, name))
        else:
            with torch.cuda.device(cuda_device):
                torch.save(self.state_dict(), os.path.join(path, name))
        return name
