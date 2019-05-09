from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import torchvision
import math

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Vgg16_Customize(nn.Module):
    def __init__(self, num_classes=1000):
        super(Vgg16_Customize, self).__init__()
        self.num_classes = num_classes
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        # self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        # self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(inplace=True)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        # self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.bn3_3 = nn.BatchNorm2d(256)
        self.relu3_3 = nn.ReLU(inplace=True)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        # self.bn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.bn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.bn4_3 = nn.BatchNorm2d(512)
        self.relu4_3 = nn.ReLU(inplace=True)

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # self.bn5_1 = nn.BatchNorm2d(512)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # self.bn5_2 = nn.BatchNorm2d(512)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # self.bn5_3 = nn.BatchNorm2d(512)
        self.relu5_3 = nn.ReLU(inplace=True)

        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.classifier = nn.Sequential(
        #     nn.Linear(7*7*512, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(4096, self.num_classes)
        # )
        self.fc1 = nn.Linear(7*7*512, 4096)
        self.relu_cls1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu_cls2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(4096, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        # x = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        x = self.relu1_2(self.conv1_2(x))
        # x = self.relu1_2(self.bn1_2(self.conv1_2(x)))
        x = self.maxpool1(x)

        x = self.relu2_1(self.conv2_1(x))
        # x = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        x = self.relu2_2(self.conv2_2(x))
        # x = self.relu2_2(self.bn2_2(self.conv2_2(x)))
        x = self.maxpool2(x)

        x = self.relu3_1(self.conv3_1(x))
        # x = self.relu3_1(self.bn3_1(self.conv3_1(x)))
        x = self.relu3_2(self.conv3_2(x))
        # x = self.relu3_2(self.bn3_2(self.conv3_2(x)))
        x = self.relu3_3(self.conv3_3(x))
        # x = self.relu3_3(self.bn3_3(self.conv3_3(x)))
        x = self.maxpool3(x)

        x = self.relu4_1(self.conv4_1(x))
        # x = self.relu4_1(self.bn4_1(self.conv4_1(x)))
        x = self.relu4_2(self.conv4_2(x))
        # x = self.relu4_2(self.bn4_2(self.conv4_2(x)))
        x = self.relu4_3(self.conv4_3(x))
        # x = self.relu4_3(self.bn4_3(self.conv4_3(x)))
        x = self.maxpool4(x)

        x = self.relu5_1(self.conv5_1(x))
        # x = self.relu5_1(self.bn5_1(self.conv5_1(x)))
        x = self.relu5_2(self.conv5_2(x))
        # x = self.relu5_2(self.bn5_2(self.conv5_2(x)))
        x = self.relu5_3(self.conv5_3(x))
        # x = self.relu5_3(self.bn5_3(self.conv5_3(x)))
        x = self.maxpool5(x)

        x = x.view(x.size(0), -1)

        # x = self.classifier(x)
        x = self.relu_cls1(self.fc1(x))
        x = self.drop1(x)
        x = self.relu_cls2(self.fc2(x))
        x = self.fc3(x) 

        return x

def vgg16_customize(pretrained=False, **kwargs):
    ## Initial model
    model_pre = torchvision.models.vgg16(pretrained=True)
    ## Load weight from pre-trained model
    model_pre.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    ## Initial customize model
    model = Vgg16_Customize(**kwargs)
    ## Customize weight
    for param_tensor in model_pre.state_dict():
        model.state_dict()[param_tensor] = model_pre.state_dict()[param_tensor]
    if pretrained:
        model.load_state_dict(torch.load('file_pt_path', map_location=device))
    ## Return customized model
    return model
