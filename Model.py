import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16
model = vgg16(pretrained=True)
model.fc = torch.nn.Linear(4096,2)


class VGG_based_multi(nn.Module):
    def __init__(self,num_classes=2):
        super(VGG_based_multi,self).__init__()
        pretrained = model
        print('model structure————————',pretrained)
        self.module1 = pretrained.features[0:4]

        self.AVP = nn.Sequential(nn.AdaptiveAvgPool2d(1))

        self.Conv2 = nn.Sequential(
            # nn.Conv2d(64,256,kernel_size=3,stride=1,padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d((8,8)))

        self.module2 = pretrained.features[4:17]

        self.Conv3 = nn.Sequential(
            nn.Conv2d(588,512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )


        self.Classifier1 = torch.nn.Sequential(
            torch.nn.Linear(120 * 1 * 1, 100),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(100, 2))

        self.Classifier2 = torch.nn.Sequential(
            torch.nn.Linear(1 *1 * 468, 100),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(100, 2))

        self.Classifier3 = torch.nn.Sequential(
            torch.nn.Linear(3 * 3 * 512, 500),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(500, 2))

        self.Multi1 = Multi1()

        self.Multi2 = Multi2()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self,input):
        x1 = self.module1(input)
        x11 = self.AVP(x1)
        x111 = torch.subtract(x1,x11)

        # multi-scale feature extraction
        xx = self.Multi1(x111)

        x2 = self.module2(x111)

        x1111 = self.Conv2(xx)

        # multi-scale feature extraction
        xxx = self.Multi2(x2)

        # feature fusion
        x_double = torch.cat((x1111,xxx),dim=1)

        x_double = self.Conv3(x_double)

        x3 = x_double
        # print('********',x3.shape)

        x3 = x3.view(-1, 3*3*512)
        s3 = self.Classifier3(x3)

        # x1 = nn.AdaptiveAvgPool2d(1)
        x1 = self.avgpool(xx)
        # print('________', x1.shape)
        # x1 = x1.view(-1, 1*1*64)
        x1 = x1.view(x1.size(0), -1)
        s1 = self.Classifier1(x1)

        x2 = self.avgpool(xxx)
        # print('________', x2.shape)
        # x2 = nn.AdaptiveAvgPool2d(1)
        # x2 = x1.view(-1, 1 * 1 * 256)
        x2 = x2.view(x2.size(0), -1)
        s2 = self.Classifier2(x2)
        # print("____________",s1,s2,s3)
        return s1,s2,s3


class Multi1(nn.Module):
    def __init__(self):
        super(Multi1,self).__init__()

        self.MaxPooling2D11 = nn.Sequential (nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                                             nn.BatchNorm2d(64),
                                             nn.ReLU(),nn.MaxPool2d((2,2)))

        self.MaxPooling2D22 = nn.Sequential (nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                                             nn.BatchNorm2d(64),
                                             nn.ReLU(),nn.MaxPool2d((2,2)))


        self.MaxPooling2D33 = nn.Sequential (nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                                             nn.BatchNorm2d(64),
                                             nn.ReLU(),nn.MaxPool2d((2,2)))

        self.conv11 = nn.Sequential(nn.Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                    nn.BatchNorm2d(16),nn.ReLU())

        self.conv22 = nn.Sequential(nn.Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                    nn.BatchNorm2d(8),nn.ReLU())

        self.conv33 = nn.Sequential(nn.Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                    nn.BatchNorm2d(8),nn.ReLU())

        self.conv44 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                    nn.BatchNorm2d(32),nn.ReLU())

        self.Conv11 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        self.Conv111 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.Conv1111 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))


    def upsample2(self,x):
        return F.upsample(x,(56,56),mode='bilinear')
    def upsample4(self,x):
        return F.upsample(x,(112,112),mode='bilinear')
    def upsample8(self,x):
        return F.upsample(x,(224,224),mode='bilinear')


    def forward(self,input):
        x1_1 = self.MaxPooling2D11(input)
        # print("………………………………………………………………", x1_1.shape)
        x1_00 = self.conv44(input)
        #print("………………………………………………………………",x1_1.shape)
        x1_2 = self.MaxPooling2D22(x1_1)
        #print("×××××××××××××××",x1_2.shape)
        x1_3 = self.MaxPooling2D33(x1_2)
        #print("@@@@@@@@@@@@@@@@@@@@@@",x1_3.shape)
        x1_33 = self.conv33(x1_3)
        #print("！！！！！！！！！！！",x1_33.shape)
        x1_22 = self.conv22(x1_2)
        #print("________",x1_22.shape)
        x1_11 = self.conv11(x1_1)

        x1_333 = self.upsample2(x1_33)
        # print("________",x1_333.shape)
        # print("________", x1_22.shape)
        # x11_2 = torch.cat((x1_22,x1_333),dim=1)
        x11_2 = torch.add(x1_22, x1_333)

        # print('##################',x11_2.shape)
        x11_22 = self.Conv11(x11_2)
        x11_222 = self.upsample4(x11_22)
        # print('~~~~~~~~~~~~~~~',x11_222.shape)
        # print('~~~~~~~~~~~~~~~',x1_11.shape)
        # x11_1 = torch.cat((x11_222,x1_11),dim=1)
        x11_1 = torch.add(x11_222, x1_11)

        x11_11 = self.Conv111(x11_1)
        x11_111 = self.upsample8(x11_11)
        # x11_0 = torch.cat((x11_111,x1_00),dim=1)
        x11_0 = torch.add(x11_111, x1_00)

        x11_00 = self.Conv1111(x11_0)
        x11_11 = self.upsample8(x11_11)
        x11_22 = self.upsample8(x11_22)
        x11_33 = self.upsample8(x1_33)
        # print('%%%%%%%%%%%%%%%%%%%%%',x11_00.shape,x11_11.shape,x11_22.shape,x11_33.shape)
        x_multex = torch.cat((x11_00,x11_11,x11_22,x11_33),dim=1)
        return x_multex


class Multi2(nn.Module):
    def __init__(self):
        super(Multi2,self).__init__()

        self.conv11 = nn.Sequential (nn.Conv2d(256,56,kernel_size=(1,1),stride=(1,1),bias=False),
                                     nn.BatchNorm2d(56))

        self.MaxPooling2D11 = nn.Sequential (nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU(),nn.MaxPool2d((2,2)))

        self.conv22 = nn.Sequential (nn.Conv2d(256,28,kernel_size=(1,1),stride=(1,1),bias=False),nn.BatchNorm2d(28))

        self.MaxPooling2D22 = nn.Sequential (nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU(),nn.MaxPool2d((2,2)))

        self.conv33 = nn.Sequential (nn.Conv2d(256,28,kernel_size=(1,1),stride=(1,1),bias=False),nn.BatchNorm2d(28))
        self.MaxPooling2D33 = nn.Sequential (nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU(),nn.MaxPool2d((2,2)))

        self.conv44 = nn.Sequential (nn.Conv2d(256,128,kernel_size=(1,1),stride=(1,1),bias=False),nn.BatchNorm2d(128))

        self.Conv11 = nn.Sequential (
            nn.Conv2d(28,56,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(56),
            nn.ReLU(inplace=True))

        self.Conv111 = nn.Sequential (
            nn.Conv2d(56,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        self.Conv1111 = nn.Sequential (
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))


    def upsample2(self,x):
        return F.upsample(x,(7,7),mode='bilinear')
    def upsample4(self,x):
        return F.upsample(x,(14,14),mode='bilinear')
    def upsample8(self,x):
        return F.upsample(x,(28,28),mode='bilinear')


    def forward(self,input):
        x1_1 = self.MaxPooling2D11(input)
        x1_00 = self.conv44(input)
        #print("………………………………………………………………",x1_1.shape)
        x1_2 = self.MaxPooling2D22(x1_1)
        #print("×××××××××××××××",x1_2.shape)
        x1_3 = self.MaxPooling2D33(x1_2)
        #print("@@@@@@@@@@@@@@@@@@@@@@",x1_3.shape)
        x1_33 = self.conv33(x1_3)
        #print("！！！！！！！！！！！",x1_33.shape)
        x1_22 = self.conv22(x1_2)
        #print("________",x1_22.shape)
        x1_11 = self.conv11(x1_1)

        # print("××××××××××",x1_33.shape)
        x1_333 = self.upsample2(x1_33)
        # print("________",x1_333.shape)
        # print("++++++++=",x1_22.shape)

        # x11_2 = torch.cat((x1_22,x1_333),dim=1)
        x11_2 = torch.add(x1_22, x1_333)

        # print('##################',x11_2.shape)
        x11_22 = self.Conv11(x11_2)
        x11_222 = self.upsample4(x11_22)
        # print('~~~~~~~~~~~~~~~',x11_222.shape)
        # print('~~~~~~~~~~~~~~~',x1_11.shape)
        # x11_1 = torch.cat((x11_222,x1_11),dim=1)
        x11_1 = torch.add(x11_222, x1_11)

        # print('$$$$$$$$$$',x11_1.shape)
        x11_11 = self.Conv111(x11_1)
        x11_111 = self.upsample8(x11_11)
        # x11_0 = torch.cat((x11_111,x1_00),dim=1)
        x11_0 = torch.add(x11_111, x1_00)

        x11_00 = self.Conv1111(x11_0)
        x11_11 = self.upsample8(x11_11)
        x11_22 = self.upsample8(x11_22)
        x11_33 = self.upsample8(x1_33)
        # print('%%%%%%%%%%%%%%%%%%%%%',x11_00.shape,x11_11.shape,x11_22.shape,x11_33.shape)
        x_multex = torch.cat((x11_00,x11_11,x11_22,x11_33),dim=1)
        return x_multex
