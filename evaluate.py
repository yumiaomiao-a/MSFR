import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
import os            
import matplotlib.pyplot as plt
import time
import torch.nn as nn
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import recall_score,f1_score,precision_score


# import data path
data_dir = './celeb-df-120-60(1.3)'
# data_dir = './timit-hq-10000-2800'
#data_dir = './data-400-train-test'
# data_dir = './new_add_exp_FF++/DF/c23'
# data_dir = './ff_all_new/data_c40'
# data_dir = './new_add_exp_FF++/F2F/c23'
# data_dir = './new_add_exp_FF++/FS/c23'
# data_dir = './second paper/face_dect/new_add_exp_FF++/NT/c23'


data_transform = {
    'train':transforms.Compose([
        transforms.Scale([224,224]),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    'test':transforms.Compose([
        transforms.Scale([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
}


image_datasets = {x:datasets.ImageFolder(root = os.path.join(data_dir,x),
                                         transform = data_transform[x]) for x in ['train', 'test']}

train_set = image_datasets['train']
test_set = image_datasets['test']


dataloader = {x:torch.utils.data.DataLoader(dataset = image_datasets[x],
                                            batch_size = 32,
                                            shuffle = True) for x in ['train','test'] }  # 读取完数据后，对数据进行装载


train_dataloader = dataloader['train']
test_dataloader = dataloader ['test']

dataset_size = {x:len(image_datasets[x]) for x in ['train','test']}


from model_final_addition import VGG_based_multi
model = VGG_based_multi()

## load the trained model
dic = torch.load('./save_model/celeb_MSFR_A.pth')
new_state_dict = {}
for k,v in dic.items():
    new_state_dict[k[7:]] = v
model.load_state_dict(new_state_dict)


def adjust_learning_rate(epoch):
    lr = 0.0002
    if epoch > 10:
        lr = lr / 10
    elif epoch > 20:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 1000
    elif epoch > 40:
        lr = lr / 10000
    elif epoch > 50:
        lr = lr / 100000
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


# multi-gpu
model = torch.nn.DataParallel(model,device_ids=[0,1])
model = model.cuda()

epoch_n = 60

def save_models(epoch):
    torch.save(model.state_dict(), "./save_model/model_{}.pth".format(epoch))
print("Chekcpoint saved")


def test():
    model.eval()
    test_acc = 0.0
    prob_all = []
    label_all = []
    prob_all_soft = []
    for i, (images, labels) in enumerate(test_dataloader):

        images = images.cuda()
        labels1 = labels.cuda()
        labels = labels.numpy().astype(np.float)

        with torch.no_grad():
            # y_pred1, y_pred2, y_pred3 = model(images)
            # outputs = (y_pred1+y_pred2+y_pred3)/3
            # outputs = model(images)
            s1,s2,s3 = model(images)

        _, prediction1 = torch.max(s1.data, 1)
        _, prediction2 = torch.max(s2.data, 1)
        _, prediction3 = torch.max(s3.data, 1)

        pred = prediction1 + prediction2 + prediction3
        prediction = torch.where(pred >= 2, 1, 0)

        test_acc += torch.sum(prediction == labels1.data)

        pred1 = s1+s2+s3
        # print('--------',pred1)
        pred = torch.sigmoid(pred1)

        w_res_cc = torch.max(pred,1)[1].cpu().numpy()
        out_pred = w_res_cc
        prob_all.extend(out_pred)
        label_all.extend(labels)
        prob_all_soft.extend(pred)


    label_all = np.array(label_all).astype(np.int64)
    prob_all = np.array(prob_all).astype(np.int64)
    acc_valid = metrics.accuracy_score(label_all, prob_all)
    recall_valid = recall_score(label_all, prob_all)
    precision_valid = precision_score(label_all, prob_all)
    f1_valid = f1_score(label_all, prob_all)

    label_all = label_all.astype(np.float32)
    prob_all_soft = torch.tensor([item.cpu().detach().numpy() for item in prob_all_soft])
    auc_valid = metrics.roc_auc_score(label_all, 1-prob_all_soft[:,0])
    # test_acc_org = test_acc / len(test_set)

    print('----',acc_valid,auc_valid,recall_valid,precision_valid,f1_valid)
    return acc_valid,auc_valid,recall_valid,precision_valid,f1_valid


if __name__ == '__main__':
    test()
