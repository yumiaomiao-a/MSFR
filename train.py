#_*_coding:utf-8 _*_
import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import torch.nn as nn
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from data_augmentation import face_eraser_gray
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import recall_score,f1_score,precision_score
from data_augmentation import face_eraser_gray,bg_eraser_gray,face_eraser_shuffle,bg_eraser_shuffle,face_eraser_change,bg_eraser_change
import cv2



# 读取数据
data_dir = './celeb-df-120-60(1.3)'
# data_dir = './timit-hq-10000-2800'
# data_dir = './data-400-train-test'
# data_dir = './ff++_all_new/data_c40'
# data_dir = './ff++_all_new/data_c40'
# data_dir = './FF++/F2F/c23'
# data_dir = './FF++/FS/c23'
# data_dir = './FF++/NT/c23'
# data_dir = './FF++/DF/c23'


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length
    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img


data_transform = {
    'train':transforms.Compose([
        transforms.Scale([224,224]),
        transforms.RandomHorizontalFlip(),
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


from Model import VGG_based_multi
model = VGG_based_multi()

## load the trained model
# dic = torch.load('xxx.pth')
# new_state_dict = {}
# for k,v in dic.items():
#     new_state_dict[k[7:]] = v
# model.load_state_dict(new_state_dict)


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
#loss_f = torch.nn.CosineEmbeddingLoss()
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

    return acc_valid,auc_valid,recall_valid,precision_valid,f1_valid



def train(num_epochs):
    best_acc = 0.0
    best_auc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0

        # print("_____________",train_dataloader)
        for i, (images, labels) in enumerate(train_dataloader):
            # put image and label to GPU
            i+=1
            images = images.cuda()
            labels = labels.cuda()

            #data augmentation
            for j in range(len(images)):
                p = np.random.rand()
                # print(p)
                if p<0.3:
                    images[j] = images[j]
                    # print('++++++==',images[i].shape)
                    # plt.imshow(images[i])
                    # plt.show()
                elif 0.3<p<0.416:
                    try:
                        images[j] = face_eraser_gray(images[j])
                        p1 = np.random.rand()
                        if p1<0.5:
                            labels[j] = 1-labels[j]
                        else:
                            labels[j] = 1-labels[j]
                    except:
                        images[j] = images[j]
                        labels[j] = labels[j]

                elif 0.416<p<0.533:
                    try:
                        images[j] = bg_eraser_gray(images[j])
                    except:
                        images[j] = images[j]

                elif 0.533<p<0.65:
                    try:
                        images[j] = face_eraser_shuffle(images[j])
                        p1 = np.random.rand()
                        # if p1<0.5:
                        #     labels[j] = 1-labels[j]
                        # else:
                        #     labels[j] = 1-labels[j]
                    except:
                        images[j] = images[j]

                elif 0.65<p<0.766:
                    try:
                        images[j] = bg_eraser_shuffle(images[j])
                    except:
                        images[j] = images[j]

                elif 0.766<p<0.882:
                    try:
                        l = labels[j]
                        fill = []
                        for m in range(len(images)):
                            if labels[m]!=l:
                                fill.append(images[m])

                        images[j] = face_eraser_change(images[j],fill)
                        labels[j] = 1-labels[j]
                    except:
                        images[j] = images[j]

                else:
                    try:
                        l = labels[j]
                        fill = []
                        for m in range(len(images)):
                            if labels[m]!=l:
                                fill.append(images[m])

                        images[j] = bg_eraser_change(images[j],fill)

                    except:
                        images[j] = images[j]



            # 清除所有累积梯度
            optimizer.zero_grad()
            s1, s2, s3 = model(images)

            # 根据实际标签和预测值计算损失
            loss1 = loss_f(s1, labels)
            loss2 = loss_f(s2, labels)
            loss3 = loss_f(s3, labels)
            # loss1 = loss_f(y_pred1, labels)
            # loss2 = loss_f(y_pred2, labels)
            # loss3 = loss_f(y_pred3, labels)
            loss = loss1 + 10*loss2 + 10*loss3
            # 传播损失
            loss.backward()

            # 根据计算的梯度调整参数
            optimizer.step()

            train_loss += loss.cpu().item() * images.size(0)
            # _, prediction = torch.max(outputs.data, 1)

            _, prediction1 = torch.max(s1.data, 1)
            _, prediction2 = torch.max(s2.data, 1)
            _, prediction3 = torch.max(s3.data, 1)

            prediction = prediction1 + prediction2 + prediction3
            prediction = torch.where(prediction >= 2, 1, 0)

            train_acc += torch.sum(prediction == labels.data)
            batch_size=32
            if i% 5 == 0:
                batch_loss = train_loss / (batch_size*i)
                batch_acc = train_acc / (batch_size*i)
                print('Epoch[{}] batch[{}],Loss:{:.4f},Acc:{:.4f}'.format( epoch, i, batch_loss, batch_acc))

            torch.cuda.empty_cache()

        # 调用学习率调整函数
        adjust_learning_rate(epoch)


        train_acc = train_acc / len(train_set)
        train_loss = train_loss / len(train_set)

        # test_acc, test_auc = test()
        test_acc, test_auc,recall, precision,f1 = test()

        # 若测试准确率高于当前最高准确率，则保存模型
        if test_acc > best_acc:
            save_models(epoch)
            best_acc = test_acc

        if test_auc > best_auc:
            best_auc = test_auc

        # 打印度量
        print(
            "Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {},Best Acc:{},Test AUC:{}, Best AUC:{},recall:{}, precision:{},f1:{}".format(
                epoch, train_acc, train_loss, test_acc, best_acc, test_auc, best_auc,recall, precision,f1))


if __name__ == '__main__':
    train(60)
    # test()
