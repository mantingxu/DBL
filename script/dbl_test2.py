import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as Fun
import copy
from DBL2 import DBLANet

output_dim = 128
inputDim = output_dim * 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DBLANet(inputDim).to(device)


class ExampleDataset(Dataset):
    # data loading
    def __init__(self, path):
        data = np.load(path, allow_pickle=True)
        self.data = data

    # working for indexing
    def __getitem__(self, index):
        fx = self.data[index][0]
        f_list = self.data[index][1]
        label = self.data[index][2]
        file = self.data[index][3]
        return fx, f_list, label, file

    # return the length of our dataset
    def __len__(self):
        return len(self.data)


# dataset = ExampleDataset('../numpy/myResTrain0520_1.npy')
# train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4, shuffle=True)
test_dataset = ExampleDataset('../numpy/myResTestAug.npy')
print(test_dataset.__len__())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
denseNet_top3_predict = ['12448', '12222', '12083', '325', '2311', '2321', '4061', '4115', '6356']

path = "../weight/dbl14.pth"
model.load_state_dict(torch.load(path))
model.eval()
count = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with torch.no_grad():
    for i, (imagesQuery, pillIdList, labels, file) in enumerate(test_loader):
        imagesQuery = imagesQuery.to(device)
        pillIdList = torch.stack(pillIdList, dim=1)
        pillIdList = torch.LongTensor(pillIdList)
        pillIdList = pillIdList.to(device)
        labels = list(labels)
        labels = [int(x) for x in labels]
        labels = torch.LongTensor(labels)
        labels = labels.to(device)
        labels_idx = []
        for idx in range(pillIdList.size(0)):
            labels_idx.append((pillIdList[idx] == labels[idx]).nonzero(as_tuple=True)[0].item())
        labels_idx = torch.LongTensor(labels_idx)
        labels_idx = labels_idx.to(device)

        outputs = model(imagesQuery, pillIdList)
        #print(outputs)
        value, indices = torch.max(outputs.data, 1)
        # index = denseNet_top3_predict.index(outputs.item[0])
        #print(indices.item())
        count += (indices == labels_idx).sum().item()

        if indices.item() != labels_idx.item():
            print(indices.item())
            print(pillIdList)
            print(file)
        #for x in range(24):
        #    if indices[x].item() != labels[x].item():
        #       print(file)
        # if indices_label.item() == indices.item():
        #     count += 1

        # _, predicted = torch.max(outputs, 1)
        # print(torch.max(outputs, 1))
print(test_dataset.__len__())
acc = count / test_dataset.__len__()
print('acc: ', acc)
print(count)
