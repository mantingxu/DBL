import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as Fun
import copy
from DBL import DBLANet

output_dim = 24
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
        f1 = self.data[index][1]
        f2 = self.data[index][2]
        f3 = self.data[index][3]
        label = self.data[index][4]
        return fx, f1, f2, f3, label

    # return the length of our dataset
    def __len__(self):
        return len(self.data)


test_dataset = ExampleDataset('../numpy/myResTest0520.npy')
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

path = "../weight/dbl2.pth"
model.load_state_dict(torch.load(path))
model.eval()
count = 0
# model.load_state_dict(best_model_wts)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with torch.no_grad():
    for i, (imagesQuery, imagesOne, imagesTwo, imagesThree, labels) in enumerate(test_loader):
        imagesQuery = imagesQuery.to(device)
        imagesOne = imagesOne.to(device)
        imagesTwo = imagesTwo.to(device)
        imagesThree = imagesThree.to(device)
        labels = Fun.one_hot(labels, num_classes=3)
        labels = labels.squeeze(1)
        labels = labels.to(torch.int64)
        labels = labels.to(device)
        # print(labels)
        outputs = model(imagesQuery, imagesOne, imagesTwo, imagesThree)

        value, indices = torch.max(outputs.data, 1)
        # print(torch.max(outputs.data, 1))
        value_label, indices_label = torch.max(labels.data, 1)

        print(indices_label.item())
        print(indices.item())
        if indices_label.item() == indices.item():
            count += 1

        # _, predicted = torch.max(outputs, 1)
        # print(torch.max(outputs, 1))
acc = count / test_dataset.__len__()
print('acc: ', acc)
