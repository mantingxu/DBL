import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import copy
from DBL4 import DBLANet
import matplotlib.pyplot as plt
import json

output_dim = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 30
learning_rate = 0.0001
inputDim = output_dim * 3

model = DBLANet(inputDim).to(device)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.02)


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


def saveModel():
    path = "../weight/dbl_pill.pth"
    torch.save(model.state_dict(), path)
    print('save')


def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None


try:
    json_file = open('../label/new_pill_class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

keys = []
denseNet_top3_predict = ['12448', '12222', '12083', '325', '2311', '2321', '4061', '4115', '6356']
for pill in denseNet_top3_predict:
    key = get_key_from_value(class_indict, pill)
    keys.append(key)

dataset = ExampleDataset('../numpy/pillTrain.npy')
print(dataset.__len__())
train_length = int(dataset.__len__() * 0.6)
valid_length = dataset.__len__() - train_length
train_set, valid_set = torch.utils.data.random_split(dataset, [train_length, valid_length])
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=1, shuffle=True)

best_loss = 999
history = []
val_history = []
output_cat = []

for epoch in range(num_epochs):
    total_val = 0
    total_loss = 0
    print(epoch)
    for i, (imagesQuery, pillIdList, labels, file) in enumerate(train_loader):
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
        # init optimizer
        optimizer.zero_grad()


        # forward -> backward -> update
        outputs = model(imagesQuery, pillIdList)
        value, indices = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels_idx)
        total_loss += loss.item()
        #for param in model.named_parameters():
        #    print(param[0], param[1].requires_grad)
        loss.backward()
        optimizer.step()
        #for idx in range(0,36):
        #    if str(idx) not in keys:
        #        model.embedding.weight.data[idx] = 0

    print(f'epoch {epoch + 1}/{num_epochs}, loss = {total_loss:.4f}')
    history.append(total_loss)

    if total_loss < best_loss:
        best_loss = total_loss
        saveModel()
        best_model_wts = copy.deepcopy(model.state_dict())
    # history.append(loss.item())

    for j, (imagesQuery_val, pillIdList_val, labels_val, file_val) in enumerate(val_loader):
        model.eval()
        imagesQuery_val = imagesQuery_val.to(device)
        pillIdList_val = torch.stack(pillIdList_val, dim=1)
        pillIdList_val = torch.LongTensor(pillIdList_val)
        pillIdList_val = pillIdList_val.to(device)
        labels_val = list(labels_val)
        labels_val = [int(x) for x in labels_val]
        labels_val = torch.LongTensor(labels_val)
        labels_val = labels_val.to(device)
        labels_idx_val = []
        for idx_val in range(pillIdList_val.size(0)):
            labels_idx_val.append((pillIdList_val[idx_val] == labels_val[idx_val]).nonzero(as_tuple=True)[0].item())
        labels_idx_val = torch.LongTensor(labels_idx_val)
        labels_idx_val = labels_idx_val.to(device)
        # init optimizer
        optimizer.zero_grad()
        with torch.no_grad():
            outputs = model(imagesQuery_val, pillIdList_val)
            value, indices = torch.max(outputs.data, 1)
            total_val += (indices == labels_idx_val).sum().item()

    acc_val = total_val / valid_set.__len__()
    print('val acc: ', acc_val)

print(best_loss)

print('Finished Training')
epochs = range(0, num_epochs)
plt.plot(epochs, history, 'g', label='Training loss')
plt.title('Train loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
