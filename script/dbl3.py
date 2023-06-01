import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import copy
from DBL3 import DBLANet
import matplotlib.pyplot as plt
import json

output_dim = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 80
learning_rate = 0.0001
inputDim = output_dim * 3

model = DBLANet(inputDim).to(device)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)


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
        return fx, np.array(f_list), label, file

    # return the length of our dataset
    def __len__(self):
        return len(self.data)


def saveModel():
    path = "../weight/dbl0601.pth"
    torch.save(model.state_dict(), path)
    print('save')


def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None


try:
    json_file = open('../label/capsule_class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

keys = []
denseNet_top3_predict = ['12448', '12222', '12083']
for pill in denseNet_top3_predict:
    key = get_key_from_value(class_indict, pill)
    keys.append(key)

dataset = ExampleDataset('../numpy/myResTrain0601.npy')

train_set, valid_set = torch.utils.data.random_split(dataset, [50 * 3, 25 * 3])
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=1, shuffle=True)

best_loss = 999
history = []
val_history = []
output_cat = []

imagesQuery, pillIdList, labels, file = next(iter(train_loader))
print(pillIdList)

for epoch in range(num_epochs):
    print(model.embedding.weight.data)
    total_val = 0
    total_loss = 0
    imagesQuery = imagesQuery.to(device)
    #pillIdList = torch.from_numpy(np.asarray(pillIdList))
    pillIdList = pillIdList.to(device)
    labels = list(labels)
    labels = [int(x) for x in labels]
    labels = torch.LongTensor(labels)
    labels = labels.to(device)
    # init optimizer
    optimizer.zero_grad()

    # forward -> backward -> update
    outputs = model(imagesQuery, pillIdList)
    value, indices = torch.max(outputs.data, 1)
    loss = criterion(outputs, labels)
    total_loss += loss.item()
    loss.backward()
    optimizer.step()
    print(f'epoch {epoch + 1}/{num_epochs}, loss = {total_loss:.4f}')
    history.append(total_loss)

    if total_loss < best_loss:
        best_loss = total_loss
        saveModel()
        best_model_wts = copy.deepcopy(model.state_dict())
    # history.append(loss.item())


print(best_loss)

print('Finished Training')
epochs = range(0, num_epochs)
plt.plot(epochs, history, 'g', label='Training loss')
plt.title('Train loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
