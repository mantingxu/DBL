import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as Fun
import copy
from DBL import DBLANet
import matplotlib.pyplot as plt

output_dim = 24

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 3000
batch_size = 64
learning_rate = 0.001
inputDim = output_dim * 3

model = DBLANet(inputDim).to(device)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.02)

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


def saveModel():
    path = "../weight/dbl2.pth"
    # torch.save(model.state_dict(), path)
    torch.save(model.state_dict(), path)


dataset = ExampleDataset('../numpy/myResTrain0520_1.npy')

train_set, valid_set = torch.utils.data.random_split(dataset, [50 * 3, 25 * 3])

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=1, shuffle=True)

best_loss = 999
history = []
val_history = []
for epoch in range(num_epochs):
    total = 0
    total_val = 0
    for i, (imagesQuery, imagesOne, imagesTwo, imagesThree, labels) in enumerate(train_loader):
        imagesQuery = imagesQuery.to(device)
        imagesOne = imagesOne.to(device)
        imagesTwo = imagesTwo.to(device)
        imagesThree = imagesThree.to(device)
        labels = Fun.one_hot(labels, num_classes=3)
        labels = labels.squeeze(1)
        labels = labels.to(torch.int64)
        labels = labels.to(device)

        # init optimizer
        optimizer.zero_grad()

        # forward -> backward -> update
        outputs = model(imagesQuery, imagesOne, imagesTwo, imagesThree)
        # print(torch.max(outputs.data, 1))
        value, indices = torch.max(outputs.data, 1)
        value_label, indices_label = torch.max(labels.data, 1)
        for idx in range(len(indices)):
            if indices[idx].item() == indices_label[idx].item():
                total += 1
        outputs.argmax()
        loss = criterion(outputs, torch.max(labels, 1)[1])
        loss.backward()
        optimizer.step()
    acc = total / train_set.__len__()
    print('acc: ', acc)
    print(f'epoch {epoch + 1}/{num_epochs}, loss = {loss.item():.4f}')
    history.append(loss.item())

    for j, (imagesQuery_val, imagesOne_val, imagesTwo_val, imagesThree_val, labels_val) in enumerate(val_loader):
        model.eval()
        imagesQuery_val = imagesQuery_val.to(device)
        imagesOne_val = imagesOne_val.to(device)
        imagesTwo_val = imagesTwo_val.to(device)
        imagesThree_val = imagesThree_val.to(device)
        labels_val = Fun.one_hot(labels_val, num_classes=3)
        labels_val = labels_val.squeeze(1)
        labels_val = labels_val.to(torch.int64)
        labels_val = labels_val.to(device)

        # init optimizer
        optimizer.zero_grad()
        with torch.no_grad():
            outputs = model(imagesQuery_val, imagesOne_val, imagesTwo_val, imagesThree_val)
            value, indices = torch.max(outputs.data, 1)
            value_label, indices_label = torch.max(labels_val.data, 1)
            for idx in range(len(indices)):
                if indices[idx].item() == indices_label[idx].item():
                    total_val += 1
            outputs.argmax()
            val_loss = criterion(outputs, torch.max(labels_val, 1)[1])

    acc_val = total_val / valid_set.__len__()
    print('val acc: ', acc_val)
    print(f'val epoch {epoch + 1}/{num_epochs}, loss = {val_loss.item():.4f}')
    val_history.append(val_loss.item())

    if val_loss.item() < best_loss:
        best_loss = val_loss.item()
        saveModel()
        best_model_wts = copy.deepcopy(model.state_dict())
    # if loss.item() < best_loss:
    #     best_loss = loss.item()
    #     saveModel()
    #     best_model_wts = copy.deepcopy(model.state_dict())

print(best_loss)

print('Finished Training')

epochs = range(0, num_epochs)
plt.plot(epochs, history, 'g', label='Training loss')
plt.title('Train loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, val_history, 'r', label='validation loss')
plt.title('Val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
