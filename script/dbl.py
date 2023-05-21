import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import copy
from DBL import DBLANet
import matplotlib.pyplot as plt

output_dim = 24

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 50
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
        f1 = self.data[index][1]
        f2 = self.data[index][2]
        f3 = self.data[index][3]
        label = self.data[index][4]
        return fx, f1, f2, f3, label

    # return the length of our dataset
    def __len__(self):
        return len(self.data)


def saveModel():
    path = "../weight/dbl3.pth"
    torch.save(model.state_dict(), path)
    print('save')


dataset = ExampleDataset('../numpy/myResTrain0520_2.npy')

train_set, valid_set = torch.utils.data.random_split(dataset, [50 * 3, 25 * 3])
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=4, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=1, shuffle=True)

best_loss = 999
history = []
val_history = []
output_cat = []
for epoch in range(num_epochs):
    total_val = 0
    total_loss = 0
    for i, (imagesQuery, imagesOne, imagesTwo, imagesThree, labels) in enumerate(train_loader):
        imagesQuery = imagesQuery.to(device)
        imagesOne = imagesOne.to(device)
        imagesTwo = imagesTwo.to(device)
        imagesThree = imagesThree.to(device)
        labels = labels.to(device)
        # init optimizer
        optimizer.zero_grad()

        # forward -> backward -> update
        outputs = model(imagesQuery, imagesOne, imagesTwo, imagesThree)
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

    for j, (imagesQuery_val, imagesOne_val, imagesTwo_val, imagesThree_val, labels_val) in enumerate(val_loader):
        model.eval()
        imagesQuery_val = imagesQuery_val.to(device)
        imagesOne_val = imagesOne_val.to(device)
        imagesTwo_val = imagesTwo_val.to(device)
        imagesThree_val = imagesThree_val.to(device)
        labels_val = labels_val.to(device)

        # init optimizer
        optimizer.zero_grad()
        with torch.no_grad():
            outputs = model(imagesQuery_val, imagesOne_val, imagesTwo_val, imagesThree_val)
            value, indices = torch.max(outputs.data, 1)
            total_val += (indices == labels_val).sum().item()

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
