import torch
import torch.nn as nn

output_dim = 128


class MyDBL(nn.Module):
    def __init__(self):
        super(MyDBL, self).__init__()
        self.linearOne = nn.Linear(36864, output_dim)
        self.linearTwo = nn.Linear(36864, output_dim)

    def forward(self, fx, fi):
        zx = self.linearOne(fx)
        zi = self.linearTwo(fi)
        tanh_layer = nn.Tanh()
        zx_prime = tanh_layer(zx)
        zi_prime = tanh_layer(zi)
        res = zx_prime * zi_prime
        return res


class MyMLP(nn.Module):
    def __init__(self, inputDim):
        super(MyMLP, self).__init__()
        self.linearOne = nn.Linear(inputDim, 128)
        self.linearTwo = nn.Linear(128, 64)
        self.linearThree = nn.Linear(64, 3)
        self.ReLUOne = nn.ReLU()
        self.ReLUTwo = nn.ReLU()

    def forward(self, f):
        o = self.linearOne(f)
        o = self.ReLUOne(o)
        o = self.linearTwo(o)
        o = self.ReLUTwo(o)
        o = self.linearThree(o)
        return o


class DBLANet(nn.Module):
    def __init__(self, inputDim):
        super(DBLANet, self).__init__()
        self.dblOne = MyDBL()
        self.dblTwo = MyDBL()
        self.dblThree = MyDBL()
        self.mlp = MyMLP(inputDim)
        self.embedding = nn.Embedding(36, 36864)

    def forward(self, fx, f_list):
        #f_list = f_list.squeeze(-2)
        #print(f_list.shape)
        # embedding layer output have many layers(based on classes)
        f_split = torch.split(f_list, 1, dim=1)
        f_split = torch.stack(f_split)
        f_split = f_split.squeeze(-1)
        embedding_f1 = self.embedding(f_split[0])
        embedding_f2 = self.embedding(f_split[1])
        embedding_f3 = self.embedding(f_split[2])
        f1_prime = self.dblOne(fx, embedding_f1)
        f2_prime = self.dblTwo(fx, embedding_f2)
        f3_prime = self.dblThree(fx, embedding_f3)
        final_f = torch.hstack((f1_prime, f2_prime, f3_prime))
        # print(final_f.shape)  # output_dim * 3
        res = self.mlp(final_f)
        # print(res)  # mlp output classes
        return res
