import torch
# import torch.nn as nn
#
# # 定義嵌入層的輸入維度和嵌入維度
# input_dim = 36  # 詞彙的總數
# embedding_dim = 300  # 嵌入向量的維度
# # 創建嵌入層
# embedding_layer = nn.Embedding(input_dim, embedding_dim)
# # 定義輸入
# input_data = torch.LongTensor([30, 29, 28])
# # 執行嵌入
# embedded_data = embedding_layer(input_data)
# # 輸出結果
# print(embedded_data)
# print(embedded_data.shape)

# a = torch.Tensor([i for i in range(10)]).reshape((2, 5))
# print(a)
# print(a.shape)
# b = a[0, :]
# print(b.shape)
#
# print(a[0, :])
# print(a[1, :])

x = torch.randn(8, 2, 4)
x.size()
z = x.view(8, -1)
print(z.size())