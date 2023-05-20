from PIL import Image
from torchvision import transforms
import torch
import glob
import numpy as np
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import euclidean

# model related
path = "../weight/capsule_accuracy_0423_append_train.pth"
model = torch.load(path)
model.eval()
img_list = []
vector_list = []


# resolve problem
# pred: 12448
# truth: 12222
# /media/wall/4TB_HDD/full_dataset/0423_dataset/test_capsule_sharpen/12222_23-0.png
# ['12448', '12222', '12083']


def get_vector(im_path):
    img = Image.open(im_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to('cuda')

    # Pass input image through feature extractor
    with torch.no_grad():
        feature_map = model.features(img_tensor)  # denseNet
        feature_map = feature_map.mean(dim=1, keepdim=True)  # 取平均值，得到單通道的特徵地圖 (1, C, H, W) =>  (1, 1, H, W)
        feature_vector = feature_map.view(1, -1)  # 將 feature_map 攤平成一個一維向量

        return feature_vector


def k_nearest_neighbors(X, query, k):
    # Calculate Euclidean distances between query and all vectors in X
    distances = [euclidean(query, x) for x in X]
    # Sort distances in ascending order and get indices of the k nearest neighbors
    indices = np.argsort(distances)[:k]

    # Return the indices of the k nearest neighbors
    return indices


def get_top3_DBL(query_pic_path, pillID):
    # image related
    image_folder = '../dataset/train/{pillID}/*.png'.format(pillID=pillID)
    for image_path in glob.glob(image_folder):
        feature_vector = get_vector(image_path)
        vector_list.append(feature_vector)
        img_list.append(image_path)

    # target image
    query_pic_vector = get_vector(query_pic_path)

    all_images_vectors = []
    for vector in vector_list:
        element = vector.cpu()
        element = element.numpy()
        all_images_vectors.append(element)

    query_image_vector = query_pic_vector.cpu().numpy()

    k = 3
    nearest_neighbors = k_nearest_neighbors(all_images_vectors, query_image_vector, k)

    # get knn avg image vector
    vectors = []
    for idx in nearest_neighbors:
        vectors.append(all_images_vectors[idx])
    avg_vector = np.mean(vectors, axis=0)
    return avg_vector


def get_label(query_path):
    pill_id = query_path.split('/')[-2]
    label = id_to_label[pill_id]
    return label


many_res = []
id_to_label = {}
denseNet_top3_predict = ['12448', '12222', '12083']
for idx in range(len(denseNet_top3_predict)):
    id_to_label[denseNet_top3_predict[idx]] = idx

print(id_to_label)

# f0 = get_top3_DBL(12448)
# f1 = get_top3_DBL(12222)
# f2 = get_top3_DBL(12083)
for pill_id in denseNet_top3_predict:
    query_pic_folder = '../dataset/train/{pill}/*.png'.format(pill=pill_id)
    for query_pic_path in glob.glob(query_pic_folder):
        fx = get_vector(query_pic_path).cpu().numpy()
        f = []
        for pill in denseNet_top3_predict:
            f.append(get_top3_DBL(query_pic_path, pill))

        f0 = f[0]
        f1 = f[1]
        f2 = f[2]
        label = get_label(query_pic_path)

        res_str = [str(fx), str(f0), str(f1), str(f2), str(label)]

        res = [fx, f0, f1, f2, label]
        print(res)
        many_res.append(res)


myRes = np.zeros(len(many_res), dtype=object)
for i in range(0, len(many_res)):
    myRes[i] = many_res[i]

np.save('../numpy/myResTrain0520', myRes)
file = np.load('../numpy/myResTrain0520.npy', allow_pickle=True)