from PIL import Image
from torchvision import transforms
import torch
import glob
import numpy as np
import json

# model related
path = "../weight/capsule_accuracy_0603.pth"
model = torch.load(path)
model.eval()
img_list = []
vector_list = []

try:
    json_file = open('../label/capsule_class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)


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
        feature_vector = feature_map.view(-1)  # 將 feature_map 攤平成一個一維向量
        return feature_vector


def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None


def get_label(query_path):
    pill_id = query_path.split('/')[-1].split('_')[0]
    key = get_key_from_value(class_indict, pill_id)
    # pred = class_indict[str(predict_cla)]
    # label = id_to_label[pill_id]
    return key


many_res = []
#id_to_label = {}
denseNet_top3_predict = ['12448', '12222', '12083', '325', '2311', '2321', '4061', '4115', '6356']
#for idx in range(len(denseNet_top3_predict)):
#    id_to_label[denseNet_top3_predict[idx]] = idx


query_pic_folder = '../dataset/aug_test/*.png'
count = 0
for query_pic_path in glob.glob(query_pic_folder):
    count += 1
    file = query_pic_path.split('/')[-1]
    label = get_label(query_pic_path)
    fx = get_vector(query_pic_path).cpu().numpy()
    f = []
    query_img = Image.open(query_pic_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(query_img).unsqueeze(0).to('cuda')
    denseNet_result = torch.squeeze(model(img_tensor))
    predict = torch.softmax(denseNet_result, dim=0)
    top5_prob, top5_id = torch.topk(predict, 3)
    top5_id = top5_id.cpu().numpy()
    ids = []
    for id in top5_id:
        ids.append(id)
    ids.sort()
    if int(label) not in ids:
        print(ids, label)
    if int(label) in ids:
        res = [fx, ids, label, file]
        many_res.append(res)

myRes = np.zeros(len(many_res), dtype=object)
print(count)
print(len(many_res))
for i in range(0, len(many_res)):
    myRes[i] = many_res[i]

np.save('../numpy/myResTestAug', myRes)

