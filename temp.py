import os
import json
import yaml
from tqdm import tqdm

# predefined object list
object_list = []
object_list += [v.lower().replace(' ', '_') for k, v in yaml.load(open('OpenImagesV7.yaml'), Loader=yaml.FullLoader)['names'].items()]
#object_list += [v.lower().replace(' ', '_').split('/')[0] for k, v in yaml.load(open('lvis.yaml'), Loader=yaml.FullLoader)['names'].items()]
#object_list += [v.lower().replace(' ', '_').split('/')[0] for k, v in yaml.load(open('Objects365.yaml'), Loader=yaml.FullLoader)['names'].items()]
#object_list += [v.lower().replace(' ', '_').split('/')[0] for k, v in yaml.load(open('coco8.yaml'), Loader=yaml.FullLoader)['names'].items()]
#object_list += [v.lower().replace(' ', '_').split('/')[0] for k, v in yaml.load(open('coco8-seg.yaml'), Loader=yaml.FullLoader)['names'].items()]
#object_list += [v.lower().replace(' ', '_').split('/')[0] for k, v in yaml.load(open('coco.yaml'), Loader=yaml.FullLoader)['names'].items()]
#object_list += ['bottle', 'bowl', 'drawer', 'cup', 'door', 'fork', 'knife', 'cupboard', 'scissors', 'wine glass', 'banana', 'bread machine', 'tap', 'pot', 'lid', 'plate', 'fridge', 'bag', 'trash bin', 'kettle', 'oven', 'pizza', 'mug', 'cucumber', 'peeler', 'mouse', 'rice cooker', 'bag', 'spatula', 'slicer', 'computer keyboard', 'phone', 'container', 'hob', 'heater', 'onion', 'tray', 'melon', 'coffee maker', 'remote', 'dishwasher', 'spoon', 'processor', 'sponge', 'package', 'dough', 'meat', 'cheese', 'blender', 'button', 'tomato']

# GPT-4o result
gpt4o_predictions = json.load(open('Seen_result_dict.json', 'r'))
pairs = [[o, gpt4o_predictions[a][o][p][1], gpt4o_predictions[a][o][p][2]] for a in gpt4o_predictions for o in gpt4o_predictions[a] for p in gpt4o_predictions[a][o]]


import torch
import clip
from PIL import Image
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

text_k = clip.tokenize(object_list).to(device)
with torch.no_grad():
    text_features_k = model.encode_text(text_k)
print(text_features_k.shape)

total, correct = 0, 0
correct_classes = []
incorrect_classes = []
for pair in tqdm(pairs):
    text_q = clip.tokenize([pair[1]]).to(device)
    with torch.no_grad():
        text_features_q = model.encode_text(text_q)
    pred = object_list[F.cosine_similarity(text_features_q, text_features_k).argmax()]
    gt = pair[0]
    
    total += 1
    correct += (pred == gt) #or (pair[1] not in object_list)
    if pred != gt:
        print(gt, pred)
    if pred == gt:
        correct_classes.append(gt)
    else:
        incorrect_classes.append(pred)

#print(list(set(correct_classes)))
print(list(set(incorrect_classes)))




print(f'Acc: {correct}/{total} = {correct/total*100}%')
# Open Image Dataset: Acc: 417/669 = 62.33183856502242%
# LVIS: Acc: 410/669 = 61.285500747384155%
# Objects365: Acc: 501/669 = 74.88789237668162%
# COCO8: Acc: 497/669 = 74.2899850523169%
# COCO: Acc: 497/669 = 74.2899850523169%
# Robo-ABC: Acc: 137/669 = 20.47832585949178%