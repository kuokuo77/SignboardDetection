import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torchvision
from torchvision import datasets, models
from torchvision.transforms import functional as FT
from torchvision import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, Dataset
import math
from PIL import Image
# import albumentations as A  # our data augmentation library
from torchvision import transforms

import matplotlib.pyplot as plt
# remove arnings (optional)
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict, deque
from tqdm import tqdm # progress bar
from torchvision.utils import draw_bounding_boxes
import xml.etree.ElementTree as ET
# Now, we will define our transforms
import sys

# parameters
img_shape = [400, 500]

# def get_transforms(train=False):
#     if train:
#         transform = A.Compose([
#             A.Resize(600, 600), # our input size can be 600px
#             A.HorizontalFlip(p=0.3),
#             A.VerticalFlip(p=0.3),
#             A.RandomBrightnessContrast(p=0.1),
#             A.ColorJitter(p=0.1),
#             ToTensorV2()
#         ], bbox_params=A.BboxParams(format='coco'))
#     else:
#         transform = A.Compose([
#             A.Resize(600, 600), # our input size can be 600px
#             ToTensorV2()
#         ], bbox_params=A.BboxParams(format='coco'))
#     return transform

def returnBbox(anno_root):
  tree = ET.parse(anno_root)
  root = tree.getroot()
  bbox_num = 0
  bbox_loc = []
  for child in root:
    if child.tag == "object":
      bbox_num += 1
      bbox_loc.append([int(child[4][0].text), int(child[4][1].text), int(child[4][2].text), int(child[4][3].text)])
  return bbox_num, bbox_loc

def split(n, train_ratio, val_ratio):
  tn = int(train_ratio*len(n))
  vn = int(val_ratio*len(n))
  ten = len(n) - tn - vn
  idxs = np.random.permutation(n)
  return idxs[0:tn], idxs[tn:tn+vn], idxs[tn+vn:]

def transformBbox(bnd_box):
  Wratio = 600/img_shape[0]
  Hratio = 600/img_shape[1]
  ratioLst = [Hratio, Wratio, Hratio, Wratio]
  bbox = []
  for box in bnd_box:
    box = [int(a * b) for a, b in zip(box, ratioLst)] 
    bbox.append(box)
  bbox = np.array(bbox)
  return(bbox)

def get_Bbox(m):
    
    m = list(filter(("[").__ne__, m))
    m = list(filter(("]").__ne__, m))
    m = list(filter((" ").__ne__, m))
    m = "".join(m)
    m = m.split(",")
    for i in range(len(m)):
        m[i] = int(m[i])
    n=4
    output=[m[i:i + n] for i in range(0, len(m), n)]
    return output

dataset_path = "/home/tonio1314/CVDL/kuokuo/signboard_dataset"

# df_dic = {
#     "index": [],
#     "img_root": [],
#     "img_name": [],
#     "anno_root": [],
#     "bbox_num": [],
#     "bbox": [],
#     "label": [],
#     "split": [],
# }
# for path in os.listdir(os.path.join(dataset_path, "image")):
#   if os.path.isfile(os.path.join(os.path.join(dataset_path, "image"), path)):
#     df_dic["img_name"].append(path)    
#     df_dic["img_root"].append(os.path.join(dataset_path, "image", path))

# for path in os.listdir(os.path.join(dataset_path, "annotation")):
#   for img_root in df_dic["img_root"]:
#     print(path.split(".")[0])
#     print(img_root.split("/")[7].split(".")[0])
#     if path.split(".")[0] == img_root.split("/")[7].split(".")[0]:
#       df_dic["anno_root"].append(os.path.join(dataset_path, "annotation", path))



# for i in range(len(df_dic["img_name"])):
#   df_dic["index"].append(i+1)
      
# for anno_path in df_dic["anno_root"]:
#   bbox_num, bbox_loc = returnBbox(anno_path)
#   df_dic["bbox"].append(bbox_loc)
#   df_dic["bbox_num"].append(bbox_num)

# train_index, val_index, test_index = split(df_dic["index"], 0.7, 0.15)
# split_index = []
# for i in range(len(df_dic["index"])):
#   split_index.append("")
# for i in range(len(train_index)):
#   for j in range(len(df_dic["index"])):
#     if df_dic["index"][j] == train_index[i]:
#       split_index[j] = "train"
# for i in range(len(val_index)):
#   for j in range(len(df_dic["index"])):
#     if df_dic["index"][j] == val_index[i]:
#       split_index[j] = "vali"
# for i in range(len(test_index)):
#   for j in range(len(df_dic["index"])):
#     if df_dic["index"][j] == test_index[i]:
#       split_index[j] = "test"
# for i in range(len(df_dic["index"])):
#   df_dic["label"].append("signboard")
# df_dic["split"] = split_index
# df = pd.DataFrame(df_dic)

df_excel_path = "/home/tonio1314/CVDL/kuokuo/signboard_detection/sb_data6.xlsx"
df = pd.read_excel(df_excel_path)

new_bbox = []
for i in range(len(df["t_bbox"])):
    new_bbox.append(get_Bbox(df["t_bbox"][i]))
df["t_bbox"] = new_bbox


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((600, 600))
])

def default_loader(path):
    return Image.open(path).convert('RGB')


class SignboardDataset(datasets.VisionDataset):
  def __init__(self, df, split = "train", transform=None, target_transform=None, loader=default_loader):
    
    self.split = split
    imgs = []
    for i in range(len(df["index"])):
      if df["split"][i] == split:
        imgs.append((df["img_root"][i], np.array(df["t_bbox"][i])))
    bbox_num = []
    for i in range(len(df["bbox_num"])):
      bbox_num.append(df["bbox_num"])
    self.bbox_num = bbox_num
    self.imgs = imgs
    self.transform = transform
    self.target_transform = target_transform
    self.loader = loader

  def __getitem__(self, index):
      fn, bbox = self.imgs[index]
      img = self.loader(fn)
      bbox_num = []
      for i in range(len(df["bbox_num"])):
        bbox_num.append(df["bbox_num"][i])
      if self.transform is not None:
        img = self.transform(img)
        
        bbox = bbox
        targ = {}
        bbox_tensor = torch.as_tensor(bbox, dtype=torch.float32)
        targ["boxes"] = torch.from_numpy(bbox)
        # targ["labels"] =  torch.tensor(np.ones((self.bbox_num[index]), dtype=np.int), dtype=torch.int64)
        # targ["labels"] =  torch.tensor([1, 1], dtype=torch.int64)
        targ["labels"] =  torch.tensor(np.ones((len(targ["boxes"])), dtype=np.int), dtype=torch.int64)
        targ["image_id"] = torch.tensor([index])
        targ["area"] = (bbox_tensor[:, 3] - bbox_tensor[:, 1]) * (bbox_tensor[:, 2] - bbox_tensor[:, 0])
        targ["iscrowd"] = torch.zeros(((len(targ["boxes"])),), dtype=torch.int64)
      return img,targ
    
  def __len__(self):
        return len(self.imgs)
    
  def __len__(self):
        return len(self.imgs)
    
  def __len__(self):
        return len(self.imgs)
    
  def __len__(self):
        return len(self.imgs)

train_dataset = SignboardDataset(df = df, split = "train", transform = transform)
val_dataset = SignboardDataset(df = df, split = "vali", transform = transform)
test_dataset = SignboardDataset(df = df, split = "test", transform = transform)


# lets load the faster rcnn model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes = 2)

model.cuda()



def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)


images,targets = next(iter(train_loader))
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]
output = model.cpu()(images, targets)

model.cuda()



# Now, and optimizer
params = [p for p in model.parameters() if p.requires_grad]
learning_rate = 0.001
momentum = 0.1
log_interval = 10
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True, weight_decay=1e-4)
# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16, 22], gamma=0.1) # lr scheduler

# save loss:
loss_classifier = []
loss_box_reg = []
loss_rpn_box_reg = []
loss_objectness = []
loss = []


def train_one_epoch(model, optimizer, loader, epoch):
    model.cuda()
    model.train()
    
#     lr_scheduler = None
#     if epoch == 0:
#         warmup_factor = 1.0 / 1000 # do lr warmup
#         warmup_iters = min(1000, len(loader) - 1)
        
#         lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor = warmup_factor, total_iters=warmup_iters)
    
    all_losses = []
    all_losses_dict = []
    
    for images, targets in tqdm(loader):
        images = list(image.cuda() for image in images)
        targets = [{k: torch.tensor(v).cuda() for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets) # the model computes the loss automatically if we pass in targets
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()
        
        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)
        
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping trainig") # train if loss becomes infinity
            print(loss_dict)
            sys.exit(1)
        
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5)
        optimizer.step()
        
#         if lr_scheduler is not None:
#             lr_scheduler.step() # 
        
    all_losses_dict = pd.DataFrame(all_losses_dict) # for printing
    print("Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
        epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
        all_losses_dict['loss_classifier'].mean(),
        all_losses_dict['loss_box_reg'].mean(),
        all_losses_dict['loss_rpn_box_reg'].mean(),
        all_losses_dict['loss_objectness'].mean()
    ))
    loss.append(np.mean(all_losses))
    loss_classifier.append(all_losses_dict["loss_classifier"].mean())
    loss_box_reg.append(all_losses_dict["loss_box_reg"].mean())
    loss_rpn_box_reg.append(all_losses_dict["loss_rpn_box_reg"].mean())
    loss_objectness.append(all_losses_dict["loss_objectness"].mean())
    
    

num_epochs=10

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_loader, epoch+1)
#     lr_scheduler.step()

model.eval()
torch.cuda.empty_cache()

img, _ = val_dataset[0]
img_int = torch.tensor(img*255, dtype=torch.uint8)
with torch.no_grad():
    prediction = model([img.cuda()])
    pred = prediction[0]

print(pred)
# fig = plt.figure(figsize=(14, 10))
# plt.imshow(draw_bounding_boxes(img_int,
#     pred['boxes'][pred['scores'] > 0.8]
#     , width=4
# ).permute(1, 2, 0))

torch.save(model, "/home/tonio1314/CVDL/kuokuo/signboard_detection/sb_dec4.pth")

fig = plt.figure()
plt.plot(loss, "-", color='orange')
plt.legend(['loss'], loc='upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig("/home/tonio1314/CVDL/kuokuo/signboard_detection/2_output/loss4.png")

fig = plt.figure()
plt.plot(loss_classifier, "-", color='red')
plt.legend(['loss_classifter'], loc='upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig("/home/tonio1314/CVDL/kuokuo/signboard_detection/2_output/loss_classifier4.png")

fig = plt.figure()
plt.plot(loss_box_reg, "-", color='skyblue')
plt.legend(['loss_box_reg'], loc='upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig("/home/tonio1314/CVDL/kuokuo/signboard_detection/2_output/loss_box_reg4.png")

fig = plt.figure()
plt.plot(loss_rpn_box_reg, "-", color='orchid')
plt.legend(['loss_rpn_box_reg'], loc='upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig("/home/tonio1314/CVDL/kuokuo/signboard_detection/2_output/loss_rpn_box_reg4.png")

fig = plt.figure()
plt.plot(loss_objectness, "-", color='seagreen')
plt.legend(['loss_objectness'], loc='upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig("/home/tonio1314/CVDL/kuokuo/signboard_detection/2_output/loss_objectness4.png")

