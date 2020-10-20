import torch
import cv2
import torch.nn as nn
import numpy as np

img = cv2.imdecode(np.fromfile("test.jpg", dtype=np.uint8), -1)

conv1 = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)
feat = conv1(img)
print(feat)


#####测试一下按列获得最大值索引的结果是否正确