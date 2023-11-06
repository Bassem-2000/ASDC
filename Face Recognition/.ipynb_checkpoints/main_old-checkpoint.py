import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from collections import OrderedDict
import torch.nn.functional as F
import dlib
from Architecture  import anti_spoofing
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = anti_spoofing().to(device)
weights_path = 'model.pth'
state_dict = torch.load(weights_path, map_location = torch.device('cpu'))
new_state_dict = OrderedDict()
for key, value in state_dict.items():
    name_key = key[7:]
    new_state_dict[name_key] = value
model.load_state_dict(new_state_dict)
model.eval()


img = cv2.imread(r"C:\Users\user\OneDrive - Delta Academy for Science\Desktop\images.jpg")

detector = dlib.get_frontal_face_detector()
faces = detector(img)

for face in faces:
    x, y, width, height = face.left(), face.top(), face.width(), face.height()
h, w, c = np.shape(img)


scale = min((h-1)/height, min((w-1)/width, 2.7))

new_width = width * scale
new_height = height * scale
center_x, center_y = width/2+x, height/2+y

left_top_x = center_x-new_width/2
left_top_y = center_y-new_height/2
right_bottom_x = center_x+new_width/2
right_bottom_y = center_y+new_height/2

if left_top_x < 0:
    right_bottom_x -= left_top_x
    left_top_x = 0

if left_top_y < 0:
    right_bottom_y -= left_top_y
    left_top_y = 0

if right_bottom_x > w-1:
    left_top_x -= right_bottom_x-w+1
    right_bottom_x = w-1

if right_bottom_y > h-1:
    left_top_y -= right_bottom_y-h+1
    right_bottom_y = h-1


image = img[int(left_top_y): int(right_bottom_y)+1,
                          int(left_top_x): int(right_bottom_x)+1]


image = cv2.resize(image, (80, 80))


image = torch.from_numpy(image.transpose((2, 0, 1))).float()
image = image.unsqueeze(0).to(torch.device('cuda:0'))
start = time.time()
with torch.no_grad():
    output = model.forward(image)
    result = F.softmax(output, dim=-1).cpu().numpy()
end = time.time()

label = np.argmax(result)
value = result[0][label]

if label == 1:
    print("Image is Real Face. Score: {:.2f}".format(value))
    result_text = "RealFace Score: {:.2f}".format(value)
    color = (255, 0, 0)
else:
    print("Image is Fake Face. Score: {:.2f}".format(value))
    result_text = "FakeFace Score: {:.2f}".format(value)
    color = (0, 0, 255)
print(end - start)