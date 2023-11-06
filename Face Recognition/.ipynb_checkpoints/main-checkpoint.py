import cv2
import math
import matplotlib.pyplot as plt
import torch
from collections import OrderedDict
import torch.nn.functional as F
import dlib
from Architecture  import anti_spoofing
import time
from PIL import Image
import face_recognition
import numpy as np

def create_model():
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
    return model


def known_persons():
    known_persons = [("bassem", "image/1.jpg"),("elsoudy", "image/ter.jpg")]
                     

    known_face_encodings = []
    known_face_names = []
    for name, image_path in known_persons:
        known_image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(known_image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)
    return known_face_encodings, known_face_names
    
    
def load_img(path):
    img = cv2.imread(path)
    return img
    
    
    
def face_pre(face, img,s = 2.7):

    x, y, width, height = (face[3], face[0], face[1], face[2])
    h, w, c = np.shape(img)


    scale = min((h-1)/height, min((w-1)/width, s))

    new_width = width * scale
    new_height = height * scale
    center_x, center_y = width/2+x, height/2+y

    left_top_x = max(center_x-new_width/2, 0)
    left_top_y = max(center_y-new_height/2, 0)
    right_bottom_x = min(center_x+new_width/2, w-1)
    right_bottom_y = min(center_y+new_height/2, h-1)


    image = img[int(left_top_y): int(right_bottom_y)+1,
                              int(left_top_x): int(right_bottom_x)+1]


    image = cv2.resize(image, (80, 80))
    return image, x, y, width, height   
    
    
    
    
def check_spoof(model, image):
    image = torch.from_numpy(image.transpose((2, 0, 1))).float()
    image = image.unsqueeze(0).to(torch.device('cpu'))
    with torch.no_grad():
        output = model.forward(image)
        result = F.softmax(output, dim=-1).cpu().numpy()
    label = np.argmax(result)
    value = result[0][label]    
    return label, value  
    
    
    
def face_recog(known_face_encodings, known_face_names, img, face):
    face_encodings = face_recognition.face_encodings(img, [face])

    for face_encoding in face_encodings:
        # Compare the current face encoding with known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        for i, match in enumerate(matches):
            if match:
                name = known_face_names[i]
                break
    return name 
    
    
def main(path):
    start = time.time()
    model = create_model()
    known_face_encodings, known_face_names = known_persons()
    img = load_img(path)
    face_locations = face_recognition.face_locations(img)
    for face in face_locations:
        
        image, x, y, width, height = face_pre(face, img)
        label, value = check_spoof(model, image)
        factor_font = int(width/10)
        if label == 1:
            print(f"Image is Real Face. Score: {value}")
            result_text = "RealFace Score: {:.2f}".format(value)
            color = (255, 0, 255)
            name = face_recog(known_face_encodings, known_face_names, img, face)
            print(f'Recognized: {name}, {result_text}')
            
            top, right, bottom, left = face
            cv2.rectangle(img,(left,top),(right,bottom), color,2)
            cv2.putText(img,f"{name}, {result_text}", ((left-20,top-10)), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.025*factor_font,(0,255,255), int(0.1*factor_font))
            
            
        else:
            
            print(f"Image is Fake Face. Score: {value}")
            result_text = "FakeFace Score: {:.2f}".format(value)
            color = (255, 0, 0)
            
            top, right, bottom, left = face
            cv2.rectangle(img, (left, top), (right, bottom), color, 2)
            
            cv2.putText(img, result_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.025*factor_font, color, int(0.1*factor_font))
            
        end = time.time()    
        print(end - start)
        plt.imshow(img)
            
            
main('image/1.jpg')        

        
        
    