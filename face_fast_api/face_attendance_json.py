import cv2
import numpy as np
import face_recognition
import torch
import torch.nn.functional as F
from collections import OrderedDict
from Architecture import anti_spoofing
import requests
from PIL import Image
from io import BytesIO
import glob
import json


def load_model():
    global model
    device = torch.device("cpu")
    model = anti_spoofing().to(device)
    weights_path = 'model.pth'
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        name_key = key[7:]
        new_state_dict[name_key] = value
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

model = load_model()

def crop_image(org_img, x, y, width, height, s=2.7, space=0):
    h, w, c = np.shape(org_img)
    scale = min((h - 1) / height, min((w - 1) / width, s))
    new_width = width * scale
    new_height = height * scale
    center_x, center_y = width / 2 + x, height / 2 + y

    left_top_x = max(center_x - new_width / 2 - space, 0)
    left_top_y = max(center_y - new_height / 2 - space, 0)
    right_bottom_x = min(center_x + new_width / 2 + space, w - 1)
    right_bottom_y = min(center_y + new_height / 2 + space, h - 1)

    image = org_img[int(left_top_y):int(right_bottom_y) + 1,
                    int(left_top_x):int(right_bottom_x) + 1]

    image = cv2.resize(image, (80, 80))

    return image

def preprocess(img):
    face_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float()
    face_tensor = face_tensor.unsqueeze(0).to(torch.device('cpu'))
    return face_tensor

def predict(face_tensor):
    with torch.no_grad():
        output = model.forward(face_tensor)
        result = F.softmax(output, dim=-1).cpu().numpy()
    label = np.argmax(result)
    value = result[0][label]
    return label, value

def loading_data(token, asdc_client_service_id):
    api = 'http://35.178.6.126:4000/asdc-client-staff/get-all-staff-for-ai'

    response = requests.get(api, headers={'Authorization':f'Bearer {token}'},params={'service_id':asdc_client_service_id})
    content = response.json()['data']
    if content ==[]:
        return "NO_one",0,0,0
    name_file = f"_{content[0]['asdc_client_id']}_{asdc_client_service_id}"
    json_file_path = f'json/json_{name_file}.json'
    if f'json\\json_{name_file}.json' not in glob.glob("json/*.json"):
        

        known_face_encodings = []
        known_face_names = []
        qr_codes = []
        img_url = []

        data = {"known_face_encodings": known_face_encodings, "known_face_names": known_face_names,"qr_codes": qr_codes,"img_url": img_url}

        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file)
    
    with open(json_file_path, 'r') as file:
        data = json.load(file)  
    known_face_encodings = data['known_face_encodings']
    known_face_names = data['known_face_names']        
    qr_codes = data["qr_codes"]
    img_urls = data["img_url"]
        
    for x in content:
        qr = x['asdc_client_staff_qrcode']
        name = x['asdc_client_staff_name']
        image_url = x['asdc_client_profile_pic_url']

        if qr not in qr_codes:


            try:
                response = requests.get(image_url)              
                image = Image.open(BytesIO(response.content))
                image = np.array(image.convert("RGB"))
                locations = face_recognition.face_locations(image)
                face_encoding = face_recognition.face_encodings(image, known_face_locations=locations)[0]
                known_face_encodings.append(face_encoding.tolist())
                known_face_names.append(name)
                qr_codes.append(qr)
                img_urls.append(image_url)
            except Exception as e:
                print(f"The image doesn't contain face: {image_url}")

        else:
            index = qr_codes.index(qr)
            if name != known_face_names[index]:
                
                known_face_names[index] = name

            if image_url !=  img_urls[index]:
                try: 
                    img_urls[index] = image_url
                    response = requests.get(image_url)              
                    image = Image.open(BytesIO(response.content))
                    image = np.array(image.convert("RGB"))
                    locations = face_recognition.face_locations(image)
                    face_encoding = face_recognition.face_encodings(image, known_face_locations=locations)[0]
                    known_face_encodings[index] = face_encoding.tolist()
                except:
                    print(f"The image doesn't contain face: {image_url}")
               
    data = {"known_face_encodings": known_face_encodings, "known_face_names": known_face_names,"qr_codes": qr_codes,"img_url": img_urls}

    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file)



    return known_face_encodings, known_face_names, qr_codes, img_urls

def classify_image(image, token, asdc_client_service_id):
    known_faces, known_names, qr_codes, img_urls = loading_data(token,asdc_client_service_id)
    if known_faces =="NO_one":
        return "NO_one","NO_one","NO_one"
    
    locations = face_recognition.face_locations(image)
    faces = face_recognition.face_encodings(image, locations)
    result_text = ""
    QRcode, img_url = None, None

    for face, location in zip(faces, locations):
        y_min, x_max, y_max, x_min = location
        cropped_image = crop_image(image, s=2.7, x=x_min, y=y_min, width=(x_max - x_min), height=(y_max - y_min))
        face_tensor = preprocess(cropped_image)
        label, value = predict(face_tensor)

        if label == 1 and value > 0.65:
            face_distances = face_recognition.face_distance(known_faces, face)
            best_match_index = np.argmin(face_distances)
            recognition_threshold = 0.6
            if face_distances[best_match_index] < recognition_threshold:
                name = known_names[best_match_index]
                QRcode = qr_codes[best_match_index]
                img_url = img_urls[best_match_index]
            else:
                name = "Unknown"
            
        else:
            name = "Fake"

    return name,QRcode, img_url



