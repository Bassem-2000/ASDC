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

def loading_data(token):
    api = 'http://35.178.6.126:4000/asdc-client-staff/get-all-staff'
    response = requests.get(api, headers={'Authorization':f'Bearer {token}'})
    content = response.json()['data']
    i = content[0]['asdc_client_id']
    if f'known_face_encodings_{i}' not in globals():
        
        globals()[f"known_face_encodings_{i}"] = []
        globals()[f"known_face_names_{i}"] = []
        globals()[f"qr_codes_{i}"] = []
        
    for x in content:
        qr = x['asdc_client_staff_qrcode']
        name = x['asdc_client_staff_name']
        image_url = x['asdc_client_profile_pic_url']
        if qr not in globals()[f"qr_codes_{i}"]:
            try:
                response = requests.get(image_url)              
                image = Image.open(BytesIO(response.content))
                image = np.array(image.convert("RGB"))
                locations = face_recognition.face_locations(image)
                face_encoding = face_recognition.face_encodings(image, known_face_locations=locations)[0]
                globals()[f"known_face_encodings_{i}"].append(face_encoding)
                globals()[f"known_face_names_{i}"].append(name)
                globals()[f"qr_codes_{i}"].append(qr)
            except Exception as e:
                print(f"The image doesn't contain face: {image_url}")
    return globals()[f"known_face_encodings_{i}"], globals()[f"known_face_names_{i}"], globals()[f"qr_codes_{i}"]

def classify_image(image, token):
    known_faces, known_names, qr_codes = loading_data(token)
    locations = face_recognition.face_locations(image)
    faces = face_recognition.face_encodings(image, locations)
    result_text = ""

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
            else:
                name = "Unknown"
            result_text += f"Recognized: {name}\n"
        else:
            result_text += "Fake\n"

    return result_text.strip()



# def initialize():
#     global known_faces, known_names, model
#     model = load_model()

#     image1 = face_recognition.load_image_file("2.jpg")
#     image_encode1 = face_recognition.face_encodings(image1)[0]

#     image2 = face_recognition.load_image_file("3.jpg")
#     image_encode2 = face_recognition.face_encodings(image2)[0]

#     known_faces = [image_encode1, image_encode2]
#     known_names = ['bassem', 'Soudy']

# initialize() 
