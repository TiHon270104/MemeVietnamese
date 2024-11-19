import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import re
import emoji
from transformers import ViTFeatureExtractor, ViTModel
import clip
import pickle
import os
import numpy as np
import easyocr  # Thư viện EasyOCR cho OCR
from tqdm import tqdm

# In[]: Đảm bảo mô hình chạy trên GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# In[]: Đường dẫn tới dữ liệu
rootFolder = r'C:\Users\ADMIN\Desktop\BanglaAbuseMeme'  # Đảm bảo không có dấu \ ở cuối
allInfo = os.path.join(rootFolder, 'Test_Meme', 'Data')  # Đường dẫn đầy đủ

# In[]: Đọc dữ liệu từ CSV
allData = pd.read_csv("Data_test.csv")

# In[]: Định nghĩa các hàm tiền xử lý văn bản
puncts = ["!", ",", ".", "?", ":", ";", "-", "_", "@", "#", "$", "%", "^", "&", "*", "(", ")", "[", "]", "{", "}", "|", "~", "`", "’", "’", "“", "”"]

def valid_vietnamese_letters(char):
    return char not in puncts

def get_replacement_vn(char):
    return char if valid_vietnamese_letters(char) else ' '

def get_valid_lines_vn(line):
    return ''.join(get_replacement_vn(letter) for letter in line)

def preprocess_sent_vn(sent):
    if isinstance(sent, float):  # Nếu là số thực, chuyển sang chuỗi
        sent = str(sent)
    sent = re.sub(r"http\S+", " ", get_valid_lines_vn(sent.lower()))
    sent = re.sub(r"@\S+", "@user", sent)
    sent = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "", sent)
    sent = emoji.demojize(sent)
    sent = re.sub(r"[:\*]", " ", sent)
    sent = re.sub(r"[<\*>]", " ", sent)
    sent = sent.replace("&amp;", " ").replace("\n", " ")
    return sent

# In[]: Kiểm tra định dạng ảnh hợp lệ
valid_formats = ['.jpg', '.jpeg', '.png', '.gif']

def is_valid_image(file_path):
    _, ext = os.path.splitext(file_path)
    return ext.lower() in valid_formats

# In[]: Hàm lấy ảnh và kiểm tra định dạng
def get_image(path):
    if not is_valid_image(path):
        raise ValueError(f"File {path} is not a valid image format.")
    image = Image.open(path).convert('RGB')
    return image

# In[]: Trích xuất đặc trưng từ ResNet152
def extract_resnet152_features(image_path):
    resnet152 = models.resnet152(pretrained=True)
    resnet152 = torch.nn.Sequential(*list(resnet152.children())[:-1])
    resnet152.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = transform(get_image(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet152(image)
    return features.cpu().flatten().numpy()

# In[]: Trích xuất đặc trưng từ ViT
def extract_vit_features(image_path):
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device).eval()

    inputs = feature_extractor(get_image(image_path), return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0][0].cpu().numpy()

# In[]: Trích xuất đặc trưng từ CLIP
def extract_clip_features(image_path, text):
    model, preprocess = clip.load("ViT-B/32", device=device)
    image = preprocess(get_image(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(preprocess_sent_vn(text), truncate=True).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    return {'text': text_features[0].cpu().numpy(), 'image': image_features[0].cpu().numpy()}

# In[]: Khởi tạo OCR bằng EasyOCR
reader = easyocr.Reader(['vi', 'en'])  # Chọn ngôn ngữ tiếng Việt và tiếng Anh

# In[]: Hàm trích xuất văn bản từ ảnh
def extract_text_from_image(image_path):
    image = np.array(get_image(image_path))
    result = reader.readtext(image)
    return " ".join([text[1] for text in result])

# In[]: Hàm tổng hợp và lưu kết quả vào cột 'Caption'
def process_data(allData, model_type="resnet"):
    trainValFeature = {}

    for i, caption in tqdm(zip(allData['Ids'], allData['Caption']), total=len(allData)):
        img_path = os.path.join(allInfo, i)

        if model_type == "resnet":
            features = extract_resnet152_features(img_path)
        elif model_type == "vit":
            features = extract_vit_features(img_path)
        elif model_type == "clip":
            features = extract_clip_features(img_path, caption)

        trainValFeature[i] = features

        # Trích xuất văn bản từ ảnh
        text_from_image = extract_text_from_image(img_path)
        allData.loc[allData['Ids'] == i, 'Caption'] = text_from_image

    # Lưu kết quả vào CSV mới
    allData.to_csv("Processed_Data.csv", index=False)

    # Lưu đặc trưng vào pickle
    output_filename = f"AllFeatures/{model_type}_features.p"
    with open(output_filename, 'wb') as fp:
        pickle.dump(trainValFeature, fp)

    return trainValFeature

# In[]: Gọi hàm để trích xuất đặc trưng
model_type = "clip"  # Thay đổi giữa "resnet", "vit", hoặc "clip"
features = process_data(allData, model_type)
