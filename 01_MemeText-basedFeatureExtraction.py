from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, XLMRobertaTokenizer, XLMRobertaModel
import torch
import pickle
from tqdm import tqdm
import pandas as pd
import emoji
import re
from normalizer import normalize

# Kiểm tra và sử dụng GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Đọc dữ liệu
allData = pd.read_csv("Processed_Data.csv")

# Danh sách các ký tự đặc biệt
puncts = [">", "+", ":", ";", "*", "’", "_", "●", "■", "•", "-", ".", "''", "``", "'", "|", "​", "!", ",", "@", "?", "\u200d", "#", "(", ")", "|", "%", "।", "=", "``", "&", "[", "]", "/", "'", "”", "‘", "‘", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Các hàm xử lý văn bản
def kiem_tra_ky_tu_viet(char):
    return char not in puncts

def get_replacement(char):
    if kiem_tra_ky_tu_viet(char):
        return char
    newlines = [10, 2404, 2405, 2551, 9576]
    return ' ' if ord(char) in newlines else ' '

def xu_ly_dong(dong):
    return ''.join(get_replacement(letter) for letter in dong)

def re_sub(pattern, repl, text):
    return re.sub(pattern, repl, text)

def tien_ich_xu_ly(sent):
    sent = re.sub(r"http\S+", " ", xu_ly_dong(sent.lower()))
    sent = re.sub(r"@\S+", "@user", sent)
    sent = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "", sent)
    sent = emoji.demojize(sent)
    sent = re_sub(r"[:\*]", " ", sent)
    sent = re.sub(r"[<\*>]", " ", sent)
    sent = sent.replace("&amp;", " ").replace("\n", " ").replace("ðŸ¤§", " ").replace("ðŸ˜¡", " ")
    return sent

# Chuẩn hóa cột Caption
allData['Caption'] = allData['Caption'].apply(lambda x: tien_ich_xu_ly(x))
allData['Caption'] = allData['Caption'].apply(lambda x: normalize(x))  # Chuẩn hóa tiếng Việt

# Hàm trích xuất đặc trưng
def extract_features(model_name, tokenizer_cls, model_cls, data, output_file):
    print(f"Extracting features using {model_name}...")
    tokenizer = tokenizer_cls.from_pretrained(model_name)
    model = model_cls.from_pretrained(model_name).to(device)
    
    embeddings = {}
    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        sentence = row['Caption']
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state
            embeddings[row['Ids']] = outputs[0][0].cpu().numpy()
    
    # Lưu đặc trưng
    with open(output_file, "wb") as fp:
        pickle.dump(embeddings, fp)
    print(f"Features saved to {output_file}")

# Trích xuất đặc trưng từ các mô hình
# PhoBERT
extract_features("vinai/phobert-base", AutoTokenizer, AutoModel, allData, "PhoBERTEmbedding.p")

# m-BERT
extract_features("bert-base-multilingual-cased", BertTokenizer, BertModel, allData, "mBERTEmbedding_vn_memes.p")

# MuRIL
extract_features("google/muril-base-cased", AutoTokenizer, BertModel, allData, "MuRILEmbedding_vn_memes.p")

# XLM-Roberta
extract_features("xlm-roberta-base", XLMRobertaTokenizer, XLMRobertaModel, allData, "xlmEmbedding_vn_memes.p")
