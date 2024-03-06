import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
import os
import sys
import pandas as pd 
import re
import shutil

cur_path = os.getcwd()
sys.path.append(f'{cur_path}/ns_eda_prj/sentiment_analysis/KoBERT/')
# print(cur_path + "/KoBERT")

#kobert
from KoBERT.kobert.utils import get_tokenizer
from KoBERT.kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert_tokenizer import KoBERTTokenizer

#transformers
from transformers import AdamW, BertModel
from transformers.optimization import get_cosine_schedule_with_warmup




# Setting parameters
max_len = 128
batch_size = 22
warmup_ratio = 0.1
num_epochs = 20
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=3,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        return self.classifier(out)

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


def predict(model, device, tok, text_list):
    value_dict = {'중립': 0.0, '호재': 0.0, '악재': 0.0}
    logits_list = []
    test_dataloader_list = []

    if str(device) == 'cuda:0':
        model.to(device)

    for sentence in text_list:
        data = [sentence, '0']
        dataset_another = [data]
        another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
        test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)
        test_dataloader_list.append(test_dataloader)

    for test_dataloader in tqdm(test_dataloader_list):
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            logits_list.extend(out.detach().cpu().numpy())

    # # 결과 처리
    for logit in logits_list:
        if np.argmax(logit) == 0:
            value_dict['중립'] += np.max(logit) * 0.33
        elif np.argmax(logit) == 1:
            value_dict['호재'] += np.max(logit)
        elif np.argmax(logit) == 2:
            value_dict['악재'] += np.max(logit)

    max_value_rate = max(value_dict.values()) / (value_dict['중립'] + value_dict['호재'] + value_dict['악재'])
    result = {'predict': max(value_dict, key=value_dict.get), 'score': max_value_rate}
    return result

def use_gpu():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def load_model(model_path, model_name):
    load_model = torch.load(f'{model_path}{model_name}model.pt')
    load_model.load_state_dict(torch.load(f'{model_path}{model_name}weights.pt'))
    checkpoint = torch.load(f'{model_path}{model_name}model.tar')   
    load_model.load_state_dict(checkpoint['model'])
    return load_model

def load_data(path, file_name):
    df = pd.read_csv(f'{path}/{file_name}.csv')
    df['text'] = df['text'].apply(lambda x: [paragraph.strip() + '.' for paragraph in x.split('.') if paragraph.strip()])
    return df

def main():
    common_path = f'{cur_path}/ns_eda_prj/sentiment_analysis'
    model_path = f"{common_path}/models/"
    model_name = "kobert/202401230936_model/"

    data_path = f'{common_path}/../news_crawling/data'
    company = '현대차'
    csv_directory = f'{data_path}/{company}'

    # data_name = ["test_news"]

    data_name = [file[:-4] for file in os.listdir(csv_directory) if file.endswith('.csv')]
    data_name.sort()
    # print(data_name)

    used_data_path = f"{csv_directory}/used"
    try:
        os.mkdir(used_data_path)
    except FileExistsError:
        pass

    save_directory_path = f"{common_path}/data/predicted/{company}"
    try:
        os.mkdir(save_directory_path)
    except FileExistsError:
        # print(f"The directory '{save_directory_path}' already exists.")
        pass

    device = use_gpu()
    model = load_model(model_path, model_name)

    bertmodel, vocab = get_pytorch_kobert_model()
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
   

    for name in data_name:
        try:
            df = load_data(csv_directory, name)
            predict_result = []
            df['prediction'] = None
            df['score'] = None

            for idx, summary in tqdm(enumerate(df['text']), total=len(df), desc=f"Processing {name}"):
                predict_result.append(predict(model, device, tok, summary))

            # Update the entire dataframe
            for update_idx, data in enumerate(predict_result):
                df.loc[update_idx, 'prediction'] = data['predict']
                df.loc[update_idx, 'score'] = data['score']

            # Save the final dataframe
            df.to_csv(f'{save_directory_path}/predicted_{name}.csv', index=False)
            shutil.move(f'{csv_directory}/{name}.csv', used_data_path)
            print(f'{name} Predict and Save Done.')

        except Exception as e:
            print(f'Error processing {name}: {e}')

if __name__ == "__main__":
    main()