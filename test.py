#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import DataLoader
import numpy as np
from data_loader import TalkDataset
from model_budling import PHI_NER
from transformers import BertTokenizer
import json
import pandas as pd

type_dict = {0:"NONE", 1:"name", 2:"location", 3:"time", 4:"contact",
             5:"ID", 6:"profession", 7:"biomarker", 8:"family",
             9:"clinical_event", 10:"special_skills", 11:"unique_treatment",
             12:"account", 13:"organization", 14:"education", 15:"money",
             16:"belonging_mark", 17:"med_exam", 18:"others"}

# FILE_PATH = "./dataset/train1_test_512_bert_data.pt" ##########
FILE_PATH = './dataset/development_1_test_512_bert_data.pt'
test_file = torch.load(FILE_PATH)
# test_file = test_file[:1] ############

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print("device:", device)

PRETRAINED_LM = "hfl/chinese-bert-wwm"
# PRETRAINED_LM = './bertwwm_pretrain_aicup/'
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_LM)
# tokenizer.add_tokens(["…"])
# tokenizer.add_tokens(['痾'])
# tokenizer.add_tokens(['誒'])
# tokenizer.add_tokens(['擤'])
# tokenizer.add_tokens(['嵗'])
# tokenizer.add_tokens(['曡'])
# tokenizer.add_tokens(['厰'])
# tokenizer.add_tokens(['聼'])
# tokenizer.add_tokens(['柺'])

len(test_file)

json_file = open('./dataset/development_1.json')
data_file = json.load(json_file)

"""
type_vote
vote type of the prediction span
input: type list (list of int)
output: type(int)
"""
def type_vote(type_list):
    type_ = [0] * 19
    for i in type_list:
        type_[i] += 1
    return np.argmax(type_)

def get_origin_text(r):
    id, s_pos, e_pos, tgt, _ = r
    tgt = data_file[id]['article'][s_pos:e_pos]
    r[3] = tgt
    return r

def check_is_same_sentence(last_r_end, r_start, id):
    article = data_file[id]['article']
    ch = article[last_r_end:r_start]
    if ch == '、':
        return True,ch
    return False,ch


def decode_pos(start_pred, end_pred , type_pred, tokens):
    text = tokenizer.convert_ids_to_tokens(tokens)
    pos_pred = start_pred + end_pred


    start_pos_list = []
    ans = []
    for i in range(len(start_pred)):
        if end_pred[i] > 0:
            if len(start_pos_list) > 0:
                _type = type_vote(type_pred[start_pos_list[0] :i+1])
                ans.append([start_pos_list[0], i, _type , text[start_pos_list[0] :i+1]])
                start_pos_list = []
        if start_pred[i] > 0:
            start_pos_list.append(i)        

    return ans





def get_position(id, span, text):
    start = 0
    span = span.replace('[SEP]', '').replace('[UNK]', '').replace('[PAD]', '')
    text = text.replace('[SEP]', '').replace('[UNK]', '').replace('[PAD]', '')
    article = data_file[id]['article'].lower()
    start = article.find(span) + span.find(text)
    
    # tpos = span.find(text)
    # sep = span.find("[SEP]", 0, tpos)
    # rsep = span.rfind("[SEP]", tpos)
    # if (sep!=-1 and rsep!=-1):
    #     span = span[sep+5 : rsep]
    # elif (sep != -1):
    #     span = span[sep+5:]
    # elif (rsep != -1):
    #     span = span[:rsep]
    # article = data_file[id]['article'].lower()
    # start = article.find(span) + span.find(text)
    return start

def get_predictions(model, testLoader, BATCH_SIZE):
    result = []
    total_count = 0 # 第n筆data
    with torch.no_grad():
        for data in testLoader:
            if next(model.parameters()).is_cuda:
                data = [t.to(device) for t in data if t is not None]

            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            ids = data[-1]
            outputs = model(input_ids=tokens_tensors, 
                      token_type_ids=segments_tensors, 
                      attention_mask=masks_tensors)

            for i in range(outputs[0].shape[0]):  # run batchsize times
                type_pred = outputs[0][i] # 19*512 into class label
                start_pred = outputs[1][i] # 3*512 into class label
                end_pred = outputs[2][i]

                type_pred = torch.argmax(type_pred,dim=-1)
                start_pred = torch.argmax(start_pred,dim=-1)
                end_pred = torch.argmax(end_pred,dim=-1)


                text_token = tokens_tensors[i]
                text  = tokenizer.convert_ids_to_tokens(text_token)
                ans = decode_pos(start_pred, end_pred, type_pred,text_token)
                
                id = ids[i].item()
                for k in range(len(ans)):
                    start_pos, end_pos, _type, tgt = ans[k]
                    tgt = ''.join(tgt)
                    span = ''.join(text[start_pos - 3 : end_pos + 3])
                    tgt = tgt.replace('#', '')
                    span = span.replace('#','')
                    s_pos = get_position(id, span, tgt)
                    r = [id, s_pos, s_pos + len(tgt), tgt, type_dict[_type]]
                    r = get_origin_text(r)

                    if s_pos < 0:
                        continue

                    if _type > 0:
                        print(r)
                        if len(result) == 0:
                            result.append(r)
                            continue
                        last_r = result[-1]

                        if last_r[0] != r[0]:
                            result.append(r)
                            continue
                        # overlap ans
                        overlap_flag = False
                        if last_r[1] < r[1] and r[1] < last_r[2] and last_r[4] == r[4]:
                            overlap_flag = True
                            print('overlap ans')
                            r[1] = last_r[1]
                            r[3] = last_r[3] + r[3][-(r[2] - last_r[2]) :]
                        
                        if last_r[2] + 1 == r[1] and r[4] == last_r[4]:
                            is_same_sentence , ch = check_is_same_sentence(last_r[2], r[1], id)
                            if is_same_sentence:
                                print('same sentence')
                                overlap_flag = True
                                r[1] = last_r[1]
                                r[3] = last_r[3] + ch + r[3]
                        if not overlap_flag:
                            result.append(r)
                        else:
                            result[-1] = r
                total_count += 1
            # break
            
    return result

"""testing"""
MODEL_PATH = "./model/train_1_remove_E90.pt" ##############
# MODEL_PATH = "./model/test_E500.pt"

model = PHI_NER()
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(device)
model.eval()

BATCH_SIZE = 6
testSet = TalkDataset("test", test_file)
testLoader = DataLoader(testSet, batch_size=BATCH_SIZE)


predictions = get_predictions(model, testLoader, BATCH_SIZE)

h = ["article_id", "start_position", "end_position", "entity_text", "entity_type"]
df = pd.DataFrame(predictions,columns=h)
print(df)
df = df.drop_duplicates()
df.to_csv('./result/dev_1_remove_overlap.tsv', index=False, sep="\t")  ##########