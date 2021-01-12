#!/usr/bin/env python
# coding: utf-8

import torch
from transformers import AdamW
from data_loader import TalkDataset
from model_budling import PHI_NER
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BATCH_SIZE = 8
# data_path = "./dataset/train_1_normalize_train_512_bert_data.pt"
data_path = './dataset/train1_remove_train_512_bert_data.pt'
list_of_dict = torch.load(data_path)

print(list_of_dict[0].keys())


""" model setting (training)"""
trainSet = TalkDataset("train", list_of_dict)
trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print("device:", device)
model = PHI_NER()
# optimizer = AdamW(model.parameters(), lr=1e-5) # AdamW = BertAdam
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# BIO_weight = torch.FloatTensor([98.33333333, 53.5694687,   1.        ]).cuda()
type_weight = torch.FloatTensor([1.00000000e+00, 4.79372641e+02, 5.39595000e+02, 4.83475676e+01,
4.38265956e+03, 1.13018437e+04, 4.99416203e+03, 9.71971425e+07,
3.90457045e+03, 2.68375880e+04, 7.15339819e+04, 3.97687436e+03,
9.71971425e+07, 1.07261502e+05, 3.57801575e+04, 1.07378815e+03,
9.71971425e+07, 4.60856189e+02, 9.71971425e+07,]).to(device)

# BIO_loss_fct = nn.CrossEntropyLoss(weight=BIO_weight)

pos_weight = torch.FloatTensor([1.0,100.0]).to(device)


kl_loss = nn.KLDivLoss(log_target=False,size_average = False)

type_loss_fct = nn.CrossEntropyLoss(weight=type_weight)
pos_loss_fct = nn.CrossEntropyLoss(weight=pos_weight)


# high-level 顯示此模型裡的 modules
print("""
name            module
----------------------""")
for name, module in model.named_children():
    if name == "bert":
        for n, _ in module.named_children():
            print(f"{name}:{n}")
#             print(_)
    else:
        print("{:15} {}".format(name, module))

""" training """
from datetime import datetime,timezone,timedelta

model = model.to(device)
model.train()

EPOCHS = 100
dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
print(dt2)
for epoch in range(EPOCHS):
    running_loss = 0.0
    type_running_loss = 0.0
    start_running_loss = 0.0
    end_running_loss = 0.0
    for data in trainLoader:        
        tokens_tensors, segments_tensors, masks_tensors,    \
        type_label, start_pos_label,end_pos_label = [t.to(device) for t in data]

        # 將參數梯度歸零
        optimizer.zero_grad()

        # forward pass
        outputs = model(input_ids=tokens_tensors, 
                      token_type_ids=segments_tensors, 
                      attention_mask=masks_tensors)

        type_pred = outputs[0]
        type_pred = torch.transpose(type_pred, 1, 2)
        type_running_loss = type_loss_fct(type_pred, type_label)

        start_pred = outputs[1]
        start_pred = torch.transpose(start_pred, 1, 2)
        start_loss = pos_loss_fct(start_pred,start_pos_label)
        
        end_pred = outputs[2]
        end_pred = torch.transpose(end_pred, 1, 2)
        end_loss = pos_loss_fct(end_pred, end_pos_label)
        
        # print(start_loss)
        # print(end_loss)


        # # KL divergence need log softmax
        # start_pred = torch.nn.functional.log_softmax(start_pred,dim=1)
        # end_pred = torch.nn.functional.log_softmax(end_pred,dim=1)
        # start_loss = kl_loss(start_pred,start_pos_label)
        # end_loss = kl_loss(end_pred, end_pos_label)


        loss = start_loss + end_loss + type_running_loss

        # backward
        loss.backward()
        optimizer.step()

        # 紀錄當前 batch loss
        running_loss += loss.item()
        type_running_loss += type_running_loss.item()
        start_running_loss += start_loss.item()
        end_running_loss += end_loss.item()


    if ((epoch + 1) % 10 == 0): #####
        CHECKPOINT_NAME = './model/train_1_remove_E' + str(epoch + 1) + '.pt' ########################
        torch.save(model.state_dict(), CHECKPOINT_NAME)

        dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
        dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
        print('%s\t[epoch %d] loss: %.3f, type_loss: %.3f, start_loss: %.3f , end_loss: %.3f' %
              (dt2, epoch + 1, running_loss, type_running_loss, start_running_loss ,end_running_loss ))