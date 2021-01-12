import torch
from torch.utils.data import Dataset
class TalkDataset(Dataset):
    def __init__(self, mode, list_of_dict):
        assert mode in ["train", "test"]
        self.mode = mode
        self.list_of_dict = list_of_dict
            
    def __getitem__(self,idx):
        data = self.list_of_dict[idx]
        ids = data['input_ids']
        BIO = data['BIO_label']
        type_ = data['type_label']

        inputid = []
        tokentype = []
        attentionmask = []
        type_label = []
        BIO_label = []
        pos = 0
        count_back = 0
        flag = 0
        while (pos < len(ids)):
            ids_512 = ids[pos : pos+512]
            for i in range(min(511, len(ids_512)-1), 0, -1):
                if (ids_512[i] == 102): # 102 = [SEP]
                    count_back += 1
                    if (count_back == 1):
                        ids_512 = [101] + ids[pos : pos+i+1] + [0] * (512 - i - 2)
                        seg_512 = [0] * 512
                        att_512 = [1] * (i + 2) + [0] * (512 - i - 2)
                        BIO_512 = [2] + BIO[pos : pos+i+1] + [2] * (512 - i - 2)
                        type_512 = [0] + type_[pos : pos+i+1] + [0] * (512 - i - 2)
                        flag = 1 if (pos+i+1 == len(ids)) else 0

                        inputid.append(ids_512)
                        tokentype.append(seg_512)
                        attentionmask.append(att_512)
                        type_label.append(type_512)
                        BIO_label.append(BIO_512)
                        
                    elif (count_back == 3): # overlap n-1 sentences 
                        pos += i
                        count_back = 0
                        break
            if (flag): # read single talk
                break

        inputid = torch.tensor(inputid)
        tokentype = torch.tensor(tokentype)
        attentionmask = torch.tensor(attentionmask)
        if (self.mode == "test"):
            return inputid, tokentype, attentionmask
        elif (self.mode == "train"):
            type_label = torch.tensor(type_label)
            BIO_label = torch.tensor(BIO_label)
        return inputid, tokentype, attentionmask, type_label, BIO_label
    
    def __len__(self):
        return len(self.list_of_dict)