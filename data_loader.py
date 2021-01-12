import torch
from torch.utils.data import Dataset
class TalkDataset(Dataset):
    def __init__(self, mode, list_of_dict):
        assert mode in ["train", "test"]
        self.mode = mode
        self.list_of_dict = list_of_dict
            
    def __getitem__(self,idx):
        inputid = self.list_of_dict[idx]['input_ids']
        tokentype = self.list_of_dict[idx]['seg']
        attentionmask = self.list_of_dict[idx]['att']
        id = self.list_of_dict[idx]['article_id']
        inputid = torch.tensor(inputid)
        tokentype = torch.tensor(tokentype)
        attentionmask = torch.tensor(attentionmask)
        if (self.mode == "test"):
            return inputid, tokentype, attentionmask, id
        elif (self.mode == "train"):
            type_label = self.list_of_dict[idx]['type_label']
            start_pos_label = self.list_of_dict[idx]['start_pos_label']
            end_pos_label = self.list_of_dict[idx]['end_pos_label']
            type_label = torch.tensor(type_label)
            start_pos = torch.tensor(start_pos_label)
            end_pos  = torch.tensor(end_pos_label)

            # start_pos = torch.softmax(start_pos.float(),dim=0)
            # end_pos = torch.softmax(end_pos.float(),dim=0)
            return inputid, tokentype, attentionmask, type_label, start_pos , end_pos
    
    def __len__(self):
        return len(self.list_of_dict)