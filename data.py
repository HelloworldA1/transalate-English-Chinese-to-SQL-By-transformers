import numpy as np
import jieba
import nltk
import torch.utils.data
# nltk.download('punkt')
from torch.utils.data import DataLoader, TensorDataset


def get_question(text):
    get1 = text.split(":")[1]
    get_quest = get1.split("|||")[0].lstrip()
    return get_quest

def get_answer(text):
    get_an = text.split(":")[1].lstrip()
    return get_an

word_list = []
tag_list = []
# with open("train.txt","r",encoding="utf-8") as f:
with open("eval_result_example.txt","r",encoding="utf-8") as f:
    i = 1
    for line in f:
        if i>2:
            i=1;
        if line != '\n':
            if i == 1:
                word_list.append(line.strip('\n'))
            else:
                tag_list.append(line.strip('\n'))
            i+=1

word_list = [get_question(i) for i in word_list]
tag_list = [get_answer(i) for i in tag_list]

# print(word_list)
# print(tag_list)

datas = [list(nltk.word_tokenize(i)) for i in word_list]
tags = [list(nltk.word_tokenize(i)) for i in tag_list]
# print(datas)
# print(tags)

#====================中文======================
# datas = [list(jieba.cut(i)) for i in word_list]
# tags = [list(nltk.word_tokenize(i)) for i in tag_list]
# # print(datas)
# # print(tags)
#=============================================

word_list = list(set([str(y) for i in word_list for y in nltk.word_tokenize(i)]))
word_dict = {word:i for i,word in enumerate(("<SOS>@<EOS>@<PAD>@"+"@".join(word_list[:])).split("@"))}
print(word_dict)
word_dict_r = [k for k, v in word_dict.items()]
print(word_dict_r)
tag_list = list(set([str(y) for i in tag_list for y in nltk.word_tokenize(i)]))
tag_dict = {word:i for i,word in enumerate(("<SOS>@<EOS>@<PAD>@"+"@".join(tag_list[:])).split("@"))}
print(tag_dict)
tag_dict_r = [k for k, v in tag_dict.items()]
print(tag_dict_r)

#=================中文定义字典======================
# word_list = list(set([str(y)  for i in word_list for y in list(jieba.cut(i))]))
# word_dict = {word:i for i,word in enumerate(("<SOS>/<EOS>/<PAD>/"+"/".join(word_list[:])).split("/"))}
# print(len(word_dict))
# word_dict_r = [k for k, v in word_dict.items()]
# # print(word_dict_r)
# tag_list = list(set([str(y) for i in tag_list for y in nltk.word_tokenize(i)]))
# # print(tag_list)
# tag_dict = {word:i for i,word in enumerate(("<SOS>/<EOS>/<PAD>/"+"/".join(tag_list[:])).split("/"))}
# # print(tag_dict)
# tag_dict_r = [k for k, v in tag_dict.items()]
# # print(tag_dict_r)
# # print(tag_dict[","])
# print(max(len(word_dict),len(tag_dict)))
#================================================

def get_item(data,tag):
    data = [['<SOS>']+item+['<EOS>']+ ['<PAD>']*50 for item in data]
    tag = [['<SOS>']+item+['<EOS>']+ ['<PAD>']*50 for item in tag]
    data = [i[:50] for i in data]
    tag = [i[:50] for i in tag]
    # print(data)
    # print(tag)
    return data,tag


class Dataset(torch.utils.data.Dataset):
    def __init__(self,datas,tags,data_dict,tag_dict):
        self.datas =datas
        self.tags = tags
        self.data_dict = data_dict
        self.tag_dict = tag_dict

    def __getitem__(self, index):
        data_nonum,tag_nonum = get_item(self.datas,self.tags)
        data_nonum = data_nonum[index]
        tag_nonum = tag_nonum[index]

        data_index = torch.LongTensor([self.data_dict[i] for i in data_nonum])
        tag_index = torch.LongTensor([self.tag_dict[i] for i in tag_nonum])
        # print("---------")
        # print(data_index)
        # print(tag_index)
        # print(data_index.shape)#torch.Size([50])
        # print(tag_index.shape)#torch.Size([50])
        return data_index,tag_index

    def __len__(self):
        assert len(self.datas) == len(self.tags)  # 每句话都有tag对应
        return len(self.tags)

loader = torch.utils.data.DataLoader(dataset = Dataset(datas,tags,word_dict,tag_dict),
                                     batch_size=8,
                                     drop_last=True,
                                     shuffle=True,
                                     collate_fn=None)
# print("--------dataloader----------")
# for batch in loader:
#     print(batch)