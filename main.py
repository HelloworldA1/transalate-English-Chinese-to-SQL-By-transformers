import torch

from data import tag_dict, loader, word_dict_r, tag_dict_r,word_dict
from mask import mask_pad, mask_tril
from model import Transformer


# 预测函数
def predict(x):
    # x = [1, 50]
    model.eval()

    # [1, 1, 50, 50]
    mask_pad_x = mask_pad(x)

    # 初始化输出,这个是固定值
    # [1, 50]
    # [[0,2,2,2...]]
    target = [tag_dict['<SOS>']] + [tag_dict['<PAD>']] * 49
    target = torch.LongTensor(target).unsqueeze(0)

    # x编码,添加位置信息
    # [1, 50] -> [1, 50, 32]
    x = model.embed(x)

    # 编码层计算,维度不变
    # [1, 50, 32] -> [1, 50, 32]
    x = model.encoder(x, mask_pad_x)

    # 遍历生成第1个词到第49个词
    for i in range(49):
        # [1, 50]
        y = target

        # [1, 1, 50, 50]
        mask_tril_y = mask_tril(y)

        # y编码,添加位置信息
        # [1, 50] -> [1, 50, 32]
        y = model.embed(y)

        # 解码层计算,维度不变
        # [1, 50, 32],[1, 50, 32] -> [1, 50, 32]
        y = model.decoder(x, y, mask_pad_x, mask_tril_y)

        # 全连接输出,39分类
        # [1, 50, 32] -> [1, 50, 39]
        out = model.fc_out(y)

        # 取出当前词的输出
        # [1, 50, 39] -> [1, 39]
        out = out[:, i, :]

        # 取出分类结果
        # [1, 39] -> [1]
        out = out.argmax(dim=1).detach()

        # 以当前词预测下一个词,填到结果中
        target[:, i + 1] = out

    return target


model = Transformer()
loss_func = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
sched = torch.optim.lr_scheduler.StepLR(optim, step_size=20, gamma=0.5)

for epoch in range(50):
    for i, (x, y) in enumerate(loader):
        # x = [8, 50]
        # y = [8, 51]
        # print(x.shape)
        # print(x)
        # print(y.shape)

        # 在训练时,是拿y的每一个字符输入,预测下一个字符,所以不需要最后一个字
        # [8, 50, 39]
        pred = model(x, y)
        # print(pred.shape)

        # [8, 50, 39] -> [400, 39]
        pred = pred.reshape(-1, max(len(word_dict),len(tag_dict)))


        # [8, 51] -> [400]
        y = y[:, :].reshape(-1)
        # print("========y.shape=========")
        # print(y.shape)

        # 忽略pad
        select = y != tag_dict['<PAD>']
        pred = pred[select]
        y = y[select]
        # print("=====pred=====")
        # print(pred)
        # print("=====y=====")
        # print(y)

        loss = loss_func(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # if i % 200 == 0:
            # [select, 39] -> [select]
        pred = pred.argmax(1)
        correct = (pred == y).sum().item()
        accuracy = correct / len(pred)
        lr = optim.param_groups[0]['lr']
        print(epoch, i, lr, loss.item(), accuracy)

    sched.step()

# 测试
for i, (x, y) in enumerate(loader):
    break

for i in range(8):
    print(i)
    print(''.join([word_dict_r[i] for i in x[i].tolist()]))
    print(''.join([tag_dict_r[i] for i in y[i].tolist()]))
    print(''.join([tag_dict_r[i] for i in predict(x[i].unsqueeze(0))[0].tolist()]))
