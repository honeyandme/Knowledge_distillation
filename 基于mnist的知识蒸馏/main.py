import numpy as np
import struct
import os

import torch.cuda
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset,DataLoader
def load_labels(file):  # 加载数据
    with open(file, "rb") as f:
        data = f.read()
    return np.asanyarray(bytearray(data[8:]), dtype=np.int32)


def load_images(file):  # 加载数据
    with open(file, "rb") as f:
        data = f.read()
    magic_number, num_items, rows, cols = struct.unpack(">iiii", data[:16])
    return np.asanyarray(bytearray(data[16:]), dtype=np.uint8).reshape(num_items, -1)

class MDataset(Dataset):
    def __init__(self,imgs,labs):
        self.imgs = imgs
        self.labs = labs
    def __getitem__(self, x):
        img = self.imgs[x]
        lab = self.labs[x]
        img = img.astype('float32')
        lab = lab.astype('int64')
        return img,lab
    def __len__(self):
        return len(self.imgs)

class Teacher_Model(nn.Module):
    def __init__(self):
        super(Teacher_Model, self).__init__()
        self.fc1 = nn.Linear(784,1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 10)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout()
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class Student_Model(nn.Module):
    def __init__(self):
        super(Student_Model, self).__init__()
        self.fc1 = nn.Linear(784,50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 10)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout()
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
def get_loss(pre,lab,t_lab):
    alpha = 0.5
    loss = loss_fn1(pre,lab)*alpha + loss_fn2(pre,t_lab)*(1-alpha)
    return loss
if __name__ == "__main__":
    train_imgs = load_images(os.path.join('dataset','train-images.idx3-ubyte'))/255.0
    train_labs = load_labels(os.path.join('dataset','train-labels.idx1-ubyte'))

    dev_imgs = load_images(os.path.join('dataset', 't10k-images.idx3-ubyte'))/255.0
    dev_labs = load_labels(os.path.join('dataset', 't10k-labels.idx1-ubyte'))

    train_dataset = MDataset(train_imgs,train_labs)
    train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True)

    dev_dataset = MDataset(dev_imgs, dev_labs)
    dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=True)

    epoch = 5
    lr = 5e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"


    #---------------Teacher model train----------------------
    teacher_model = Teacher_Model().to(device)
    opt = torch.optim.Adam(teacher_model.parameters(),lr = lr)
    loss_fn = nn.CrossEntropyLoss()
    for e in range(epoch):
        teacher_model.train()
        for batch_imgs,batch_labs in tqdm(train_dataloader):
            batch_imgs = batch_imgs.to(device)
            batch_labs = batch_labs.to(device)


            pre = teacher_model(batch_imgs)
            loss = loss_fn(pre,batch_labs)
            loss.backward()
            opt.step()
            opt.zero_grad()
        print(loss)
        teacher_model.eval()
        acc = 0
        for batch_imgs,batch_labs in tqdm(dev_dataloader):
            batch_imgs = batch_imgs.to(device)
            batch_labs = batch_labs.to(device)
            pre = teacher_model(batch_imgs)
            pre = torch.argmax(pre,dim=-1)
            acc += int(torch.sum(pre==batch_labs))
        print(f"{e},teacher acc:{acc/len(dev_dataset):.4f}")

    # ---------------Student model train----------------------

    student_model = Student_Model().to(device)
    opt = torch.optim.Adam(student_model.parameters(), lr=lr)
    loss_fn1 = nn.CrossEntropyLoss()
    loss_fn2 = nn.KLDivLoss(reduction='batchmean')
    softmax = nn.Softmax(dim=-1)
    for e in range(epoch):
        student_model.train()
        for batch_imgs, batch_labs in tqdm(train_dataloader):
            batch_imgs = batch_imgs.to(device)
            batch_labs = batch_labs.to(device)

            pre = student_model(batch_imgs)
            loss = get_loss(pre,batch_labs,softmax(teacher_model(batch_imgs)))
            loss.backward()
            opt.step()
            opt.zero_grad()
        print(loss)
        student_model.eval()
        acc = 0
        for batch_imgs, batch_labs in tqdm(dev_dataloader):
            batch_imgs = batch_imgs.to(device)
            batch_labs = batch_labs.to(device)
            pre = student_model(batch_imgs)
            pre = torch.argmax(pre, dim=-1)
            acc += int(torch.sum(pre == batch_labs))
        print(f"{e},Student acc:{acc / len(dev_dataset):.4f}")