import torch
import os 
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as opt
import model
import dataset
import torch.nn as nn

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])



#///////////////////////////////////////////////////////
num_epochs = 100
lr = 0.01
#////////////////////////////////////////////////////////
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.CNN_Model().to(device)
train_dataset = dataset.Img_DataSet('train.tsv',transform)
train_dataloader = DataLoader(train_dataset,batch_size = 32, shuffle = True)
criterion = nn.CrossEntropyLoss()
optimizer = opt.AdamW(model.parameters(),lr=lr)
#////////////////////////////////////////////////////////////
val_dataset = dataset.Img_DataSet('val.tsv')
val_dataloader = DataLoader(val_dataset,batch_size = 32, shuffle = False)






def train():
    model.train()
    train_losses = []
    train_accs = []
    best_score = 0
    for epoch in range(num_epochs):
        print("{}epoch start".format(epoch))
        running_loss = 0
        running_acc = 0
        for imgs,labels in train_dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs,labels)
            loss.backward()
            running_loss+=loss.item()
            preds = torch.argmax(outputs,dim=1)
            running_acc += torch.mean(pred.eq(labels).float())
            optimizer.step()

        running_loss /= len(train_dataloader)
        running_acc /= len(train_dataloader)
        losses.append(running_loss)
        accs.append(running_acc)

    if epoch % 10 == 0:
        score = val(model)
        if score > best_score:
            best_score = score
            model_save(model)
            prtin('モデル更新')



def val(model):
    val_running_loss = 0.0
    val_running_acc = 0.0
    for val_imgs,val_labels in val_dataloader:
        val_imgs = val_imgs.to(device)
        val_labels = val_labels.to(device)
        val_output = model(val_imgs)
        val_loss = criterion(val_output,val_labels)
        val_running_loss += val_loss.item()
        val_pred = torch.argmax(val_output,dim=1)
        val_running_acc += torch.mean(val_pred.eq(val_labels).float())
    val_running_loss /= len(validation_dataloader)
    val_running_acc /= len(validation_dataloader)
    val_losses.append(val_running_loss)
    val_accs.append(val_running_acc)
    print("epoch: {}, loss: {}, acc: {}    " \
    "val_epoch: {}, val_loss: {}, val_acc: {}".format(epoch, running_loss, running_acc, epoch, val_running_loss, val_running_acc))
    return val_running_acc

def mode_save(model):
    torch.save(model.state_dict(), 'params')

if __name__ =="__main__":
    train()