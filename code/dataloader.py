from torch.utils.data import Dataset
import os 
from PIL import Image
import pandas as pd 
from sklearn.preprocessing import LabelEncoder

class Img_DataSet(Dataset):
    def __init__(self,img_dir,transform = None):
        self.img_paths,self.labels = self.get_img_path()
        self.transform = transform
        self.le = LabelEncoder()

    def __getitem__(self,index):
        path = os.path.join('data','train',self.img_paths[index])
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        encoded_label = self.label_encode()
        label = encoded_label[index]

        return img, label

    def __len__(self):
        return len(self.img_paths)

    def get_img_path(self):
        file = pd.read_table('./train.tsv',sep = "\t",usecols = ['id','expression'])
        img_paths = list(file['id'])
        labels = list(file['expression'])
        return img_paths,labels

    def label_encode(self):
        label = self.le.fit_transform(self.labels)
        return label

    def label_decode(self,label_list):
        label = self.le.inverse_transform(label_list)
        return label