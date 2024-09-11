import os,cv2,sys
from PIL import Image
from data_augment import clahe
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms

train_txt=os.path.join(sys.path[0], 'Dataset', 'train.txt')
val_txt=os.path.join(sys.path[0], 'Dataset', 'val.txt')
test_txt=os.path.join(sys.path[0], 'Dataset', 'test.txt')
assert os.path.exists(train_txt),'train.txt not found'
assert os.path.exists(val_txt),'val.txt not found'
assert os.path.exists(test_txt),'test.txt not found'

class TJ_dataset(Dataset):
    def __init__(self,task_txt,config):
        super(TJ_dataset,self).__init__()

        with open(task_txt,'r',encoding='utf-8') as task_reader:
            self.task_pairs=task_reader.readlines()

        self.img_trans=transforms.Compose([
            transforms.Resize(config.img_size),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.task_pairs)
    
    def __getitem__(self,idx):
        img_path,label=self.task_pairs[idx].strip().split(' ')
        img=cv2.imread(img_path,flags=cv2.IMREAD_GRAYSCALE)
        img=clahe(img)
        img=Image.fromarray(img,mode='L')
        img=self.img_trans(img)

        return img, int(label)

class TJ_dataloader():
    def __init__(self,config) -> None:
        self.train_set=TJ_dataset(train_txt,config)
        self.val_set=TJ_dataset(val_txt,config)
        self.test_set=TJ_dataset(test_txt,config)

        self.train_loader=DataLoader(self.train_set,
                                     batch_size=config.batch_size,
                                     shuffle=True,
                                     num_workers=config.num_workers)
        self.val_loader=DataLoader(self.val_set,
                                     batch_size=config.batch_size,
                                     shuffle=False,
                                     num_workers=config.num_workers)
        self.test_loader=DataLoader(self.test_set,
                                     batch_size=config.batch_size,
                                     shuffle=False,
                                     num_workers=config.num_workers)