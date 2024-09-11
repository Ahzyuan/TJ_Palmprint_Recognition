import argparse,os,json,torch,sys
import numpy as np
import torch.nn as nn
import torch.optim as optim
from time import time,localtime
from test import test
from model import TJ_model
from helper_func import cal_acc,time_trans
from data_loader import TJ_dataloader
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', type=str, default=os.path.join(sys.path[0],'Config'))   
parser.add_argument('--dataset', type=str, default='TJ')   
config = parser.parse_args()       
with open(os.path.join(config.config_dir, 'hyp_{}.json'.format(config.dataset)), 'r') as f:
    config.__dict__ = json.load(f)    

save_dir=os.path.join(sys.path[0],'Results','saved_model')
log_dir=os.path.join(sys.path[0],'Results','train_log')
tb_log_dir=os.path.join(sys.path[0],'Results','tb_log')
for _ in [save_dir,log_dir,tb_log_dir]:
    os.makedirs(_,exist_ok=True)
struct_time=localtime()
weight_save_path=os.path.join(save_dir, 'best_{}-{}-{}T{}-{}.pth'.format(*struct_time[:5]))
task_train_log_path=os.path.join(log_dir,'train_log_{}-{}-{}T{}-{}.txt'.format(*struct_time[:5]))
config.weight = weight_save_path

dataloader = TJ_dataloader(config)

seed = config.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

epoch=config.epoch
lr=config.lr
weight_decay=config.weight_decay
momentum=config.momentum

writer=SummaryWriter(tb_log_dir)
model = TJ_model(config).to(config.device)
#model=nn.DataParallel(model,device_ids=device_ids)
#model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer  = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

torch.backends.cudnn.benchmark = True

print("\033[0;33;40mtraining...\033[0m")
time_start=time()
time_collect=torch.zeros(epoch)
for i in range(epoch):
    epoch_start_time=time()
    model.train()
    iters_num=len(dataloader.train_loader)
    loss_collect=torch.zeros(iters_num)
    
    for batch,(img,labels) in enumerate(dataloader.train_loader):
        img,labels = img.to(config.device), labels.to(config.device)
        optimizer.zero_grad()
        logit = model(img)
        loss = criterion(logit,labels)
        loss.backward()
        optimizer.step()
        print("\repoch: {}/{}, iters: {}/{}, loss: {:.3f}".format(
            str(i+1).zfill(len(str(epoch))), 
            epoch, 
            str(batch+1).zfill(len(str(iters_num))), 
            iters_num, loss),
            end='')
        loss_collect[batch]=loss
    
    print()
    top1_acc,top5_acc=cal_acc(config,dataloader.val_loader, model)
    time_end=time()
    time_collect[i]=time_end-epoch_start_time
    epoch_loss=loss_collect.mean().item()
    avg_epoch_time=time_collect.sum()/(i+1)
    writer.add_scalar('{}/epoch_loss'.format(config.dataset),epoch_loss,i)
    writer.add_scalar('{}/top1_acc'.format(config.dataset), top1_acc, i)
    writer.add_scalar('{}/top5_acc'.format(config.dataset), top5_acc, i)
    if i==0:
        best_epoch=i
        best_loss=epoch_loss
        best_acc=top1_acc
        best_acc5=top5_acc
    else:
        if top1_acc>=best_acc:
            best_epoch=i
            best_loss=epoch_loss
            best_acc=top1_acc
            best_acc5=top5_acc

            model_weight = model.state_dict()
            torch.save(model.state_dict(), weight_save_path)
    
    log_context="epoch: {}, avg_loss: {:.3f}, Top 1_acc: {:.3f}, Top 5_acc: {:.3f}\n".format(i+1,epoch_loss,top1_acc,top5_acc)
    log_context+="Average {:.1f} s/epoch | ".format(avg_epoch_time)
    log_context+="Spend: {}\n".format(time_trans(time_end-time_start))
    log_context+='Best: epoch_{}, loss_{:.3f}, Top 1_acc: {:.3f}, Top 5_acc: {:.3f}\n'.format(best_epoch+1,best_loss,best_acc,best_acc5)
    log_context+='-'*40+'\n'
    
    with open(task_train_log_path,'a',encoding='utf-8') as log_writer:
        print('\033[0;33;40m'+'-'*40+'\033[0m')
        log_writer.writelines(log_context)
        print("\033[0;33;40m{}\033[0m".format(log_context))
writer.close()

print('\033[0;33;40mFinal performance:\033[0m')
test(config, model, dataloader)

