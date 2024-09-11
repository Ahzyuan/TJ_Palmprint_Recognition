import torch
from tqdm import tqdm

def time_trans(time):
    h=time//3600
    min=(time-h*3600)//60
    s=time-h*3600-min*60
    return "{:.0f} h {:.0f} mins {:.1f}s".format(h,min,s)

def cal_acc(config,dataloader,model):
    model.eval()
    iter_num=len(dataloader)
    top1_iter_acc=torch.zeros(iter_num)
    top5_iter_acc=torch.zeros(iter_num)
    with torch.no_grad():
        for iter,(img,labels) in tqdm(enumerate(dataloader),desc='Eval'):
            img,labels = img.to(config.device), labels.to(config.device)
            logit = model(img)  # 16,600
            predict_labels=torch.argsort(logit,dim=1,descending=True)
            top1_iter_acc[iter]=sum(predict_labels[:,0]==labels)/len(labels)
            top5_iter_acc[iter]=sum(list(
                                    map(lambda true_label,pred_label: true_label in pred_label,
                                        labels,predict_labels[:,:5]
                                        )))/len(labels)
    top1_acc=top1_iter_acc.mean().item()
    top5_acc=top5_iter_acc.mean().item()
    return top1_acc,top5_acc


