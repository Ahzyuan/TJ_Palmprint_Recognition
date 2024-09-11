import os,shutil,random,sys,argparse
from tqdm import tqdm
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--origin_set', type=str, default=os.path.join(sys.path[0],'Dataset','ROI'),help='Path to the roi dataset')   
parser.add_argument('-d', '--destination', type=str, default=os.path.join(sys.path[0],'Dataset'),help='Path to store the final dataset')
args = parser.parse_args() 
args.origin_set = os.path.abspath(args.origin_set)
args.destination = os.path.abspath(args.destination)    
assert os.path.exists(args.origin_set), f'The path of dataset {args.origin_set} doesn\'t exist'
assert os.path.exists(args.destination), f'The path of destination {args.destination} doesn\'t exist'

root=args.origin_set 
des=args.destination
train_rate=0.6
val_rate=0.2
test_rate=0.2
assert train_rate+val_rate+test_rate==1,'The sum of dividing rates isn\'t equal to 1' 

if os.path.exists(des):
    shutil.rmtree(des)
os.mkdir(des)

ses1_path=os.path.join(root,'session1')
ses2_path=os.path.join(root,'session2')
ses1_list=glob(ses1_path+'\\*.bmp')
ses2_list=glob(ses2_path+'\\*.bmp')

for palm_cls in tqdm(range(0,600)):
    cls_dir=os.path.join(des,str(palm_cls))
    os.mkdir(cls_dir)

    pick_list=ses1_list[:10]+ses2_list[:10]
    ses1_list=ses1_list[10:]
    ses2_list=ses2_list[10:]

    rename_s1=lambda img_path: os.path.join(cls_dir,'s1_'+os.path.split(img_path)[-1])
    rename_s2=lambda img_path: os.path.join(cls_dir,'s2_'+os.path.split(img_path)[-1])
    target_list=list(map(rename_s1,pick_list[:10]))+list(map(rename_s2,pick_list[10:]))

    list(map(lambda src,des:shutil.copyfile(src,des),pick_list,target_list))

    cls_img_num=len(target_list)
    train_num=int(cls_img_num*train_rate)
    val_num=int(cls_img_num*val_rate)
    test_num=int(cls_img_num*test_rate)
    random.shuffle(target_list)

    train_list=target_list[:train_num]
    val_list=target_list[train_num:val_num+train_num]
    test_list=target_list[val_num+train_num:]

    with open(os.path.join(des,'train.txt'),'a',encoding='utf-8') as train_writer:
        train_writer.writelines((' '+str(palm_cls)+'\n').join(train_list)+' '+str(palm_cls)+'\n')
    
    with open(os.path.join(des,'val.txt'),'a',encoding='utf-8') as val_writer:
        val_writer.writelines((' '+str(palm_cls)+'\n').join(val_list)+' '+str(palm_cls)+'\n')
    
    with open(os.path.join(des,'test.txt'),'a',encoding='utf-8') as test_writer:
        test_writer.writelines((' '+str(palm_cls)+'\n').join(test_list)+' '+str(palm_cls)+'\n')