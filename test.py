import torch,argparse,os,json,sys
from time import localtime
from model import TJ_model
from helper_func import cal_acc
from data_loader import TJ_dataloader

def test(config, model=None, dataloader=None, test_log_dir=None):
    weight_path = getattr(config, "weight", None)
    assert model is not None or weight_path is not None, 'model or weight_path must be provided'

    if model is None:
        assert os.path.exists(weight_path), f'{weight_path} not exist'
        model=TJ_model(config).to(config.device)
        model_dict = model.state_dict()
        saved_dict = torch.load(weight_path, map_location=config.device) 
        model_dict.update(saved_dict)
        model.load_state_dict(model_dict)
    dataloader = dataloader if dataloader is not None else TJ_dataloader(config) 
    test_log_dir = test_log_dir if test_log_dir is not None else os.path.join(sys.path[0], 'Results', 'test_log')
    os.makedirs(test_log_dir,exist_ok=True)

    test_acc=cal_acc(config,dataloader.test_loader,model)
    struct_time=localtime()
    test_res='-'*10 + '{}-{}-{}  {}:{} {} '.format(*struct_time[:5], os.path.basename(weight_path)) + '-'*10 + '\n'
    test_res+='Top 1_acc: {:.3f}  Top 5_acc: {:.3f}\n'.format(*test_acc)
    print(test_res)
    return test_res

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='TJ') 
    parser.add_argument('-c', '--config_dir', type=str, default=os.path.join(sys.path[0],'Config'))   
    parser.add_argument('-w', '--weight', type=str, default=os.path.join(sys.path[0], 'Results', 'saved_model', 'best.pth'))
    parser.add_argument('-l', '--log_dir', type=str, default=os.path.join(sys.path[0], 'Results', 'test_log'))  
    config = parser.parse_args() 
    config.config_dir = os.path.abspath(config.config_dir)
    config.weight = os.path.abspath(config.weight)    
    config.log_dir = os.path.abspath(config.log_dir)  
    with open(os.path.join(config.config_dir, 'hyp_{}.json'.format(config.dataset)), 'r') as f:
        config.__dict__ = json.load(f)  
    
    test_res = test(config, test_log_dir=config.log_dir)

    with open(os.path.join(config.log_dir, 'model_performance.txt'),'a',encoding='utf-8') as perform_writer:
        perform_writer.write(test_res)