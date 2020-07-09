from dataloader.dataset import *
from utils.utility import *
from utils.resume import *
from models.model import *
from loss.seg_loss import *
from torch.autograd import Variable
from scipy import ndimage
import copy
import cv2

from argparse import *
import yaml

def write_seq_images(opt, input_imgs, input_mask, predicted, idx,key):
    def _func(x):
        return (x.numpy()*255).astype(np.uint8).transpose(1,2,0).squeeze()

    base_num = idx * input_imgs.shape[0]

    mask_refine = opt.writer.refiner.bin(predicted)
    
    input_imgs = scale_img_back(input_imgs, device=opt.gpu)
    input_imgs, input_mask, predicted = input_imgs.cpu(), input_mask.cpu(), predicted.cpu()
    mask_refine = mask_refine.cpu()

    mask_raw = copy.deepcopy(predicted)
    mask_raw[mask_raw >= 0.5] = 1
    mask_raw[mask_raw < 0.5] = 0

    #refine_bin = copy.deepcopy(refine)

    mask_out = input_imgs * mask_refine 

    for i, (img, mask, out, maskout,raw,refine) in enumerate(zip(input_imgs, input_mask, predicted,mask_out,mask_raw,mask_refine)):
        img = cv2.cvtColor(_func(img), cv2.COLOR_RGB2BGR)
        maskout = cv2.cvtColor(_func(maskout), cv2.COLOR_RGB2BGR)
        mask = _func(mask)
        out = _func(out)
        raw = _func(raw)
        refine = _func(refine)


        dump_folder = os.path.join(opt.dump_folder,str(base_num+i))

        if not os.path.exists(dump_folder):
            os.mkdir(dump_folder)

        cv2.imwrite(join(dump_folder, key+"_img_%04d"%(base_num+i)+ ".png"), img)
        cv2.imwrite(join(dump_folder, key+"_mask_%04d"%(base_num+i)+ ".png"), mask)
        cv2.imwrite(join(dump_folder, key+"_pred_%04d"%(base_num+i)+ ".png"), out)
        cv2.imwrite(join(dump_folder, key+"_raw_%04d"%(base_num+i)+ ".png"), raw)
        cv2.imwrite(join(dump_folder, key+"_refine_%04d"%(base_num+i)+ ".png"), refine)
        cv2.imwrite(join(dump_folder, key+"_maskout_%04d"%(base_num+i)+ ".png"), maskout)


        opt.writer.update_metric(**opt.writer.evaluator(input_mask, mask_refine))


            
def validate(opt,key):
    opt.G.eval()
    opt.writer.reset()

    with torch.no_grad():
        for idx, (input_imgs, input_mask) in enumerate(opt.val_loader):
            #print(idx)
            input_imgs = input_imgs.cuda(opt.gpu)
            predicted = opt.G(input_imgs)
            write_seq_images(opt, input_imgs, input_mask, predicted, idx,key)

        opt.writer.dump_metric("Val Metric",1)

def folder_init(opt):
    ''' tensorboard initialize: create tensorboard filename based on time tick and hyper-parameters

        Args:
            opt: parsed options from cmd or .yml(in config/ folder)
            
        Returns:
            opt: add opt.dump_folder and return opt
        
    '''    
    #opt.dump_folder = os.path.join(opt.dump_folder, opt.experiment_name)
    
    #if not os.path.exists(opt.dump_folder):
    #    os.makedirs(opt.dump_folder)
    return opt

def init_settings(opt, data_path):
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(1124)
    opt.gpu = torch.device(opt.device_id)
    opt.dice_criterion = DiceLoss().cuda(opt.gpu)
    opt.bce_criterion = nn.BCEWithLogitsLoss().cuda(opt.gpu)
    opt.self_op = SelfOperation(opt)

    opt = init_models(opt)
    opt = init_dataset(opt, data_path)
    opt = folder_init(opt)
    opt.writer = TensorWriter(opt)#SingleSummary(opt)
    
    opt.ones = torch.ones(opt.batch_size).cuda(opt.gpu)
    opt.zeros = torch.zeros(opt.batch_size).cuda(opt.gpu)

    
    return opt

def init_models(opt):
    # model init
    generator = ModelFactory(opt)
    generator.register_hook_model(get_generator_model)
    generator.register_hook_optimizer(get_ralamb_optimizer)

    opt.G, opt.optim_G, opt.sche_G = generator()

    return opt


def init_dataset(opt, data_path):

    opt.portrait_val = SegmentationDataset(opt, split="test", folder=data_path)
    opt.val_loader =  DataLoader(opt.portrait_val, collate_fn=segmentation_collate_fn, batch_size=opt.batch_size, shuffle=False, pin_memory=True,num_workers=4)
    return opt

## option args
def parse_opts():
    parser = ArgumentParser(description="segmentation")
    #parser.add_argument('--common_setting',default='config/common_config.yml',type=str,help="fixed parameters")
    parser.add_argument('--hyper_setting', default='config/dynamic_config.yml',type=str,help="hyper-parameters of experiments")
    parser.add_argument('--device_id', default=2, type=int,help="CUDA device.")
    #parser.add_argument('--resume', default='checkpt.pt',type=str,help="restore checkpoint")
    parser.add_argument('--comments', default="mask_together", type=str)
    
    opt =  parser.parse_args()

    #model_opt = yaml.safe_load(open(opt.common_setting,"r"))
    hyper_opt = yaml.safe_load(open(opt.hyper_setting,"r"))
    opt = Namespace(**hyper_opt,**vars(opt))
    #opt = Namespace(**model_opt, **hyper_opt,**vars(opt))
    return opt

def main():
    
    pretrain_model_dir = '/home/clc/loginfo/'

    dump_dir = "/home/clc/dump_results/pic_saved/"
    key_list = ["fwiou","miou","mpa","pa"]  

    opt = parse_opts()
    
    opt.current_folder = os.getcwd()
    
    suffix = "_crop_size_128_batch_size_64_epochs_10000"
    

    experiment_mapp = {
        'Flowers102_10': {'name':'2020_06_11_12_53',
                 'data':'/home/clc/data/segmentation_data/Flowers102/'},
    }


    opt = parse_opts()

    for k, item in experiment_mapp.items():

        experiment_name = item['name']
        data_path = item['data']

        experiment_name = experiment_name + suffix
        opt.experiment_name = experiment_name
        
        print("method-->",k)

        for key in key_list:  
            
            opt.dump_folder = join(dump_dir,k, key)

            if not os.path.exists(opt.dump_folder):
                os.makedirs(opt.dump_folder)

            opt = init_settings(opt,data_path)

            opt.G = Resume(pretrain_model_dir,experiment_name).resume_model(opt.G, model_path=None, key=key, state=False)
            opt.G.cuda(opt.gpu)
            validate(opt,key)


    opt.writer.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("ctrl + c ")
