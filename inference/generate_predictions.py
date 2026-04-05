import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from lib.pvt import PolypPVT
from utils.dataloader import test_dataset
from collections import OrderedDict
from CaraNet import caranet
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Core args
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g. PolypPVT)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test datasets root')

    # Optional
    parser.add_argument('--testsize', type=int, default=352, help='Testing image size')
    parser.add_argument('--testset_names', nargs='+', type=str, default=None, help='List of dataset names (optional)')  # ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'PolypGen','all_testsets']
    parser.add_argument('--pth_path', type=str, default=None, help='Path to model weights (optional)')
    parser.add_argument('--save_path', type=str, default=None, help='Where to save predictions (optional)')
    opt = parser.parse_args()

    # -------------------------
    # Auto-resolve paths
    # -------------------------

    # Model weights
    if opt.pth_path is None:
        opt.pth_path = os.path.join("models", opt.model, "best.pth")

    # Save path
    if opt.save_path is None:
        opt.save_path = os.path.join("outputs", "predictions", opt.model)

    # Dataset names (AUTO)
    if opt.testset_names is None:
        opt.testset_names = sorted([
            d for d in os.listdir(opt.data_path)
            if os.path.isdir(os.path.join(opt.data_path, d))
        ])

        print(f"[INFO] Auto-detected datasets: {opt.testset_names}")
    
    opt = parser.parse_args()
    
    if opt.model == 'PolypPVT':
        model = PolypPVT()
        model.load_state_dict(torch.load(opt.pth_path))
    
    elif opt.model == 'CaraNet':
        model = caranet()
        weights = torch.load(opt.pth_path)

        new_state_dict = OrderedDict()

        for k, v in weights.items():
            if 'total_ops' not in k and 'total_params' not in k:
                name = k
                new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

    model.cuda()
    model.eval()
    
    for _data_name in opt.testset_names: 

        data_path = opt.data_path + _data_name
        save_path = opt.save_path + _data_name

        os.makedirs(save_path, exist_ok=True)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        
        test_loader = test_dataset(image_root, gt_root, opt.testsize)
        
        for i in range(test_loader.size):

            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            if opt.model == 'PolypPVT':
                P1,P2 = model(image)
                res = F.upsample(P1+P2, size=gt.shape, mode='bilinear', align_corners=False)
            elif opt.model == 'CaraNet':
                res5,res4,res2,res1 = model(image)
                res = res5
                res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            
            cv2.imwrite(save_path+name, res*255)
       
        print(_data_name, 'Finish!')

