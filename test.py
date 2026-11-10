import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger
import torch.nn as nn

def GAM__vis(model,test_transform):
    from PIL import Image
    from pytorch_grad_cam import GradCAMPlusPlus, GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image,preprocess_image
    from numpy import asarray
    import matplotlib.pyplot as plt
    import random
    import os
    import cv2
    import numpy as np
    import torch
    # softmax = nn.Softmax(dim=0)
    # 
    # torch.autograd.set_detect_anomaly(True)
    full='/home/ubuntu-user/Desktop/reid/data/Occluded_REID/occluded_body_images'
    # rgb_img=Image.open(full).convert('RGB')
    # rgb_img =test_transform(rgb_img)
    # rgb_img = rgb_img.unsqueeze(0)
    # rgb_img=rgb_img.to('cuda')
    # feat  = model(rgb_img)
     
     
    model.eval()
    dir_list = os.listdir(full)
    for dirt in dir_list:
 
        rgb_img=Image.open(os.path.join(full,dirt)).convert('RGB')
        rgb_img =test_transform(rgb_img)
        input_image=rgb_img.unsqueeze(0)
        input_image=input_image.to('cuda')
        target_layers=[model.base.blocks[-1].norm1]
         
        # Construct the CAM object once, and then re-use it on many images:
        # GradCAM for clean face
        # GradCAM++ for occluded face
        # cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        def reshape_transform(tensor, height=22, width=11):
            result = tensor[:, 1 :  , :].reshape(tensor.size(0),
            height, width, tensor.size(2))
            # Bring the channels to the first dimension,d
            # like in CNNs.
            result = result.transpose(2, 3).transpose(1, 2)
            # print(result.shape)
            return result

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True,reshape_transform=reshape_transform)
        cam.batch_size = 1
        print('before')
        grayscale_cam = cam(input_tensor=input_image, targets=None, aug_smooth=True,eigen_smooth=True)
        print('after')
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        plt.imsave('test_{}'.format(item),visualization)
        break
      

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num,val_transforms,_ = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.load_param(cfg.TEST.WEIGHT)
    # GAM__vis(model,val_transforms)

    if cfg.DATASETS.NAMES == 'VehicleID':
        for trial in range(10):
            train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num,_ = make_dataloader(cfg)
            rank_1, rank5 = do_inference(cfg,
                 model,
                 val_loader,
                 num_query)
            if trial == 0:
                all_rank_1 = rank_1
                all_rank_5 = rank5
            else:
                all_rank_1 = all_rank_1 + rank_1
                all_rank_5 = all_rank_5 + rank5

            logger.info("rank_1:{}, rank_5 {} : trial : {}".format(rank_1, rank5, trial))
        logger.info("sum_rank_1:{:.1%}, sum_rank_5 {:.1%}".format(all_rank_1.sum()/10.0, all_rank_5.sum()/10.0))
    else:
       do_inference(cfg,
                 model,
                 val_loader,
                 num_query)

