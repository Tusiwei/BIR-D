"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

# from guided_diffusion import dist_util, logger
from guided_diffusion import logger
from guided_diffusion.script_util_x0 import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from save_image_utils import save_images
from npz_dataset import NpzDataset, DummyDataset
from imagenet_dataloader.imagenet_dataset import ImageFolderDataset

import torchvision.transforms as transforms
from PIL import Image,ImageFilter
import cv2
import pdb
import random
import math
import torch
import torch.nn as nn

import MyLoss
os.environ['CUDA_VISIBLE_DEVICES']='6'
def get_dataset(path, global_rank, world_size):
    if os.path.isfile(path): # base_samples could be store in a .npz file
        dataset = NpzDataset(path, rank=global_rank, world_size=world_size)
    else:
        dataset = ImageFolderDataset(path, label_file='./imagenet_dataloader/imagenet_val_labels.pkl', transform=None, 
                        permute=True, normalize=True, rank=global_rank, world_size=world_size)
    return dataset

def gray_dgp(image):
    n = int(image.shape[0])
    h = int(image.shape[2])
    w = int(image.shape[3])
    r = image[:, 0, :, :]
    g = image[:, 1, :, :]
    b = image[:, 2, :, :]
    gray = 1/3 * r + 1/3 * g + 1/3 * b
    image_res = gray.view(n, 1, h, w).expand(n, 3, h, w)
    return image_res


def main():
    args = create_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = th.device('cuda')
    save_dir = args.save_dir if len(args.save_dir)>0 else None

    # dist_util.setup_dist()
    logger.configure(dir = save_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        th.load(args.model_path, map_location="cpu")
    )
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()


    L_color = MyLoss.L_color()


    def color_cond_fn(x, t,weight=None,light_mask=None,guidance_scale = None,corner=None,  y=None, x_lr=None, sample_noisy_x_lr=False, diffusion=None, sample_noisy_x_lr_t_thred=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            loss = 0
            if not x_lr is None:
                x_lr = x_lr[:, :, corner[0]:corner[0]+corner[2], corner[1]:corner[1]+corner[2]]
                
                device_x_in_lr = x_in.device
                x_in_lr = x_in
                weight.requires_grad_()
                light_mask.requires_grad_()
                in_channels = 3
                conv_layer = nn.Conv2d(in_channels, in_channels, kernel_size=9, padding=4, bias=False,groups=3)
                conv_layer.weight = nn.Parameter(weight)
                x_in_lr = conv_layer((x_in_lr+1)/2)
                
                x_in_lr = gray_dgp(x_in_lr)
                x_in_lr.to(th.uint8)
                

                if sample_noisy_x_lr:
                    t_numpy = t.detach().cpu().numpy()
                    spaced_t_steps = [diffusion.timestep_reverse_map[t_step] for t_step in t_numpy]
                    if sample_noisy_x_lr_t_thred is None or spaced_t_steps[0] < sample_noisy_x_lr_t_thred:
                        print('Sampling noisy lr')
                        spaced_t_steps = th.Tensor(spaced_t_steps).to(t.device).to(t.dtype)
                        x_lr = diffusion.q_sample(x_lr, spaced_t_steps)

                x_lr = x_lr.to(device_x_in_lr) 
                x_lr = (x_lr + 1) / 2
                mse = (x_in_lr - x_lr) ** 2
                mse = mse.mean(dim=(1,2,3))
                mse = mse.sum()

                loss_col = torch.mean(L_color(x_in))
                
                conv_layer.zero_grad()
                mse.backward(retain_graph=True)
                grad_conv_kernel = conv_layer.weight.grad
                weight = weight- grad_conv_kernel * 0.01
                
                loss = - mse * args.img_guidance_scale
                # loss = loss + mse * guidance_scale

                print('step t %d, mse is %.8f' % (t[0], mse))
            return weight, light_mask, th.autograd.grad(loss, x_in)[0],mse


    def model_fn(x, t, y=None):
        assert y is not None
        # assert light_factor is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("loading dataset...")
    # load gan or vae generated images
    if args.start_from_scratch and args.use_img_for_guidance:
        pass
    else:
        if args.start_from_scratch:
            dataset = DummyDataset(args.num_samples, rank=args.global_rank, world_size=args.world_size)
        else:
            dataset = get_dataset(args.dataset_path, args.global_rank, args.world_size)
        dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    # load lr images that are used for guidance 
    if args.use_img_for_guidance:
        dataset_lr = get_dataset(args.base_samples, args.global_rank, args.world_size)
        dataloader_lr = th.utils.data.DataLoader(dataset_lr, batch_size=args.batch_size, shuffle=False, num_workers=16)
        if args.start_from_scratch:
            dataset = DummyDataset(len(dataset_lr), rank=0, world_size=1)
            dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)
        dataloader = zip(dataloader, dataloader_lr)

    # args.save_png_files=True
    if args.save_png_files:
        print(logger.get_dir())
        os.makedirs(os.path.join(logger.get_dir(), 'images'), exist_ok=True)
        os.makedirs(os.path.join(logger.get_dir(), 'lr'), exist_ok=True)
        # os.makedirs(os.path.join(logger.get_dir(), 'gt'), exist_ok=True)
        start_idx = args.global_rank * dataset.num_samples_per_rank

    logger.log("sampling...")
    all_images = []
    all_labels = []
    for i, data in enumerate(dataloader):
        if args.use_img_for_guidance:
            image, label = data[0]
            image_lr, label = data[1]
            image_lr = image_lr.to(device)
            cond_fn = lambda x,t,weight,light_mask,guidance_scale,corner,y : color_cond_fn(x, t,weight=weight, light_mask=light_mask,guidance_scale=guidance_scale, corner=corner, y=y, x_lr=image_lr, sample_noisy_x_lr=args.sample_noisy_x_lr, diffusion=diffusion, sample_noisy_x_lr_t_thred=args.sample_noisy_x_lr_t_thred)
        else:
            image, label = data
            cond_fn = lambda x,t,y : general_cond_fn(x, t, y=y, x_lr=None)
        if args.start_from_scratch:
            shape = (image.shape[0], 3, args.image_size, args.image_size)
        else:
            shape = list(image.shape)
        if args.start_from_scratch and not args.use_img_for_guidance:
            classes = th.randint(low=0, high=NUM_CLASSES, size=(shape[0],), device=device)
        else:
            classes = label.to(device).long()
        image = image.to(device)
        model_kwargs = {}
        model_kwargs["y"] = classes
        weight = torch.tensor([[0.0034, 0.0052, 0.0072, 0.0087, 0.0092, 0.0087, 0.0072, 0.0052, 0.0034],
        [0.0052, 0.0081, 0.0111, 0.0134, 0.0143, 0.0134, 0.0111, 0.0081, 0.0052],
        [0.0072, 0.0111, 0.0152, 0.0184, 0.0196, 0.0184, 0.0152, 0.0111, 0.0072],
        [0.0087, 0.0134, 0.0184, 0.0222, 0.0236, 0.0222, 0.0184, 0.0134, 0.0087],
        [0.0092, 0.0143, 0.0196, 0.0236, 0.0252, 0.0236, 0.0196, 0.0143, 0.0092],
        [0.0087, 0.0134, 0.0184, 0.0222, 0.0236, 0.0222, 0.0184, 0.0134, 0.0087],
        [0.0072, 0.0111, 0.0152, 0.0184, 0.0196, 0.0184, 0.0152, 0.0111, 0.0072],
        [0.0052, 0.0081, 0.0111, 0.0134, 0.0143, 0.0134, 0.0111, 0.0081, 0.0052],
        [0.0034, 0.0052, 0.0072, 0.0087, 0.0092, 0.0087, 0.0072, 0.0052, 0.0034]])
        weight = torch.unsqueeze(weight, dim=0)
        weight = weight.repeat(3, 1, 1, 1)
        weight = weight.to(device)
        weight = weight.float()
        
        light_mask =  th.rand([256,256], device=device)/10000
        guidance_scale = torch.tensor(6000)
        sample_y = image_lr

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        if args.start_from_scratch:
            sample, weight, light_mask,guidance_scale = sample_fn(
                model_fn,
                shape,
                weight,
                light_mask,
                guidance_scale,
                sample_y,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=device
            )
        else:
            sample = sample_fn(
                model_fn,
                shape,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=device,
                noise=image,
                denoise_steps=args.denoise_steps
            )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        image_lr = ((image_lr + 1) * 127.5).clamp(0, 255).to(th.uint8)
        image_lr = image_lr.permute(0, 2, 3, 1)
        image_lr = image_lr.contiguous()

        sample = sample.detach().cpu().numpy()
        classes = classes.detach().cpu().numpy()
        image_lr = image_lr.detach().cpu().numpy()
        if args.save_png_files:
            save_images(sample, classes, start_idx + len(all_images) * args.batch_size, os.path.join(logger.get_dir(), 'images'))
        save_images(image_lr, label.long(), start_idx + len(all_images) * args.batch_size, save_dir=os.path.join(logger.get_dir(), 'lr'))
        all_images.append(sample)
        all_labels.append(classes)
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    if args.save_numpy_array:
        arr = np.concatenate(all_images, axis=0)
        label_arr = np.concatenate(all_labels, axis=0)

        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_rank_{args.global_rank}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    # dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=100,
        batch_size=1,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    parser.add_argument("--device", default=1, type=int, help='the cuda device to use to generate images')
    parser.add_argument("--global_rank", default=0, type=int, help='global rank of this process')
    parser.add_argument("--world_size", default=1, type=int, help='the total number of ranks')
    parser.add_argument("--save_dir", type=str, help='the directory to save the generate images')
    parser.add_argument("--save_png_files", action='store_false', help='whether to save the generate images into individual png files')
    parser.add_argument("--save_numpy_array", action='store_true', help='whether to save the generate images into a single numpy array')
    
    parser.add_argument("--denoise_steps", default=25, type=int, help='number of denoise steps')
    parser.add_argument("--dataset_path", type=str, help='path to the generated images. Could be an npz file or an image folder')
    
    parser.add_argument("--use_img_for_guidance", action='store_false', help='whether to use a (low resolution) image for guidance. If true, we generate an image that is similar to the low resolution image')
    parser.add_argument("--img_guidance_scale", default=4500, type=float, help='guidance scale')
    parser.add_argument("--base_samples", type=str, help='the directory or npz file to the guidance imgs. This folder should have the same structure as dataset_path, there should be a one to one mapping between images in them')
    parser.add_argument("--sample_noisy_x_lr", action='store_false', help='whether to first sample a noisy x_lr, then use it for guidance. ')
    parser.add_argument("--sample_noisy_x_lr_t_thred", default=1e8, type=int, help='only for t lower than sample_noisy_x_lr_t_thred, we add noise to lr')
    
    parser.add_argument("--start_from_scratch", action='store_false', help='whether to generate images purely from scratch, not use gan or vae generated samples')
    # num_samples is defined elsewhere, num_samples is only valid when start_from_scratch and not use img as guidance
    # if use img as guidance, num_samples will be set to num of guidance images
    # parser.add_argument("--num_samples", type=int, default=50000, help='num of samples to generate, only valid when start_from_scratch is true')
    return parser

import pdb
if __name__ == "__main__":
    main()
