# BIR-D: Taming Generative Diffusion Prior for Universal Blind Image Restoration
In this study, we aim to use a well-trained DDPM to learn the prior distribution of images and ultimately solve non-blind and blind problems in various image restoration tasks.

## Download Checkpoints and Data
Download pretrained uncondition DDPMs on ImageNet-256 from (https://github.com/openai/guided-diffusion). 
Then download 1000 images from the validation set of the Imagenet dataset as the input set for the deblurring task. 
The download address is [https://github.com/XingangPan/deep-generative-prior/](https://github.com/XingangPan/deep-generative-prior/)

For the downloaded dataset folder, command
```
python /BIR-D/imagenet_dataloader/imagenet_dataset_anysize.py
```
can be used to automatically generate NPZ files that meet the requirements. 
Noted that the 97th line of code in (/BIR-D/imagenet_dataloader/imagenet_dataset_anysize.py) needs to be modified to the address of the image dataset first.

## Environment
```
pip install -r requirements.txt
```

### Deblurring Task
A given set of degraded images can be used for testing, and custom degradation can also be used to test the blind image restoration performance of BIR-D.
```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
```
```
python main.py \
$MODEL_FLAGS \
--save_dir [Path of storing output results]
--base_samples [Path of the npz file corresponding to the downloaded Imagenet 1k dataset]
```

Our paper is inspired by
- [https://generativediffusionprior.github.io/](https://generativediffusionprior.github.io/)(the GDP repo)
- [https://0x3f3f3f3fun.github.io/projects/diffbir/](https://0x3f3f3f3fun.github.io/projects/diffbir/)(the DiffBIR repo)


