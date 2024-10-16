<div align="center">

<h1>BIR-D: Taming Generative Diffusion Prior for Universal Blind Image Restoration</h1>

In this study, we aim to use a DDPM to learn the prior distribution of images and ultimately solve non-blind and blind problems in various image restoration tasks.

<div>
    <a href='https://Tusiwei.github.io/' target='_blank'>Siwei Tu</a><sup>1</sup>&emsp;
    <a href='https://shangchenzhou.com/' target='_blank'>Weidong Yang</a><sup>1,†</sup>&emsp;
    <a href='https://scholar.google.com.sg/citations?user=fMXnSGMAAAAJ&hl=en' target='_blank'>Ben Fei</a><sup>2,†</sup>&emsp;
</div>

<div>
    <sup>1</sup>Fudan University&emsp; 
    <sup>2</sup>Chinese University of Hong Kong&emsp; 
</div>



<img src="assets/teaser.png" width="800px"/>

---

</div>


## :diamonds: Checkpoints and Environment
- Download pretrained uncondition DDPMs on ImageNet-256 from (https://github.com/openai/guided-diffusion). 


- For the downloaded dataset folder, command below can be used to automatically generate NPZ files that meet the requirements: 
```
# generate .npz files
python /BIR-D/dataloader/imagenet_dataset_end.py 
```

- Use the fllowing command to configurate environment: 
```
# install python dependencies
pip install -r requirements.txt
pip install -e .
```

## Tasks

### :fire:Blind Image Restoration
A given set of degraded images can be used for testing, and custom degradation can also be used to test the blind image restoration performance of BIR-D.
```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
```
```
python linear.py \
$MODEL_FLAGS \
--save_dir [Path of storing output results]
--base_samples [Path of the npz file corresponding to the downloaded Imagenet 1k dataset]
```

<img src="assets/blur.png" width="800px"/>

### :fire:Blind Face Restoration / Motion Blur Reduction

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
```

```
python blind_image_restoration.py \
$MODEL_FLAGS \
--save_dir [Path of storing output results]
--base_samples [Path of the blind image restoration dataset]
```

<img src="assets/blind.png" width="800px"/>

### :fire:Multi-Degradation Image Restoration

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
```

```
python multi_restoration.py \
$MODEL_FLAGS \
--save_dir [Path of storing output results]
--base_samples [Path of the multi-degradation dataset]
```


<img src="assets/multi.png" width="800px"/>

### :fire:Low-light Enhancement

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
```

```
python lowlight.py \
$MODEL_FLAGS \
--save_dir [Path of storing output results]
--base_samples [Path of the low-light enhancement dataset]
```


<img src="assets/lowlight.png" width="800px"/>


## :clap: Acknowledgement

The authors would like to thank Zhaoyang Lyu for his technical assistance. This work was supported
by the National Natural Science Foundation of China (U2033209)

Our paper is inspired by:
- [https://generativediffusionprior.github.io/](https://generativediffusionprior.github.io/)(the GDP repo)
- [https://0x3f3f3f3fun.github.io/projects/diffbir/](https://0x3f3f3f3fun.github.io/projects/diffbir/)(the DiffBIR repo)

Thanks for their awesome works!

## :telephone_receiver: Contact

If you have any other questions, please feel free to send them via email tusiwei906@gmail.com to contact us. :rose:

