<div align="center">

<h1>BIR-D: Taming Generative Diffusion Prior for Universal Blind Image Restoration</h1>

In this study, we aim to use a well-trained DDPM to learn the prior distribution of images and ultimately solve non-blind and blind problems in various image restoration tasks.

<div>
    <a href='https://Tusiwei.github.io/' target='_blank'>Siwei Tu</a><sup>1</sup>&emsp;
    <a href='https://shangchenzhou.com/' target='_blank'>Weidong Yang</a><sup>1,†</sup>&emsp;
    <a href='https://scholar.google.com.sg/citations?user=fMXnSGMAAAAJ&hl=en' target='_blank'>Ben Fei</a><sup>2</sup>&emsp;
</div>

<div>
    <sup>1</sup>Fudan University&emsp; 
    <sup>2</sup>Chinese University of Hong Kong&emsp; 
</div>

---

</div>


## :diamonds: Download Checkpoints and Data
- Download pretrained uncondition DDPMs on ImageNet-256 from (https://github.com/openai/guided-diffusion). 
- Then download 1000 images from the validation set of the Imagenet dataset.
- The download address is [https://github.com/XingangPan/deep-generative-prior/](https://github.com/XingangPan/deep-generative-prior/)

For the downloaded dataset folder, command below can be used to automatically generate NPZ files that meet the requirements. 
```
python /BIR-D/imagenet_dataloader/imagenet_dataset_anysize.py
```

### Environment
```
pip install -r requirements.txt
```

## :fire:Blind Image Restoration
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

## :thumbsup: Our paper is inspired by
- [https://generativediffusionprior.github.io/](https://generativediffusionprior.github.io/)(the GDP repo)
- [https://0x3f3f3f3fun.github.io/projects/diffbir/](https://0x3f3f3f3fun.github.io/projects/diffbir/)(the DiffBIR repo)


