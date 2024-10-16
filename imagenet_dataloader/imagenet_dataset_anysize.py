import os
from PIL import Image
import numpy as np
import pickle

import torchvision.transforms as transforms
import torch
import torch.utils.data as data
import numpy


class ImageFolderDataset(data.Dataset):
    def __init__(self, folder_path, label_file='imagenet_val_labels.pkl', transform=None, 
                        permute=True, normalize=True, rank=0, world_size=1, return_numpy=False):
        self.folder_path = folder_path
        self.imgs = []
        valid_images = ['.jpeg', '.png', '.jpg']
        for f in os.listdir(self.folder_path):
            ext = os.path.splitext(f)[1]
            if ext.lower() in valid_images:
                self.imgs.append(f)
        
        if 'label' in self.imgs[0]:
            self.imgs = sorted(self.imgs, key = lambda x : int(x.split('_')[0]))
        else:
            self.imgs = sorted(self.imgs)
        print('Find %d images in the folder %s' % (len(self.imgs), self.folder_path))

        if world_size > 1:
            num_samples_per_rank = int(np.ceil(len(self.imgs) / world_size))
            start = rank * num_samples_per_rank
            end = (rank+1) * num_samples_per_rank
            self.imgs = self.imgs[start:end]
            self.num_samples_per_rank = num_samples_per_rank
        else:
            self.num_samples_per_rank = len(self.imgs)

        self.len = len(self.imgs)
        print('This process hanldes %d images in the folder %s' % (self.len, self.folder_path))
        
        
        self.transform = transform 
        self.permute = permute
        self.normalize = normalize
        self.return_numpy = return_numpy

        if not label_file is None:
            if isinstance(label_file, int):
                self.labels = label_file
            else:
                handle = open(label_file, 'rb')
                data = pickle.load(handle)
                handle.close()
                self.labels = data['imgs_label_dict']

        else:
            self.labels = None

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_name = self.imgs[index]
        x = Image.open(os.path.join(self.folder_path, img_name))

        if not self.transform is None:
            x = self.transform(x)

        x = torch.from_numpy(np.array(x))
        if len(x.shape) == 2:
            x = torch.stack([x,x,x], dim=2)
        if x.shape[2] == 4:
            x = x[:,:,0:3]  
        if self.permute:
            x = x.permute(2,0,1)
        if self.normalize:

            x = x.to(torch.float)
            x = x/255 * 2 - 1

        if self.return_numpy:
            x = x.detach().numpy()

        if isinstance(self.labels, int):
            y = self.labels
        elif isinstance(self.labels, dict):
            key = img_name.split('.')[0]
        y = 990
        return x, y

def save_image_tensor(x, label, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    x = x.permute(0,2,3,1) # shape BHWC
    x = x.detach().cpu().numpy().astype(numpy.uint8)
    label = label.detach().cpu().numpy().astype(numpy.uint8)

    for i in range(label.shape[0]):
        save_name = str(i) + '_label_' + str('197') + '.png'
        Image.fromarray(x[i]).save(os.path.join(save_dir, save_name))

if __name__ == '__main__':
    import pdb
    path=''
    image_size = 256
    transform = transforms.Compose(
            [   
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size)
            ]
        )

    dataset = ImageFolderDataset(path, transform=transform, permute=True, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
    imgs = []
    labels = []

    
    for i in range(len(dataset)):
        x = dataset.__getitem__(i)
        expanded_x = x[0].unsqueeze(0)    
        imgs.append(expanded_x)

    for i, data in enumerate(dataloader):
        x, y = data
        labels.append(y)
    
    imgs = torch.cat(imgs, dim=0)
    labels = torch.cat(labels, dim=0)
    image_size2 = 256
    file_name_high = ('data_path_%d' % image_size2)
    save_image_tensor(imgs, labels, file_name_high)

    imgs = imgs.permute(0,2,3,1) # shape BHWC
    imgs = imgs.detach().cpu().numpy().astype(numpy.uint8)
    labels = labels.detach().cpu().numpy().astype(numpy.uint8)
    file_name_high_1 = ('npz_path_%d.npz' % image_size2)
    np.savez(file_name_high_1, imgs, labels)





