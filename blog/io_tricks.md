## Tricks to improve I/O throughoutput when training SSL and diffusion models
> Nov 4, 2025 by Jiachen 

This blog introduces several techniques to improve the I/O throughoutput when training SSL model (e.g. MoCo) and diffuison model.

### The I/O bottleneck
Overall, image data loading contains the following stages:  
- 1. File reading
- 2. Image decoding, e.g. Jpeg
- 3. Data augmentation, e.g., horizontal flip, ColorJitter, RandomResizedCrop, etc.
- 4. Transfer data from Memory to GPU

Let's start with some typical pytorch data loading pipeline utilized in diffusion model training:
```python

import torch
from torchvision import transforms

class ImageDataset(torch.utils.data.Dataset):
    def __init__(*args, **kwargs):
        ...
        self.data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

    ...

    def __getitem__(self, idx):
        path = self.images[idx]
        label = self.labels[idx]

        pil = Image.open(path)
        tensor = self.data_transforms(pil)

        return tensor, label

def main():
    ...
    dataset = ImageDataset(*args, **kwargs)
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=8, drop_last=True, shuffle=True, 
        pin_memory=True
    )

    for batch in dataloader:
        images, labels = batch
        images = images.to("cuda")
        labels = labels.to("cuda")
```

This pipeline reads, decodes and augments images sequentially on CPU before transferring them to GPUs. With an appropriate number of data loading processes (the `num_workers` parameter), it works fine on dataset like ImageNet-256, yet causes substantial data loading overhead on higher-resolution images, e.g., ImageNet-512.

Why? This is because the pipeline fails to consider several key aspects when processing and loading data:
- 1. transforms.ToTensor() converts input data into tensor float object. Operating with float object is expensive as it includes memory allocation before stacking with other samples in the same batch.
- 2. 