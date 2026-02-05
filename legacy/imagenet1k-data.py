# Install timm if you haven't
# pip install timm

from timm.data import create_transform
from torchvision import datasets

def build_swin_loader(data_dir, batch_size, input_size=224, is_training=True):
    if is_training:
        # This function from TIMM creates the EXACT augmentations used in DeiT/Swin
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1', # The Swin/DeiT standard policy
            interpolation='bicubic',
            re_prob=0.25, # Random Erasing probability
            re_mode='pixel',
            re_count=1,
        )
        dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    else:
        # Validation transforms
        t = []
        size = int((256 / 224) * input_size) # Resize to 256 for 224 input
        transform = transforms.Compose([
            transforms.Resize(size, interpolation=3), # 3 is Bicubic
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8, # Try to max this out on your Pro 6000
        pin_memory=True,
        shuffle=is_training
    )