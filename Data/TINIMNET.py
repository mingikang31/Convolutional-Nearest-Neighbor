import os
import shutil



def organize_folder():
    val_dir = '/Users/mingikang/Developer/KVT/Data/tiny-imagenet-200/val'
    images_dir = os.path.join(val_dir, 'images')
    annotations_file = os.path.join(val_dir, 'val_annotations.txt')

    # Read image → label mapping
    with open(annotations_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split('\t')
            img_name, label = parts[0], parts[1]

            label_dir = os.path.join(val_dir, label)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)

            src_path = os.path.join(images_dir, img_name)
            dst_path = os.path.join(label_dir, img_name)

            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)

    # Optional cleanup: remove old 'images' directory
    shutil.rmtree(images_dir)
    print("✅ Validation folder reorganized successfully!")
    
    
def create_dataset():
    
    import os
    import json
    from torchvision import datasets, transforms
    from torchvision.datasets.folder import ImageFolder, default_loader
    import os
    from torchvision import datasets, transforms

    val_dir = '/Users/mingikang/Developer/KVT/Data/tiny-imagenet-200/val'

    # Define basic transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Tiny ImageNet images are 64x64
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean/std
                            std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    nb_classes = len(dataset.classes)

    # Example usage
    print(f"Loaded Tiny ImageNet validation dataset with {len(dataset)} images and {nb_classes} classes.")