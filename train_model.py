import torch.optim as optim
import os
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
import torch
from Model.model import Model
from preprocess_data import CelebASpoofDataset, create_balanced_sampler
from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch.utils.data import Dataset, DataLoader

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model()

    checkpoint = torch.load("/kaggle/input/model-5/pytorch/default/1/model.pth", map_location=device)
            
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
            
    model.load_state_dict(new_state_dict)

    if torch.cuda.device_count() > 1:
        print(f"🔥 Đã kích hoạt {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    DATA_ROOT = "/kaggle/input/celeba-spoof-for-face-antispoofing/CelebA_Spoof_/CelebA_Spoof"
    LIST_FILE = "/kaggle/input/celeba-spoof-for-face-antispoofing/CelebA_Spoof_/CelebA_Spoof/metas/intra_test/train_label.txt"
    LIST_TEST_FILE ="/kaggle/input/celeba-spoof-for-face-antispoofing/CelebA_Spoof_/CelebA_Spoof/metas/intra_test/test_label.txt"

    transform_train = A.Compose([
        A.Perspective(scale=(0.05,0.15), keep_size=True, p=0.5),
        A.RandomResizedCrop(size=(256,256), scale=(0.3, 1.0), ratio=(0.8,1.2), p=1.0),  
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=5),
        ], p=0.4),
        A.ImageCompression(quality_range=(40, 60), p=0.8),
        A.OneOf([
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), elementwise=True, p=0.5),        
            A.ISONoise(color_shift=(0.02, 0.10), intensity=(0.1, 0.5))
        ], p=0.8),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.7),
        A.Downscale(scale_range=(0.3, 0.7), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    transform_val = A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    try:
        with open(LIST_FILE, 'r') as f:
            all_lines = f.readlines()
            labels = [int(line.split()[1]) for line in all_lines]
            train_lines, val_lines = train_test_split(all_lines, test_size=0.2, random_state=42, stratify=labels)
            train_dataset = CelebASpoofDataset(
                root_dir=DATA_ROOT,
                data_lines=train_lines,
                transform=transform_train
            )

            val_dataset = CelebASpoofDataset(
                root_dir=DATA_ROOT,
                data_lines=val_lines,
                transform=transform_val
            )
    except Exception as e:
        print("Chưa tìm thấy đường dẫn đúng, bạn hãy check lại os.listdir nhé!")
        print("Lỗi:", e)



    sampler = create_balanced_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        sampler=sampler,
        shuffle=False,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2
    )

    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, depth_targets, labels in progress_bar:
            images = images.to(device)
            depth_targets = depth_targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, depth_targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': running_loss / (progress_bar.n + 1)})
        avg_train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, depth_targets, labels in val_loader:
                images = images.to(device)
                depth_targets = depth_targets.to(device)

                outputs = model(images)
                loss = criterion(outputs, depth_targets)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\n📊 KẾT QUẢ EPOCH {epoch+1}:")
        print(f"   -> Train Loss: {avg_train_loss:.4f}")
        print(f"   -> Val Loss:   {avg_val_loss:.4f}")
        
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), f"ddmodel_epoch_{epoch+1}.pth")
        else: torch.save(model.state_dict(), f"ddmodel_epoch_{epoch+1}.pth")
        print("-> Đã lưu model!")
    print("🎉 Đã huấn luyện xong!")