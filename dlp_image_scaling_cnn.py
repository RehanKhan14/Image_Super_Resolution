import torch
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2)
        )

    def forward(self, x):
        return self.model(x)

import torch.nn.functional as F
import math

def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    return 20 * torch.log10(max_val / torch.sqrt(mse))

import torch.optim as optim
import time

def train(model, dataloader, num_epochs=50, lr=1e-4):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\nStarting Model Training\n")
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_psnr = 0
        start_time = time.time()
        
        for lr_imgs, hr_imgs, fname in dataloader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            preds = model(lr_imgs)
            loss = criterion(preds, hr_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if ((epoch+1)%2==0):
                total_psnr += psnr(preds, hr_imgs).item()
        
        avg_loss = total_loss / len(dataloader)
        end_time = time.time()
        epoch_duration = end_time - start_time
        if ((epoch+1)%2==0):
            total_psnr += psnr(preds, hr_imgs).item()
            avg_psnr = total_psnr / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - PSNR: {avg_psnr:.2f} dB - Time Taken: {epoch_duration/60:.1f} minutes")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Time Taken: {epoch_duration/60:.1f} minutes")
        # print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(dataloader):.4f}")

from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

def infer(model, lr_img_tensor):
    model.eval()
    with torch.no_grad():
        lr_img_tensor = lr_img_tensor.unsqueeze(0).to(device)
        sr_img_tensor = model(lr_img_tensor).squeeze(0).cpu().clamp(0, 1)
    return sr_img_tensor

device = torch.device("cuda")
print(f"Cuda Available: {torch.cuda.is_available()}")
model = SRCNN().to(device)

"""## Dataset Stuff"""

#download from google drive
# !pip install -q gdown
# !gdown --id 1BEUG_jnbzqd_Nr8ITocEDJxMI0sNdmcE --output ILSVRC2013_val.tar
# # https://drive.google.com/file/d/1BEUG_jnbzqd_Nr8ITocEDJxMI0sNdmcE/view?usp=sharing

# # unzip
# !mkdir -p /content/ILSVRC2013
# !tar -xvf ILSVRC2013_val.tar -C /content/ILSVRC2013/

from PIL import Image, ImageFile
import os

# def make_lr_images(input_dir, output_dir, scale=3):
#     os.makedirs(output_dir, exist_ok=True)
#     for filename in os.listdir(input_dir):
#         if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             continue
#         img = Image.open(os.path.join(input_dir, filename)).convert('RGB')
#         w, h = img.size
#         new_w, new_h = w // scale, h // scale
#         lr_img = img.resize((new_w, new_h), Image.BICUBIC)
#         lr_img = lr_img.resize((w, h), Image.BICUBIC)  # Upsample back to HR size
#         lr_img.save(os.path.join(output_dir, filename))



ImageFile.LOAD_TRUNCATED_IMAGES = True  #  allow partial image loads

def make_lr_images(hr_dir, lr_dir, scale=3):
    os.makedirs(lr_dir, exist_ok=True)
    for fname in os.listdir(hr_dir):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        try:
            img = Image.open(os.path.join(hr_dir, fname)).convert('RGB')

            w, h = img.size
            w, h = (w // scale) * scale, (h // scale) * scale
            img = img.resize((w, h), Image.BICUBIC)

            lr = img.resize((w // scale, h // scale), Image.BICUBIC)
            lr_up = lr.resize((w, h), Image.BICUBIC)

            lr_up.save(os.path.join(lr_dir, fname))
        except Exception as e:
            print(f"Skipping {fname}: {e}")

hr_dir = './ILSVRC2013/ILSVRC2013_DET_val' 
lr_dir = './ILSVRC2013/lr'
# make_lr_images(hr_dir, lr_dir, scale=3)

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ImageSuperResolutionDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, size=(256, 256)):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.size = size

        self.filenames = [
            f for f in os.listdir(hr_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.exists(os.path.join(lr_dir, f))
        ]
        self.to_tensor = transforms.ToTensor()
        self.resizer = transforms.Resize(size)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        hr = Image.open(os.path.join(self.hr_dir, fname)).convert('RGB')
        lr = Image.open(os.path.join(self.lr_dir, fname)).convert('RGB')

        hr = self.resizer(hr)
        lr = self.resizer(lr)

        return self.to_tensor(lr), self.to_tensor(hr), fname

import os

hr_files = set(os.listdir(hr_dir))
lr_files = set(os.listdir(lr_dir))

print("HR images:", len(hr_files))
print("LR images:", len(lr_files))
print("Matching files:", len(hr_files & lr_files))

from torch.utils.data import DataLoader, random_split

# Prepare data
# hr_dir = '/ILSVRC2013/ILSVRC2013_DET_val'  # or /hr if you've cleaned
# lr_dir = '/ILSVRC2013/lr'


dataset = ImageSuperResolutionDataset(hr_dir, lr_dir)
print(len(dataset)) 

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

train(model, train_loader, num_epochs=20, lr=1e-4)

# save the model
torch.save(model.state_dict(), "srcnn_rgb_9x5x5 20 epochs 3x scale.pth")

#load the model
# model = SRCNN().to(device)  
# model.load_state_dict(torch.load("srcnn_rgb_9x5x5 20 epochs 3x scale.pth"))
model.eval()  

total_psnr = 0.0

with torch.no_grad():
    for lr_img, hr_img, fname in test_loader:
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        pred = model(lr_img).clamp(0, 1)
        total_psnr += psnr(pred, hr_img)

average_psnr = total_psnr / len(test_loader)
print(f"Average PSNR on test set: {average_psnr:.2f} dB")

import matplotlib.pyplot as plt


lr_img, hr_img, fname = test_dataset[0]
sr_img = infer(model, lr_img)  

def show_image(tensor, title=""):
    img = tensor.permute(1, 2, 0).cpu().numpy()
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); show_image(lr_img, "Low-Res Input")
plt.subplot(1,3,2); show_image(sr_img, "SRCNN Output")
plt.subplot(1,3,3); show_image(hr_img, "Ground Truth")
plt.savefig("srcnn_comparison.png", bbox_inches="tight", dpi=300)
plt.show()

import os

output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)

from torchvision.utils import save_image

model.eval()
with torch.no_grad():
    for i, (lr_img, hr_img, fname) in enumerate(test_loader):
        lr_img = lr_img.to(device)

        # Inference
        sr_img = model(lr_img).clamp(0, 1)
        
        base_name = os.path.splitext(fname[0])[0]
        # Save output
        save_path = os.path.join(output_dir, f"sr_{base_name}.png")
        save_image(sr_img.squeeze(0), save_path)

        if i < 3:  
            print(f"Saved: {save_path}")