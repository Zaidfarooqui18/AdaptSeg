import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import os
import json
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "train_dir": os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'train'),
    "val_dir":   os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'val'),
    "batch_size": 2,
    "lr": 1e-4,
    "n_epochs": 40,
    "img_w": 224,
    "img_h": 224,
    "mining_start_epoch": 5,
    "output_dir": os.path.join(script_dir, 'train_stats'),
    "model_path": os.path.join(script_dir, 'segmentation_head.pth'),
    "log_file":   os.path.join(script_dir, 'train_stats', 'training_log.json'),
}

value_map = {0:0,100:1,200:2,300:3,500:4,550:5,700:6,800:7,7100:8,10000:9}
n_classes = len(value_map)
CLASS_NAMES = ['Background','Trees','Lush Bushes','Dry Grass','Dry Bushes','Ground Clutter','Logs','Rocks','Landscape','Sky']

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw, new in value_map.items():
        new_arr[arr == raw] = new
    return Image.fromarray(new_arr)

class DesertAugment:
    def __init__(self, w, h, train=True):
        self.w, self.h, self.train = w, h, train
    def __call__(self, image, mask):
        image = image.resize((self.w, self.h), Image.BILINEAR)
        mask  = mask.resize((self.w, self.h),  Image.NEAREST)
        if not self.train:
            return image, mask
        if np.random.rand() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)
        if np.random.rand() > 0.3:
            image = TF.adjust_brightness(image, np.random.uniform(0.6, 1.4))
            image = TF.adjust_contrast(image, np.random.uniform(0.7, 1.3))
        if np.random.rand() > 0.6:
            img_t = TF.to_tensor(image)
            img_t = transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 2.0))(img_t)
            img_t[0] = torch.clamp(img_t[0] * np.random.uniform(1.0, 1.2), 0, 1)
            img_t[2] = torch.clamp(img_t[2] * np.random.uniform(0.7, 1.0), 0, 1)
            image = TF.to_pil_image(img_t)
        return image, mask

class MaskDataset(Dataset):
    def __init__(self, data_dir, augment, normalize):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.augment   = augment
        self.normalize = normalize
        self.data_ids  = sorted(os.listdir(self.image_dir))
        self.sample_weights = np.ones(len(self.data_ids))
        print(f"  Found {len(self.data_ids)} images in {data_dir}")
    def __len__(self):
        return len(self.data_ids)
    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        mask  = Image.open(os.path.join(self.masks_dir, data_id))
        mask  = convert_mask(mask)
        image, mask = self.augment(image, mask)
        img_t  = self.normalize(image)
        mask_t = transforms.ToTensor()(mask) * 255
        return img_t, mask_t, idx

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem  = nn.Sequential(nn.Conv2d(in_channels, 128, 7, padding=3), nn.GELU())
        self.block = nn.Sequential(
            nn.Conv2d(128, 128, 7, padding=3, groups=128), nn.GELU(),
            nn.Conv2d(128, 128, 1), nn.GELU(),
        )
        self.classifier = nn.Conv2d(128, out_channels, 1)
    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        return self.classifier(self.block(self.stem(x)))

def compute_iou(pred, target, num_classes=10):
    pred   = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    ious   = []
    for c in range(num_classes):
        tp = ((pred == c) & (target == c)).sum().float()
        fp = ((pred == c) & (target != c)).sum().float()
        fn = ((pred != c) & (target == c)).sum().float()
        d  = tp + fp + fn
        ious.append((tp / d).item() if d > 0 else float('nan'))
    return float(np.nanmean(ious)), ious

@torch.no_grad()
def update_sample_weights(classifier, backbone, dataset, device, cfg):
    print("  Computing uncertainty weights...")
    classifier.eval()
    old_aug = dataset.augment
    dataset.augment = DesertAugment(cfg['img_w'], cfg['img_h'], train=False)
    loader  = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)
    weights = np.zeros(len(dataset))
    for imgs, _, idxs in loader:
        imgs = imgs.to(device)
        feats  = backbone.forward_features(imgs)["x_norm_patchtokens"]
        logits = classifier(feats)
        up     = F.interpolate(logits, size=(cfg['img_h'], cfg['img_w']), mode="bilinear", align_corners=False)
        probs  = torch.softmax(up, dim=1)
        entropy = (-probs * torch.log(probs + 1e-8)).sum(dim=1).mean(dim=[1,2])
        for i, idx in enumerate(idxs.numpy()):
            weights[idx] = entropy[i].item()
    dataset.augment = old_aug
    classifier.train()
    mn, mx = weights.min(), weights.max()
    if mx > mn:
        weights = 0.1 + 0.9 * (weights - mn) / (mx - mn)
    dataset.sample_weights = weights
    return weights

def save_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    pairs = [('train_loss','val_loss','Loss'),('train_iou','val_iou','IoU'),
             ('train_dice','val_dice','Dice'),('train_pixel_acc','val_pixel_acc','Pixel Acc')]
    for ax, (tr, vl, title) in zip(axes.flatten(), pairs):
        ax.plot(history[tr], label='train')
        ax.plot(history[vl], label='val')
        ax.set_title(title); ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'))
    plt.close()
    print(f"  Saved plots to {output_dir}/")

def main():
    cfg = CONFIG
    os.makedirs(cfg['output_dir'], exist_ok=True)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Apple Silicon MPS detected - GPU training enabled!")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    train_aug = DesertAugment(cfg['img_w'], cfg['img_h'], train=True)
    val_aug   = DesertAugment(cfg['img_w'], cfg['img_h'], train=False)
    print("\nLoading datasets...")
    train_ds = MaskDataset(cfg['train_dir'], train_aug, normalize)
    val_ds   = MaskDataset(cfg['val_dir'],   val_aug,   normalize)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False, num_workers=0)
    print("\nLoading DINOv2 backbone (downloading if first time ~100MB)...")
    torch.hub._validate_not_a_forked_repo = lambda a,b,c: True
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval().to(device)
    print("Backbone ready!")
    with torch.no_grad():
        sample, _, _ = train_ds[0]
        feats = backbone.forward_features(sample.unsqueeze(0).to(device))["x_norm_patchtokens"]
    n_emb = feats.shape[2]
    print(f"Embedding dim: {n_emb}")
    classifier = SegmentationHeadConvNeXt(
        in_channels=n_emb, out_channels=n_classes,
        tokenW=cfg['img_w']//14, tokenH=cfg['img_h']//14
    ).to(device)
    optimizer = optim.SGD(classifier.parameters(), lr=cfg['lr'], momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['n_epochs'])
    criterion = nn.CrossEntropyLoss()
    history = {k: [] for k in ['train_loss','val_loss','train_iou','val_iou','train_dice','val_dice','train_pixel_acc','val_pixel_acc']}
    log = {"epochs":[], "train_loss":[], "val_loss":[], "mean_iou":[], "per_class_iou":[]}
    best_iou = 0.0
    print(f"\nTraining for {cfg['n_epochs']} epochs...\n")
    print(f"{'Epoch':>6} {'T-Loss':>9} {'V-Loss':>9} {'mIoU':>8}")
    print("-" * 40)
    for epoch in range(1, cfg['n_epochs'] + 1):
        if epoch >= cfg['mining_start_epoch'] and epoch % 5 == 0:
            update_sample_weights(classifier, backbone, train_ds, device, cfg)
            sampler = WeightedRandomSampler(train_ds.sample_weights, len(train_ds.sample_weights), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], sampler=sampler, num_workers=0)
        classifier.train()
        t_losses = []
        for imgs, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch:02d} train", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.no_grad():
                feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits  = classifier(feats)
            outputs = F.interpolate(logits, size=(cfg['img_h'], cfg['img_w']), mode="bilinear", align_corners=False)
            labels  = masks.squeeze(1).long()
            loss    = criterion(outputs, labels)
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            t_losses.append(loss.item())
        scheduler.step()
        classifier.eval()
        v_losses, all_iou = [], []
        with torch.no_grad():
            for imgs, masks, _ in tqdm(val_loader, desc=f"Epoch {epoch:02d} val  ", leave=False):
                imgs, masks = imgs.to(device), masks.to(device)
                feats   = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits  = classifier(feats)
                outputs = F.interpolate(logits, size=(cfg['img_h'], cfg['img_w']), mode="bilinear", align_corners=False)
                labels  = masks.squeeze(1).long()
                v_losses.append(criterion(outputs, labels).item())
                miou, ciou = compute_iou(outputs, labels)
                all_iou.append((miou, ciou))
        t_loss   = float(np.mean(t_losses))
        v_loss   = float(np.mean(v_losses))
        mean_iou = float(np.nanmean([x[0] for x in all_iou]))
        class_iou = [float(np.nanmean([x[1][c] for x in all_iou])) for c in range(n_classes)]
        print(f"{epoch:>6} {t_loss:>9.4f} {v_loss:>9.4f} {mean_iou:>8.4f}" + (" BEST!" if mean_iou > best_iou else ""))
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(classifier.state_dict(), cfg['model_path'])
        log["epochs"].append(epoch)
        log["train_loss"].append(round(t_loss,4))
        log["val_loss"].append(round(v_loss,4))
        log["mean_iou"].append(round(mean_iou,4))
        log["per_class_iou"].append([round(v,4) if not np.isnan(v) else None for v in class_iou])
        with open(cfg['log_file'], "w") as f:
            json.dump(log, f, indent=2)
        for k,v in [('train_loss',t_loss),('val_loss',v_loss),('train_iou',mean_iou),('val_iou',mean_iou),('train_dice',mean_iou),('val_dice',mean_iou),('train_pixel_acc',mean_iou),('val_pixel_acc',mean_iou)]:
            history[k].append(v)
    save_plots(history, cfg['output_dir'])
    print(f"\nTraining complete! Best mIoU: {best_iou:.4f}")
    print(f"Model saved: {cfg['model_path']}")
    print(f"Now run: python test_segmentation.py")

if __name__ == "__main__":
    main()
