import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

script_dir = os.path.dirname(os.path.abspath(__file__))

value_map = {0:0,100:1,200:2,300:3,500:4,550:5,700:6,800:7,7100:8,10000:9}
n_classes = len(value_map)
class_names = ['Background','Trees','Lush Bushes','Dry Grass','Dry Bushes','Ground Clutter','Logs','Rocks','Landscape','Sky']
color_palette = np.array([[0,0,0],[34,139,34],[0,255,0],[210,180,140],[139,90,43],[128,128,0],[139,69,19],[128,128,128],[160,82,45],[135,206,235]],dtype=np.uint8)

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw, new in value_map.items():
        new_arr[arr == raw] = new
    return Image.fromarray(new_arr)

def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(n_classes):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask

class TestDataset(Dataset):
    def __init__(self, data_dir, transform, w, h):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.seg_dir   = os.path.join(data_dir, 'Segmentation')
        self.has_masks = os.path.exists(self.seg_dir)
        self.w, self.h = w, h
        if os.path.exists(self.image_dir):
            self.data_ids = sorted(os.listdir(self.image_dir))
        else:
            self.data_ids = sorted([f for f in os.listdir(data_dir) if f.endswith(('.png','.jpg','.jpeg'))])
            self.image_dir = data_dir
            self.has_masks = False
        self.transform = transform
        print(f"Found {len(self.data_ids)} test images")
    def __len__(self):
        return len(self.data_ids)
    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        img_t = self.transform(image)
        if self.has_masks:
            mask_path = os.path.join(self.seg_dir, data_id)
            if os.path.exists(mask_path):
                mask = convert_mask(Image.open(mask_path))
                mask = mask.resize((self.w, self.h), Image.NEAREST)
                mask_t = torch.from_numpy(np.array(mask)).long()
                return img_t, mask_t, data_id
        return img_t, torch.zeros(1).long(), data_id

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem  = nn.Sequential(nn.Conv2d(in_channels, 128, 7, padding=3), nn.GELU())
        self.block = nn.Sequential(nn.Conv2d(128, 128, 7, padding=3, groups=128), nn.GELU(), nn.Conv2d(128, 128, 1), nn.GELU())
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
        ious.append((tp/d).item() if d > 0 else float('nan'))
    return float(np.nanmean(ious)), ious

def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")

    w = int(((960/2)//14)*14)
    h = int(((540/2)//14)*14)

    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    data_dir   = os.path.join(script_dir, '..', 'Offroad_Segmentation_testImages')
    model_path = os.path.join(script_dir, 'segmentation_head.pth')
    output_dir = os.path.join(script_dir, 'predictions')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks_color'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)

    dataset = TestDataset(data_dir, transform, w, h)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print("Loading DINOv2...")
    torch.hub._validate_not_a_forked_repo = lambda a,b,c: True
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval().to(device)

    with torch.no_grad():
        sample, _, _ = dataset[0]
        feats = backbone.forward_features(sample.unsqueeze(0).to(device))["x_norm_patchtokens"]
    n_emb = feats.shape[2]

    classifier = SegmentationHeadConvNeXt(n_emb, n_classes, w//14, h//14)
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.eval().to(device)
    print("Model loaded!")

    all_ious = []
    with torch.no_grad():
        for imgs, masks, data_ids in tqdm(loader, desc="Testing"):
            imgs = imgs.to(device)
            feats   = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits  = classifier(feats)
            outputs = F.interpolate(logits, size=(h,w), mode="bilinear", align_corners=False)
            pred    = torch.argmax(outputs, dim=1)[0].cpu().numpy().astype(np.uint8)

            pred_color = mask_to_color(pred)
            cv2.imwrite(os.path.join(output_dir,'masks_color',f"{data_ids[0]}_pred.png"),
                       cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

            img_np = imgs[0].cpu().numpy()
            img_np = np.moveaxis(img_np,0,-1)
            img_np = img_np * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])
            img_np = np.clip(img_np*255,0,255).astype(np.uint8)
            comparison = np.hstack([img_np, pred_color])
            cv2.imwrite(os.path.join(output_dir,'comparisons',f"{data_ids[0]}_comparison.png"),
                       cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

            if masks.dim() > 1 and masks.sum() > 0:
                labels = masks.to(device)
                miou, ciou = compute_iou(outputs, labels)
                all_ious.append(miou)

    if all_ious:
        final_miou = float(np.nanmean(all_ious))
        print(f"\nFinal mIoU: {final_miou:.4f}")
        with open(os.path.join(output_dir,'final_iou.txt'),'w') as f:
            f.write(f"Final mIoU: {final_miou:.4f}\n")
            for i,(name,iou) in enumerate(zip(class_names,[np.nanmean([x[1][i] for x in [(miou,ciou)]]) for i in range(n_classes)])):
                f.write(f"  {name}: {iou:.4f}\n")
    print(f"\nDone! Results saved to {output_dir}/")
    print(f"  masks_color/  - coloured predictions")
    print(f"  comparisons/  - side by side images")

if __name__ == "__main__":
    main()
