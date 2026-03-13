# AdaptSeg 🏜️
Uncertainty-Guided Semantic Segmentation for Off-road Autonomy using DINOv2 backbone with desert domain-shift augmentation — Duality AI Hackathon 2026


### Duality AI Offroad Segmentation Hackathon 2026

## What We Built
A semantic segmentation model that identifies 10 desert classes 
in unseen environments using uncertainty-guided hard mining.

## Our Approach
- DINOv2 backbone (Facebook AI) for powerful feature extraction
- Uncertainty Mining — model studies its own weak spots every 5 epochs
- Desert augmentation — sandstorm simulation, sun-angle variation
- Trained on 2857 images, tested on 1002 unseen images

## Classes
Background, Trees, Lush Bushes, Dry Grass, Dry Bushes,
Ground Clutter, Logs, Rocks, Landscape, Sky

## Results
- Training mIoU: 0.3490
- Test mIoU: 0.2519

## How to Run
conda activate EDU
python train_segmentation.py
python test_segmentation.py

## Tech Stack
PyTorch | DINOv2 | Python 3.10
