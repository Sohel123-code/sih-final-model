---
title: SAR Oil Spill Detector & Thickness Classifier
emoji: 🛰️
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
license: mit
---

# SAR Oil Spill Detector & Thickness Classifier

A **Hierarchical Two-Stage Deep Learning Pipeline** (CNN + Swin Transformer hybrid backbones) for automated detection and thickness characterization of marine oil spills from **Sentinel-1 SAR** (Synthetic Aperture Radar) imagery.

## Hierarchical Pipeline Architecture

```
                       Input SAR Image (224×224)
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │   Stage 1: Oil Detector      │
                   │   (ResNet-18 + Swin-Tiny)    │
                   └──────────────┬───────────────┘
                                  │
                   ┌──────────────┴───────────────┐
                   ▼                              ▼
             [NO OIL DETECTED]             [OIL DETECTED]
           (Clean Sea / Lookalike)                │
                                                  ▼
                                   ┌──────────────────────────────┐
                                   │  Stage 2: Thickness Model    │
                                   │   (ResNet-18 + Swin-Tiny)    │
                                   └──────────────┬───────────────┘
                                                  │
                            ┌─────────────────────┼─────────────────────┐
                            ▼                     ▼                     ▼
                       Thin_Sheen             Moderate           Thick_Emulsified
```

## Features
- **Two-Stage Hierarchical Inference**: Prevents false positive thickness assignments on clean sea or look-alike images.
- **Single & Batch Processing**: Analyze single SAR scenes or batch upload multiple images at once.
- **Detailed Confidence Breakdown**: Stage 1 oil probability and Stage 2 thickness class distribution.

