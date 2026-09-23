# Industrial Anomaly Detection & Localization

Pixel-level anomaly detection and localization pipeline built for the ADL Challenge at Politecnico di Milano. The task involves detecting defective regions across multi-view industrial images and producing dense anomaly score maps.

## Overview

- **Segmentation Models:** Trained UNet++, DeepLabV3+, and FPN architectures with ImageNet-pretrained encoders on real and synthetic anomaly masks.
- **Normality Baselines:** Integrated feature-space detectors (PatchCore and PaDiM) to capture subtle deviations without relying on masks.
- **Ensembling & Blending:** Applied class-specific micro-blending to weight model contributions dynamically based on object class.
- **Results:** Achieved a top private leaderboard score of **0.9000** with the specialist blend pipeline.

## Team

Merve Rana Kızıl, İsmail Emre Gümüş, Neslihan Pelin Metin, , Samet Aydın.
