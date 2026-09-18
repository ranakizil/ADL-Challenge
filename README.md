# Challenge of the PoliMi course of Advanced Deep Learning, 2026

## Anomaly Detection & Localization Challenge

In `notebooks/ADL_Final_Best_Pipeline.ipynb` you can find the complete implementation of multi-view industrial image anomaly detection and pixel-level localization for the ADL challenge.

| Test Image | Detected Anomaly Heatmap |
| :---: | :---: |
| <img src="images/test_image.png" width="350"/> | <img src="images/detected_anomaly.png" width="350"/> |

## Project Structure

* **Task 1: Synthetic Anomaly Generation & Data Preprocessing**

  * Handles dataset across 8 distinct industrial object classes (`class_01` to `class_08`) containing clean images and limited real anomaly masks.
  * Generates synthetic anomaly samples on-the-fly to increase training sample diversity and bridge the gap between small training sets and test distributions.
  * Applies ImageNet-pretrained normalization, random spatial transformations, and image resizing (`320px` and `352px` resolutions).

* **Task 2: Supervised Semantic Segmentation & Model Backbones**

  * Trains multiple segmentation architectures using AdamW optimizer and cosine learning rate scheduling:
    * `UNet++` with `EfficientNet-B3` and `EfficientNet-B4` encoders
    * `DeepLabV3+` with `ResNet50` encoder
    * `FPN` with `ResNet34`, `ResNet18`, and `EfficientNet-B4` encoders
  * Outputs two-dimensional anomaly probability maps $\hat{M} = f_\theta(x)$ per test image.

* **Task 3: Normality-Based Anomaly Scoring**

  * Extracts intermediate feature maps from clean training samples using `PatchCore` and `PaDiM` feature-space models.
  * Computes patch-level distance metrics at inference time to detect deviations from normal feature distributions without depending on synthetic masks.

* **Task 4: Class-Specific Micro-Blending & Submission Encoding**

  * Combines complementary model maps into a weighted ensemble average $M_{\text{final}} = \sum_{k=1}^K w_k M_k$.
  * Applies class-specific micro-blending using specialist models (e.g., `Run41` FPN ResNet18 specialist on `class_01`–`class_05`).
  * Encodes the final 2D anomaly heatmaps into the required `q8rle` submission format for leaderboard evaluation.
