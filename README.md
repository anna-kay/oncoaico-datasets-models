## 1. Overview
This repository contains datasets, code, and experimental setups for machine learning tasks focused on segmentation and detection in colonoscopy data.

## 2. Data
**Pre-existing datasets used in this repo:**

1. CVC-300
2. CVC-ColonDB
3. CVC-ClinicDB
4. ETIS-LaribPolypDB
5. Kvasir SEG (HyperKvasir), [paper](https://www.nature.com/articles/s41597-020-00622-y), [github](https://github.com/DebeshJha/Kvasir-SEG), [link](https://datasets.simula.no/hyper-kvasir/)
6. PolypGen, [paper](https://www.nature.com/articles/s41597-023-01981-y), [github](https://github.com/DebeshJha/PolypGen), [link](https://www.synapse.org/Synapse:syn26376615/wiki/)
7. LDPolypVideo, [paper](), [link]()
8. REAL-COLON, [paper](https://www.nature.com/articles/s41597-024-03359-0), [link](https://plus.figshare.com/articles/media/REAL-colon_dataset/22202866)

**How the datasets are used & Data processing:**

The development of ONCO-AICO faced several data-related challenges. Although multiple colonoscopy datasets are available, many lack segmentation annotations, while those that include them are often small, limiting deep learning performance. Most datasets also do not provide predefined train–validation–test splits, increasing the risk of data leakage and reducing experimental consistency. Additionally, data is typically sourced from a limited number of clinical environments, constraining model generalization.

Given that ONCO-AICO is designed for training on colonoscopy videos, it must handle realistic conditions where most frames contain no findings (negative samples). However, existing datasets primarily consist of positive samples, which can lead to false positives during inference.

Attempts to incorporate additional datasets (e.g., CVC-300, CVC-ColonDB, ETIS-LaribPolypDB) were hindered by the absence of official data splits, making their use less reliable. Moreover, no available dataset explicitly included both positive and negative samples.

To address these limitations, custom datasets were constructed:

- **Segmentation Dataset**: Combines multiple sources to improve generalization.
- **Mixed Dataset**: Extends the segmentation dataset with negative (healthy) samples, enabling more realistic training and reducing false positives.

<br>

| Dataset | Train | Validation | Test | Total |
| :------------: | :------------: | :------------: | :------------: | :------------: |
| Kvasir-SEG| 900 images | 100 | - | |
| CVC-ClinicDB | 600 | 62 | - | |
| CVC-300 | - | - | 60 | |
| CVC-ColonDB | - | - | 380 | |
| ETIS-LaribPolypDB | - | -| 196 | |
| PolypGen | 1122 | 289 | 342 |
| Segmentation Dataset | 2622 | 451 | - 978 | |
<br>

<br>

| | Dataset | Train | Validation | Test | Total |
| :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | 
| Positive | Segmentation Dataset| 2622 | 451 | 978 | 4051 |
| Negative | REAL-Colon | 1423 | 203 | 408 | 2034 | 
| Negative |  LDPolypVideo | 1429 | 204 | 409 | 2042 | 
| | Mixed Dataset | 5474 | 858 | 1795 | 8127 |
<br>


## 3. Models
Models used in this repo:
1. **CaraNet**, [paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12032/120320D/CaraNet--context-axial-reverse-attention-network-for-segmentation-of/10.1117/12.2611802.short), [github](https://github.com/AngeLouCN/CaraNet)
2. **Polyp-PVT**, [paper](https://arxiv.org/abs/2108.06932), [github](https://github.com/dengpingfan/polyp-pvt)
3. **MSRF-Net**, [paper](https://ieeexplore.ieee.org/abstract/document/9662196/), [github](https://github.com/NoviceMAn-prog/MSRF-Net)


## 4. Project Structure
```
|-
|-
|-

```

## 5. How to run

Lorem Ipsum

## 6. Acknowledgement
This repo relies on the works CaraNet, Polpy-PVT, MSRF-Net

## 7. Context

This work is related to ONCO-AICO is an AI-assisted training platform for junior endoscopists to improve polyp detection skills. The platform utilizes annotated colonoscopy videos, along with explainable AI (xAI) feedback and performance scoring.

Note: This repo focuses solely on the underlying datasets and machine learning experiments.

This work was conducted as part of the EU-funded ONCOSCREEN project (Horizon Europe grant agreement No. 101097036).
