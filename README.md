# Lightweight Transfer Learning for Automated Tuberculosis Screening from Chest Radiographs

This repository contains the code accompanying the paper:

> **Lightweight Transfer Learning for Automated Tuberculosis Screening from Chest Radiographs**
> Dedi Irawan & Sudarmaji, Faculty of Computer Science, Universitas Muhammadiyah Metro, Lampung, Indonesia.
> Submitted to *Journal of ICT Research and Applications (J. ICT Res. Appl.)*, Institut Teknologi Bandung.

The study compares five pre-trained CNN architectures (ResNet50, EfficientNetB0, MobileNetV2, DenseNet121, VGG16) for tuberculosis (TB) detection on chest X-ray images, with a focus on **decision-threshold optimization to prioritize recall on the minority TB class**, plus analyses of model efficiency and multi-seed robustness.

---

## Key idea

On imbalanced TB datasets, a high accuracy at the default decision threshold (0.5) can hide a large number of **missed TB cases** (false negatives) — clinically the most dangerous error. This project shows how tuning the decision threshold on a validation set substantially improves TB recall, and reports results across multiple random seeds (mean ± standard deviation) together with a statistical comparison.

## Repository structure

```
tb-screening-repo/
├── README.md                  <- this file
├── LICENSE                    <- MIT license
├── requirements.txt           <- Python dependencies
├── .gitignore
├── notebooks/                 <- ready-to-run Google Colab notebooks
│   ├── 1_TB_Detection.ipynb            (single + 5-model pipeline)
│   ├── 2_TB_Detection_threshold.ipynb  (5 models + threshold optimization)
│   ├── 3_TB_MultiSeed_3seed.ipynb      (3-seed robustness + significance)
│   └── 4_TB_MultiSeed_5seed.ipynb      (full 5-seed version)
├── src/
│   └── tb_pipeline.py         <- reusable training/evaluation functions
└── docs/
    └── USAGE.md               <- step-by-step instructions
```

## Dataset

The experiments use the public **TB Chest Radiography Database** (4,200 images: 3,500 Normal + 700 TB):

> Rahman, T. et al., "Reliable Tuberculosis Detection Using Chest X-ray with Deep Learning, Segmentation and Visualization," *IEEE Access*, 8, pp. 191586–191601, 2020. DOI: 10.1109/ACCESS.2020.3031384

Available on Kaggle: `tawsifurrahman/tuberculosis-tb-chest-xray-dataset`

> **Note:** The dataset is *not* included in this repository (it is large and has its own license). The notebooks download it automatically.

## Quick start (Google Colab — recommended)

1. Open any notebook in the `notebooks/` folder in [Google Colab](https://colab.research.google.com/).
2. Set the runtime to **GPU** (Runtime → Change runtime type → T4 GPU).
3. Run the cells top to bottom. The dataset downloads automatically.

The multi-seed notebooks save each (model, seed) result to Google Drive, so they **resume automatically** if the Colab session disconnects.

## Quick start (local)

```bash
pip install -r requirements.txt
# then run the notebooks with Jupyter, or import functions from src/tb_pipeline.py
```

## Main results (3-seed, mean ± std)

| Model          | Accuracy        | Recall (TB)     | F1              | AUC             |
|----------------|-----------------|-----------------|-----------------|-----------------|
| EfficientNetB0 | 0.9899 ± 0.0056 | 0.9845 ± 0.0108 | 0.9711 ± 0.0154 | 0.9990 ± 0.0011 |
| VGG16          | 0.9836 ± 0.0064 | 0.9713 ± 0.0076 | 0.9503 ± 0.0217 | 0.9987 ± 0.0011 |
| MobileNetV2    | 0.9746 ± 0.0042 | 0.8825 ± 0.0234 | 0.9203 ± 0.0147 | 0.9856 ± 0.0076 |
| ResNet50       | 0.9709 ± 0.0064 | 0.9221 ± 0.0479 | 0.9154 ± 0.0139 | 0.9948 ± 0.0016 |
| DenseNet121    | 0.9455 ± 0.0064 | 0.8131 ± 0.0654 | 0.8344 ± 0.0270 | 0.9695 ± 0.0034 |

*Reproducibility note: exact numbers depend on the random seeds and the library versions; small variations are expected.*

## Citation

If you use this code, please cite the paper (details to be updated upon publication) and the dataset (Rahman et al., 2020).

## License

Released under the MIT License (see `LICENSE`). The dataset retains its own license from the original authors.
