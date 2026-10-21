# Brain Tumor Detection

CNN that classifies brain MRI images as tumor (`yes`) or no tumor (`no`). Built with Keras, written to run in Google Colab.

[Open in Colab](https://colab.research.google.com/github/GirishNaik711/CodeClauseInternship_Brain_Tumor_Classification/blob/main/Brain_Tumor_detection.ipynb)

## Requirements

- Python 3 with TensorFlow/Keras 2.x
- pandas, numpy, matplotlib, scikit-learn, opencv-python

```
pip install tensorflow pandas numpy matplotlib scikit-learn opencv-python
```

## Data

The notebook expects an `archive.zip` containing three folders of `.jpg` images:

| Folder | Images | Use |
|--------|--------|-----|
| `yes/` | 1500 | tumor, labeled |
| `no/` | 1500 | no tumor, labeled |
| `pred/` | 60 | unlabeled, used only for the final predictions |

The 3000 labeled images are shuffled and split 90/10 into train and test (2700 / 300). 10% of the train part is held out for validation (2430 train / 270 validation). Images are loaded as 200x200 grayscale and scaled to 0-1.

## Usage

1. Open the notebook in Colab.
2. Upload `archive.zip` to `/content`.
3. Run all cells.

The paths are hard-coded to `/content/yes`, `/content/no` and `/content/pred`. To run locally, change them in the "Path Initialization" cells.

## Models

Both are `Sequential` CNNs with a 512-unit dense layer and a 2-way softmax, trained with RMSprop (lr 0.001) and categorical cross-entropy.

| | Model 1 | Model 2 |
|---|---|---|
| Conv blocks | 4 (32, 64, 128, 256) | 5 (32, 64, 128, 128, 256) |
| Dropout | 0.2 after each block, 0.5 before dense | 0.5 before dense only |
| Parameters | 13.5M | 2.6M |
| Epochs | 10 | 50 |
| Test accuracy | 0.97 (loss 0.08) | 0.99 (loss 0.18) |

Model 2's validation loss climbs over the 50 epochs (about 0.09 at epoch 10, 0.48 at epoch 50) while training accuracy keeps rising, so it is overfitting.
