# Project 2 Design Notes

## Objective

Build a skin lesion classification pipeline for three classes:

- `mel`
- `nv`
- `vasc`

The final submission code must read test samples from:

- `./image`
- `./mask`

and generate `output.csv` with two columns:

```csv
image_id,dx
```

`image_id` is the test image id, and `dx` is the predicted class.

## Data

The training data is stored in `project2/Data_Proj2`:

- `image/`: RGB lesion images
- `mask/`: corresponding binary lesion masks
- `label.csv`: ground-truth labels

The dataset contains original images and augmented images. The test data is also expected to contain augmented samples, so the pipeline should avoid fragile assumptions about image order and should rely on the documented image/mask naming convention.

## Implementation Path

The project is organized as a staged image classification pipeline:

1. Load images, masks, and labels.
2. Preprocess images by applying lesion masks.
3. Normalize RGB values using lesion-region statistics.
4. Extract interpretable lesion features.
5. Analyze feature quality with dimensionality reduction and model-based importance.
6. Train and validate a classifier.
7. Export the final prediction workflow to `Submitdemo_Proj2/demo.ipynb`.
8. Generate `output.csv` in the required format.

## Current Scope

The current implementation focuses on the first stage:

- image reading
- mask reading
- lesion masking
- RGB normalization
- visual validation of preprocessing

This stage is implemented in:

```text
project2/Data_Proj2/evaluate.ipynb
```

## Completed Components

### Image and Mask Loading

Implemented functions:

```python
natural_key(path)
find_data_dir()
read_images_and_masks(data_dir=None, image_size=(224, 224))
```

Design details:

- Images are sorted with natural numeric ordering to avoid lexical order issues such as `1, 10, 100, 2`.
- The data directory is resolved automatically so the notebook can be run either from the repository root or from `project2/Data_Proj2`.
- Images are converted to RGB arrays.
- Masks are converted to grayscale and thresholded into boolean arrays.
- Images and masks are resized to `(224, 224)` for consistent downstream processing.

Outputs:

```python
image_ids
images
masks
```

Expected formats:

- `image_ids`: list of image ids such as `1`, `1_aug1`
- `images`: list of `uint8` RGB arrays with shape `(224, 224, 3)`
- `masks`: list of boolean arrays with shape `(224, 224)`

### Mask-Based Preprocessing

Implemented function:

```python
apply_masks(images, masks, background_value=0)
```

Design details:

- Only pixels inside the lesion mask are preserved.
- Background pixels are set to `background_value`, currently `0`.
- This reduces the influence of surrounding skin/background when extracting lesion features.

Output:

```python
images_masked
```

### RGB Normalization

Implemented function:

```python
normalize_images(images_masked, masks, eps=1e-6)
```

Design details:

- Normalization statistics are computed only from lesion pixels.
- RGB values are first scaled from `[0, 255]` to `[0, 1]`.
- Mean and standard deviation are computed independently for the three RGB channels.
- Background pixels remain `0` after normalization.

Core computation:

```python
lesion_pixels = np.concatenate([
    image[mask].astype(np.float32) / 255.0
    for image, mask in zip(images_masked, masks)
    if np.any(mask)
])

mean = lesion_pixels.mean(axis=0)
std = lesion_pixels.std(axis=0) + eps
```

Since `lesion_pixels` has shape `(num_lesion_pixels, 3)`, both `mean` and `std` have shape `(3,)`, corresponding to RGB channels.

Output:

```python
images_normalized
normalization_stats
```

where:

```python
normalization_stats = {
    "mean": mean,
    "std": std,
}
```

### Preprocessing Visualization

Implemented function:

```python
denormalize_for_display(image_normalized, mask, mean, std)
```

Purpose:

- Standardized RGB values are not valid display colors.
- Directly displaying z-score normalized images may create misleading colors.
- For visual validation, normalized images are converted back to RGB-like `[0, 1]` values before display.

Core computation:

```python
display[mask] = image_normalized[mask] * std + mean
display = np.clip(display, 0, 1)
```

The test cell displays:

1. Original image
2. Binary mask
3. Masked image
4. Denormalized image

The denormalized image should visually match the masked image. This confirms that RGB normalization and inverse transformation are consistent.

## Feature Engineering Plan

The next stage is to build a feature table from the masked lesion region.

Candidate feature groups:

- RGB channel mean, standard deviation, median, and percentiles
- HSV channel statistics
- lesion area ratio
- color variance and color contrast
- darkness/brightness statistics
- red/blue dominance indicators
- simple shape features derived from the mask

Feature extraction should produce one row per image:

```text
image_id, feature_1, feature_2, ..., feature_n, dx
```

The label `dx` is only used for training/evaluation and must not be required for final test prediction.

## Feature Evaluation Plan

Feature quality should not be judged only by raw two-dimensional scatter plots, because such plots only show two features at a time and can be visually misleading.

Recommended evaluation methods:

- PCA for linear dimensionality-reduction visualization
- t-SNE for local cluster visualization
- UMAP if the environment supports `umap-learn`
- Random Forest feature importance for ranking features
- validation accuracy and confusion matrix for quantitative evaluation

The final decision should prioritize validation performance and robustness on augmented samples, not only visual separation.

## Modeling Plan

Candidate classifiers:

- Random Forest
- SVM
- KNN
- Logistic Regression

The preferred first model is Random Forest because it handles mixed feature scales reasonably well, provides feature importance, and works well with small tabular datasets.

If the final execution environment lacks `scikit-learn`, a lightweight fallback classifier can be implemented with `numpy`, such as:

- nearest centroid classifier
- k-nearest neighbors

## Dependencies

Current preprocessing stage:

```bash
pip install numpy pillow matplotlib
```

Recommended for modeling and evaluation:

```bash
pip install pandas scikit-learn
```

Optional for UMAP visualization:

```bash
pip install umap-learn
```

## Next Tasks

1. Load `label.csv` and align labels with `image_ids`.
2. Implement lesion feature extraction.
3. Save a feature table for inspection.
4. Add PCA/t-SNE visualization.
5. Train a baseline classifier.
6. Evaluate with validation split and confusion matrix.
7. Move the final prediction pipeline into `Submitdemo_Proj2/demo.ipynb`.
