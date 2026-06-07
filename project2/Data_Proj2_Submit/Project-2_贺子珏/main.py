#!/usr/bin/env python3
"""Command-line inference script for BIP Project 2.

This file is the script version of submit.ipynb. It loads a saved two-stage
skin-lesion classifier, extracts the same features used during training, and
writes predictions to a CSV file.
"""

from pathlib import Path
from collections.abc import Sequence

import cv2
import numpy as np
import pandas as pd
from scipy import stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.segmentation import slic
from sklearn.metrics import accuracy_score

# =============================================================================
# Section 1: image loading helper functions
# =============================================================================
# helper functions for reading image-mask pairs
def collect_image_paths(image_dir: str | Path, suffixes: tuple[str, ...] = ('.jpg', '.jpeg', '.png')) -> list[Path]:
    """Collect input image files in a deterministic order."""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f'Image directory does not exist: {image_dir}')
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in suffixes)


def find_mask_path(mask_dir: str | Path, image_name: str) -> Path:
    """Find a mask file for one image using the project naming convention."""
    mask_dir = Path(mask_dir)
    candidates = [
        mask_dir / f'mask_{image_name}',
        mask_dir / image_name,
        mask_dir / f'{Path(image_name).stem}.png',
        mask_dir / f'mask_{Path(image_name).stem}.png',
        mask_dir / f'{Path(image_name).stem}.jpg',
        mask_dir / f'mask_{Path(image_name).stem}.jpg',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'Cannot find mask for image {image_name} in {mask_dir}')


def load_image_mask_pairs(
    image_dir: str | Path,
    mask_dir: str | Path,
    max_images: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Load BGR images, grayscale masks, and filenames for inference."""
    image_paths = collect_image_paths(image_dir)
    if max_images is not None:
        image_paths = image_paths[:int(max_images)]

    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    filenames: list[str] = []
    failed_files: list[str] = []

    for image_path in image_paths:
        try:
            mask_path = find_mask_path(mask_dir, image_path.name)
        except FileNotFoundError:
            failed_files.append(image_path.name)
            continue

        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            failed_files.append(image_path.name)
            continue

        images.append(image)
        masks.append(mask)
        filenames.append(image_path.name)

    print(f'Loaded {len(images)} image-mask pairs from {image_dir}')
    if failed_files:
        print(f'[WARNING] Skipped {len(failed_files)} files without readable image/mask. Example: {failed_files[:5]}')
    if not images:
        raise ValueError('No readable image-mask pairs were loaded.')
    return images, masks, filenames

# =============================================================================
# Section 2: preprocessing functions
# =============================================================================
def shades_of_gray(img: np.ndarray, p: int = 6) -> np.ndarray:
    """Shades of Gray color constancy."""
    img_float = img.astype(np.float32) / 255.0
    b, g, r = cv2.split(img_float)

    l_b = np.power(np.mean(np.power(b, p)), 1.0 / p)
    l_g = np.power(np.mean(np.power(g, p)), 1.0 / p)
    l_r = np.power(np.mean(np.power(r, p)), 1.0 / p)

    if l_b == 0 or l_g == 0 or l_r == 0:
        return img.copy()

    l_max = np.max([l_b, l_g, l_r])
    img_corrected = cv2.merge([
        b * (l_max / l_b) * 255.0,
        g * (l_max / l_g) * 255.0,
        r * (l_max / l_r) * 255.0,
    ])

    return np.clip(img_corrected, 0, 255).astype(np.uint8)


def dull_razor_hair_removal(
    img: np.ndarray,
    kernel_size: int = 10,
    threshold: int = 10,
    inpaint_radius: int = 1,
) -> np.ndarray:
    """Remove hair-like dark structures with the Dull-Razor method."""
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray_scale, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)

    return cv2.inpaint(img, hair_mask, inpaint_radius, cv2.INPAINT_TELEA)


def pipeline_preview(
    input_dir: str | Path,
    output_dir: str | Path,
    progress_interval: int = 100,
    color_power: int = 6,
    median_kernel_size: int = 3,
    hair_kernel_size: int = 10,
    hair_threshold: int = 10,
    inpaint_radius: int = 1,
) -> None:
    """Process jpg images and save only the final preprocessed results."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_path.glob('*.jpg'))
    total_images = len(image_paths)

    if total_images == 0:
        print(f"[ERROR] 在文件夹 '{input_path}' 中没有找到 jpg 图片。")
        return

    print(f"[SUCCESS] 成功检索到 {total_images} 张 jpg 图片，开始批量处理...")

    processed_count = 0
    failed_files: list[str] = []

    for index, image_path in enumerate(image_paths, start=1):
        img = cv2.imread(str(image_path))
        if img is None:
            failed_files.append(image_path.name)
            continue

        try:
            img_cc = shades_of_gray(img, p=color_power)
            img_blur = cv2.medianBlur(img_cc, median_kernel_size)
            img_final = dull_razor_hair_removal(
                img_blur,
                kernel_size=hair_kernel_size,
                threshold=hair_threshold,
                inpaint_radius=inpaint_radius,
            )
            cv2.imwrite(str(output_path / image_path.name), img_final)
            processed_count += 1
        except Exception:
            failed_files.append(image_path.name)
            continue

        if index % progress_interval == 0 or index == total_images:
            print(f"[PROGRESS] 已处理 {index}/{total_images} 张图片")

    print(f"[SUCCESS] 批量处理完成，已保存 {processed_count} 张图片至: {output_path}")
    if failed_files:
        print(f"[WARNING] 跳过 {len(failed_files)} 张读取失败或处理失败的图片。")


def preprocess_image_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    progress_interval: int = 100,
) -> None:
    """Run the same preprocessing pipeline used during training."""
    pipeline_preview(
        input_dir=input_dir,
        output_dir=output_dir,
        progress_interval=progress_interval,
    )

# =============================================================================
# Section 3: feature extraction functions
# =============================================================================
# functions for Shape features and asymmetry features

def _to_binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary


def extract_lesion_feature_shape(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    eps: float = 1e-8,
) -> tuple[np.ndarray, list[str]]:
    """Extract lesion shape features from masks."""
    feature_rows: list[list[float]] = []
    feature_names = [
        "shape_area",
        "shape_perimeter",
        "shape_circularity",
        "shape_max_diameter",
        "shape_equiv_diameter",
        "shape_aspect_ratio",
        "shape_eccentricity",
        "shape_compactness",
        "shape_solidity",
        "shape_rectangularity",
        "shape_elongation",
        "shape_defect_ratio",
    ]

    for mask in masks:
        thresh = _to_binary_mask(mask)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0 or np.sum(thresh) == 0:
            feature_rows.append([0.0] * len(feature_names))
            continue

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        circularity = (4.0 * np.pi * area) / (perimeter ** 2 + eps)

        points = contour.reshape(-1, 2)
        if len(points) > 1:
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            max_diameter = float(np.sqrt(np.max(np.sum(diff ** 2, axis=-1))))
        else:
            max_diameter = 0.0

        equiv_diameter = float(np.sqrt(4.0 * area / np.pi)) if area > 0 else 0.0

        if len(contour) >= 5:
            (_, _), (d1, d2), _ = cv2.fitEllipse(contour)
            major_axis = max(d1, d2)
            minor_axis = min(d1, d2)
            aspect_ratio = major_axis / (minor_axis + eps)
            eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor_axis ** 2) / (major_axis ** 2 + eps))))
        else:
            aspect_ratio = 1.0
            eccentricity = 0.0

        compactness = equiv_diameter / (max_diameter + eps)
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / (hull_area + eps)
        defect_ratio = max(0.0, hull_area - area) / (area + eps)

        (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(contour)
        bbox_w = max(rect_w, rect_h)
        bbox_h = min(rect_w, rect_h)
        rectangularity = area / (bbox_w * bbox_h + eps)
        elongation = bbox_h / (bbox_w + eps)

        feature_rows.append([
            area,
            perimeter,
            circularity,
            max_diameter,
            equiv_diameter,
            aspect_ratio,
            eccentricity,
            compactness,
            solidity,
            rectangularity,
            elongation,
            defect_ratio,
        ])

    return np.asarray(feature_rows, dtype=np.float32), feature_names


def extract_lesion_feature_asymmetry(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    eps: float = 1e-8,
) -> tuple[np.ndarray, list[str]]:
    """Extract lesion asymmetry features from masks."""
    feature_rows: list[list[float]] = []
    feature_names = [
        "asymmetry_area_ratio",
        "asymmetry_x_axis",
        "asymmetry_xy_sum",
        "asymmetry_rotation",
        "asymmetry_fullness",
    ]

    for mask in masks:
        thresh = _to_binary_mask(mask)
        total_area = float(np.sum(thresh == 255))

        if total_area == 0:
            feature_rows.append([0.0] * len(feature_names))
            continue

        moments = cv2.moments(thresh)
        if moments["m00"] != 0:
            cx = int(round(moments["m10"] / moments["m00"]))
            cy = int(round(moments["m01"] / moments["m00"]))
        else:
            cx, cy = thresh.shape[1] // 2, thresh.shape[0] // 2

        diff_x = cv2.absdiff(thresh, cv2.flip(thresh, 1))
        diff_y = cv2.absdiff(thresh, cv2.flip(thresh, 0))
        area_diff_x = float(np.sum(diff_x == 255))
        area_diff_y = float(np.sum(diff_y == 255))

        asymmetry_x_axis = (area_diff_x / total_area) * 100.0
        asymmetry_xy_sum = ((area_diff_x + area_diff_y) / total_area) * 100.0

        left_area = float(np.sum(thresh[:, :cx] == 255))
        right_area = float(np.sum(thresh[:, cx:] == 255))
        asymmetry_area_ratio = (abs(left_area - right_area) / total_area) * 100.0

        h, w = thresh.shape
        rot_mat = cv2.getRotationMatrix2D((cx, cy), 180, 1.0)
        rotated_mask = cv2.warpAffine(thresh, rot_mat, (w, h), flags=cv2.INTER_NEAREST)
        fs_mask = cv2.bitwise_xor(thresh, rotated_mask)
        a_mask = cv2.bitwise_or(thresh, rotated_mask)
        asymmetry_rotation = 1.0 - (float(np.sum(fs_mask == 255)) / (float(np.sum(a_mask == 255)) + eps))

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if len(contour) >= 5:
                (_, _), (d1, d2), _ = cv2.fitEllipse(contour)
                equiv_ellipse_area = np.pi * (d1 / 2.0) * (d2 / 2.0)
                asymmetry_fullness = equiv_ellipse_area / (total_area + eps)
            else:
                asymmetry_fullness = 1.0
        else:
            asymmetry_fullness = 0.0

        feature_rows.append([
            asymmetry_area_ratio,
            asymmetry_x_axis,
            asymmetry_xy_sum,
            asymmetry_rotation,
            asymmetry_fullness,
        ])

    return np.asarray(feature_rows, dtype=np.float32), feature_names


# functions for enhanced shape and border-complexity features: Shape_plus

_SHAPE_PLUS_FEATURE_NAMES = [
    'shape_plus_fractal_dimension',
    'shape_plus_boundary_gradient_mean',
    'shape_plus_boundary_gradient_std',
    'shape_plus_boundary_gradient_p90',
    'shape_plus_radial_distance_mean',
    'shape_plus_radial_distance_std',
    'shape_plus_radial_distance_cv',
    'shape_plus_radial_roughness',
]


def _largest_contour_from_mask(mask: np.ndarray) -> np.ndarray | None:
    binary = _to_binary_mask(mask)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return None
    return contour


def _contour_edge_mask(mask: np.ndarray) -> np.ndarray:
    binary = _to_binary_mask(mask)
    kernel = np.ones((3, 3), dtype=np.uint8)
    edge = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    return edge > 0


def _box_count_fractal_dimension(edge_mask: np.ndarray, min_box_size: int = 2) -> float:
    """Estimate boundary fractal dimension with box counting."""
    edge = np.asarray(edge_mask, dtype=bool)
    if edge.sum() == 0:
        return 0.0

    h, w = edge.shape
    max_power = int(np.floor(np.log2(min(h, w))))
    sizes = [2 ** p for p in range(max_power, 0, -1) if 2 ** p >= min_box_size]
    counts: list[int] = []
    valid_sizes: list[int] = []

    for size in sizes:
        pad_h = (size - h % size) % size
        pad_w = (size - w % size) % size
        padded = np.pad(edge, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=False)
        blocks = padded.reshape(padded.shape[0] // size, size, padded.shape[1] // size, size)
        count = int(np.any(blocks, axis=(1, 3)).sum())
        if count > 0:
            counts.append(count)
            valid_sizes.append(size)

    if len(counts) < 2:
        return 0.0

    coeffs = np.polyfit(np.log(1.0 / np.asarray(valid_sizes, dtype=float)), np.log(np.asarray(counts, dtype=float)), 1)
    return float(coeffs[0])


def _boundary_gradient_features(image: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    binary = _to_binary_mask(mask)
    if binary.shape[:2] != gray.shape[:2]:
        binary = cv2.resize(binary, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

    edge_mask = _contour_edge_mask(binary)
    grad_x = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    vals = grad_mag[edge_mask]
    if vals.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals)), float(np.percentile(vals, 90))


def _radial_boundary_features(mask: np.ndarray, eps: float = 1e-8) -> tuple[float, float, float, float]:
    contour = _largest_contour_from_mask(mask)
    if contour is None:
        return 0.0, 0.0, 0.0, 0.0

    moments = cv2.moments(contour)
    if moments['m00'] == 0:
        points = contour.reshape(-1, 2).astype(np.float32)
        center = points.mean(axis=0)
    else:
        center = np.array([moments['m10'] / moments['m00'], moments['m01'] / moments['m00']], dtype=np.float32)

    points = contour.reshape(-1, 2).astype(np.float32)
    distances = np.sqrt(np.sum((points - center[None, :]) ** 2, axis=1))
    if distances.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    mean_dist = float(np.mean(distances))
    std_dist = float(np.std(distances))
    cv_dist = std_dist / (mean_dist + eps)

    if distances.size >= 3:
        roughness = float(np.mean(np.abs(np.diff(distances, append=distances[0]))) / (mean_dist + eps))
    else:
        roughness = 0.0

    return mean_dist, std_dist, cv_dist, roughness


def extract_lesion_feature_shape_plus(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Extract enhanced boundary-complexity and border-gradient features."""
    rows: list[list[float]] = []

    for image, mask in zip(images, masks):
        binary = _to_binary_mask(mask)
        if image.shape[:2] != binary.shape[:2]:
            binary = cv2.resize(binary, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        edge_mask = _contour_edge_mask(binary)
        fractal_dimension = _box_count_fractal_dimension(edge_mask)
        grad_mean, grad_std, grad_p90 = _boundary_gradient_features(image, binary)
        radial_mean, radial_std, radial_cv, radial_roughness = _radial_boundary_features(binary)

        rows.append([
            fractal_dimension,
            grad_mean,
            grad_std,
            grad_p90,
            radial_mean,
            radial_std,
            radial_cv,
            radial_roughness,
        ])

    return np.asarray(rows, dtype=np.float32), _SHAPE_PLUS_FEATURE_NAMES.copy()


# functions for shape invariant moment features: Hu moments

_SHAPE_HU_FEATURE_NAMES = [
    f'shape_hu_moment_{idx}'
    for idx in range(1, 8)
]


def _log_transform_hu_moments(hu_moments: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    hu = np.asarray(hu_moments, dtype=np.float64).reshape(-1)
    return -np.sign(hu) * np.log10(np.abs(hu) + eps)


def _compute_single_hu_moment_features(mask: np.ndarray) -> list[float]:
    binary = _to_binary_mask(mask)
    if np.sum(binary > 0) == 0:
        return [0.0] * len(_SHAPE_HU_FEATURE_NAMES)

    moments = cv2.moments((binary > 0).astype(np.uint8))
    hu = cv2.HuMoments(moments).flatten()
    hu_log = _log_transform_hu_moments(hu)
    hu_log = np.nan_to_num(hu_log, nan=0.0, posinf=0.0, neginf=0.0)
    return [float(value) for value in hu_log]


def extract_lesion_feature_hu_moments(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Extract 7 log-transformed Hu invariant moments from lesion masks."""
    rows = [_compute_single_hu_moment_features(mask) for mask in masks]
    return np.asarray(rows, dtype=np.float32), _SHAPE_HU_FEATURE_NAMES.copy()


# functions for principal-axis asymmetry features: Asymmetry_plus

_ASYMMETRY_PLUS_FEATURE_NAMES = [
    'asymmetry_plus_major_axis_xor_pct',
    'asymmetry_plus_minor_axis_xor_pct',
    'asymmetry_plus_major_axis_area_ratio',
    'asymmetry_plus_minor_axis_area_ratio',
    'asymmetry_plus_combined_xor_pct',
    'asymmetry_plus_pca_angle_deg',
    'asymmetry_plus_axis_ratio',
    'asymmetry_plus_centroid_offset_ratio',
]


def _principal_axis_aligned_mask(
    mask: np.ndarray,
    padding: int = 8,
    eps: float = 1e-8,
) -> tuple[np.ndarray, tuple[int, int], float, float, float]:
    """Rasterize lesion pixels in a PCA-aligned coordinate system."""
    binary = _to_binary_mask(mask)
    coords_yx = np.argwhere(binary > 0)
    if coords_yx.shape[0] < 3:
        empty = np.zeros((1, 1), dtype=bool)
        return empty, (0, 0), 0.0, 1.0, 0.0

    points_xy = coords_yx[:, ::-1].astype(np.float32)
    centroid = points_xy.mean(axis=0)
    centered = points_xy - centroid[None, :]
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    major_vec = eigvecs[:, 0]
    angle_deg = float(np.degrees(np.arctan2(major_vec[1], major_vec[0])))
    axis_ratio = float(np.sqrt((eigvals[0] + eps) / (eigvals[1] + eps))) if eigvals.shape[0] > 1 else 1.0

    aligned = centered @ eigvecs
    min_xy = np.floor(aligned.min(axis=0)).astype(int)
    max_xy = np.ceil(aligned.max(axis=0)).astype(int)
    width = int(max_xy[0] - min_xy[0] + 1 + 2 * padding)
    height = int(max_xy[1] - min_xy[1] + 1 + 2 * padding)
    width = max(width, 1)
    height = max(height, 1)

    shifted = np.rint(aligned - min_xy[None, :] + padding).astype(int)
    shifted[:, 0] = np.clip(shifted[:, 0], 0, width - 1)
    shifted[:, 1] = np.clip(shifted[:, 1], 0, height - 1)

    aligned_mask = np.zeros((height, width), dtype=np.uint8)
    aligned_mask[shifted[:, 1], shifted[:, 0]] = 255
    aligned_mask = cv2.morphologyEx(aligned_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    aligned_bool = aligned_mask > 0

    axis_center_x = int(round(-min_xy[0] + padding))
    axis_center_y = int(round(-min_xy[1] + padding))
    axis_center_x = int(np.clip(axis_center_x, 0, width - 1))
    axis_center_y = int(np.clip(axis_center_y, 0, height - 1))

    original_h, original_w = binary.shape
    bbox_center = np.array([original_w / 2.0, original_h / 2.0], dtype=np.float32)
    centroid_offset = float(np.linalg.norm(centroid - bbox_center) / (np.sqrt(original_h ** 2 + original_w ** 2) + eps))

    return aligned_bool, (axis_center_x, axis_center_y), angle_deg, axis_ratio, centroid_offset


def _axis_fold_xor_pct(aligned_mask: np.ndarray, axis: str, center: tuple[int, int], eps: float = 1e-8) -> float:
    """Compute non-overlap percentage after reflecting around one axis through the PCA centroid."""
    if aligned_mask.sum() == 0:
        return 0.0

    h, w = aligned_mask.shape
    cx, cy = center
    reflected = np.zeros_like(aligned_mask, dtype=bool)
    ys, xs = np.nonzero(aligned_mask)

    if axis == 'major':
        reflected_y = 2 * cy - ys
        valid = (reflected_y >= 0) & (reflected_y < h)
        reflected[reflected_y[valid], xs[valid]] = True
    elif axis == 'minor':
        reflected_x = 2 * cx - xs
        valid = (reflected_x >= 0) & (reflected_x < w)
        reflected[ys[valid], reflected_x[valid]] = True
    else:
        raise ValueError("axis must be 'major' or 'minor'.")

    xor_area = float(np.logical_xor(aligned_mask, reflected).sum())
    lesion_area = float(aligned_mask.sum())
    return (xor_area / (lesion_area + eps)) * 100.0


def _axis_area_ratio(aligned_mask: np.ndarray, axis: str, center: tuple[int, int], eps: float = 1e-8) -> float:
    """Compute area imbalance across one PCA axis."""
    if aligned_mask.sum() == 0:
        return 0.0

    cx, cy = center
    if axis == 'major':
        side_a = float(aligned_mask[:cy, :].sum())
        side_b = float(aligned_mask[cy + 1:, :].sum())
    elif axis == 'minor':
        side_a = float(aligned_mask[:, :cx].sum())
        side_b = float(aligned_mask[:, cx + 1:].sum())
    else:
        raise ValueError("axis must be 'major' or 'minor'.")

    return abs(side_a - side_b) / (float(aligned_mask.sum()) + eps) * 100.0


def _compute_single_asymmetry_plus_features(mask: np.ndarray) -> list[float]:
    aligned_mask, center, angle_deg, axis_ratio, centroid_offset = _principal_axis_aligned_mask(mask)
    major_xor = _axis_fold_xor_pct(aligned_mask, axis='major', center=center)
    minor_xor = _axis_fold_xor_pct(aligned_mask, axis='minor', center=center)
    major_area_ratio = _axis_area_ratio(aligned_mask, axis='major', center=center)
    minor_area_ratio = _axis_area_ratio(aligned_mask, axis='minor', center=center)
    combined_xor = 0.5 * (major_xor + minor_xor)

    return [
        float(major_xor),
        float(minor_xor),
        float(major_area_ratio),
        float(minor_area_ratio),
        float(combined_xor),
        float(angle_deg),
        float(axis_ratio),
        float(centroid_offset),
    ]


def extract_lesion_feature_asymmetry_plus(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Extract PCA principal-axis asymmetry features from lesion masks."""
    rows = [_compute_single_asymmetry_plus_features(mask) for mask in masks]
    return np.asarray(rows, dtype=np.float32), _ASYMMETRY_PLUS_FEATURE_NAMES.copy()


# functions for color features

def _safe_skew(vals: np.ndarray, eps: float = 1e-6) -> float:
    vals = np.asarray(vals, dtype=np.float32)
    if vals.size < 3 or np.nanstd(vals) < eps:
        return 0.0
    value = stats.skew(vals, nan_policy="omit")
    return float(0.0 if np.isnan(value) else value)


def _normalized_rgb(rgb: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    total = r + g + b + eps
    return np.stack([r / total, g / total, b / total], axis=-1).astype(np.float32)


def _ohta_color_space(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    return np.stack([
        (r + g + b) / 3.0,
        (r - b) / 2.0,
        (2.0 * g - r - b) / 4.0,
    ], axis=-1).astype(np.float32)


def _gevers_l123(rgb: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    total = r + g + b + eps
    return np.stack([
        ((r - g) ** 2) / total,
        ((r - b) ** 2) / total,
        ((g - b) ** 2) / total,
    ], axis=-1).astype(np.float32)


def _build_peripheral_regions(
    lesion_mask: np.ndarray,
    transition_ratio: float = 0.10,
    ring_ratio: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    outside_mask = ~lesion_mask
    lesion_area = int(lesion_mask.sum())

    if lesion_area <= 0 or outside_mask.sum() == 0:
        empty = np.zeros_like(lesion_mask, dtype=bool)
        return empty, empty

    dist_outside = cv2.distanceTransform(outside_mask.astype(np.uint8), cv2.DIST_L2, 5)
    outside_indices = np.argwhere(outside_mask)
    order = np.argsort(dist_outside[outside_mask])
    outside_indices_sorted = outside_indices[order]

    n_transition = int(round(transition_ratio * lesion_area))
    n_inner = int(round(ring_ratio * lesion_area))
    n_outer = int(round(ring_ratio * lesion_area))

    start_inner = min(n_transition, len(outside_indices_sorted))
    end_inner = min(start_inner + n_inner, len(outside_indices_sorted))
    end_outer = min(end_inner + n_outer, len(outside_indices_sorted))

    inner_mask = np.zeros_like(lesion_mask, dtype=bool)
    outer_mask = np.zeros_like(lesion_mask, dtype=bool)

    if end_inner > start_inner:
        inner_idx = outside_indices_sorted[start_inner:end_inner]
        inner_mask[inner_idx[:, 0], inner_idx[:, 1]] = True

    if end_outer > end_inner:
        outer_idx = outside_indices_sorted[end_inner:end_outer]
        outer_mask[outer_idx[:, 0], outer_idx[:, 1]] = True

    return inner_mask, outer_mask


def _region_mean_std(img3: np.ndarray, channel_names: Sequence[str], region_mask: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for idx, channel_name in enumerate(channel_names):
        vals = img3[:, :, idx][region_mask].astype(np.float32)
        if vals.size == 0:
            vals = img3[:, :, idx].reshape(-1).astype(np.float32)
        out[channel_name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }
    return out


def _extract_single_color_features(image: np.ndarray, mask: np.ndarray, eps: float = 1e-6) -> dict[str, float]:
    feats: dict[str, float] = {}
    img_rgb_uint8 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = img_rgb_uint8.astype(np.float32)
    h_img, w_img = rgb.shape[:2]

    mask_binary = _to_binary_mask(mask)
    if mask_binary.shape[:2] != rgb.shape[:2]:
        mask_binary = cv2.resize(mask_binary, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

    lesion_mask = mask_binary > 127
    if lesion_mask.sum() == 0:
        lesion_mask = np.ones((h_img, w_img), dtype=bool)

    norm_rgb = _normalized_rgb(rgb, eps=eps)
    hsv = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] *= 2.0
    ohta = _ohta_color_space(rgb)
    gevers = _gevers_l123(rgb, eps=eps)
    luv = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2Luv).astype(np.float32)
    lab = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2Lab).astype(np.float32)
    ycrcb = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    ycbcr = np.stack([ycrcb[:, :, 0], ycrcb[:, :, 2], ycrcb[:, :, 1]], axis=-1).astype(np.float32)

    color_spaces_all = {
        "RGB": (rgb, ["R", "G", "B"]),
        "rgb": (norm_rgb, ["r", "g", "b"]),
        "HSV": (hsv, ["H", "S", "V"]),
        "Ohta": (ohta, ["I1", "I2", "I3"]),
        "Gevers": (gevers, ["l1", "l2", "l3"]),
        "Luv": (luv, ["L_luv", "u_luv", "v_luv"]),
        "Lab": (lab, ["L_lab", "a_lab", "b_lab"]),
        "YCbCr": (ycbcr, ["Y", "Cb", "Cr"]),
    }

    for space_name, (space_img, channel_names) in color_spaces_all.items():
        for idx, channel_name in enumerate(channel_names):
            vals = space_img[:, :, idx][lesion_mask].astype(np.float32)
            if vals.size == 0:
                vals = space_img[:, :, idx].reshape(-1).astype(np.float32)
            feats[f"{space_name}_{channel_name}_max"] = float(np.max(vals))
            feats[f"{space_name}_{channel_name}_min"] = float(np.min(vals))
            feats[f"{space_name}_{channel_name}_mean"] = float(np.mean(vals))
            feats[f"{space_name}_{channel_name}_var"] = float(np.var(vals))
            feats[f"{space_name}_{channel_name}_std"] = float(np.std(vals))
            feats[f"{space_name}_{channel_name}_skew"] = _safe_skew(vals, eps=eps)

    inner_mask, outer_mask = _build_peripheral_regions(lesion_mask)
    regional_spaces = {
        "RGB": (rgb, ["R", "G", "B"]),
        "rgb": (norm_rgb, ["r", "g", "b"]),
        "HSV": (hsv, ["H", "S", "V"]),
        "Ohta": (ohta, ["I1", "I2", "I3"]),
        "Gevers": (gevers, ["l1", "l2", "l3"]),
        "Luv": (luv, ["L_luv", "u_luv", "v_luv"]),
    }

    for space_name, (space_img, channel_names) in regional_spaces.items():
        region_stats = {
            "lesion": _region_mean_std(space_img, channel_names, lesion_mask),
            "inner": _region_mean_std(space_img, channel_names, inner_mask),
            "outer": _region_mean_std(space_img, channel_names, outer_mask),
        }
        for channel_name in channel_names:
            for stat_name in ["mean", "std"]:
                for region_a, region_b in [("outer", "inner"), ("outer", "lesion"), ("inner", "lesion")]:
                    a = region_stats[region_a][channel_name][stat_name]
                    b = region_stats[region_b][channel_name][stat_name]
                    feats[f"{space_name}_{channel_name}_{stat_name}_{region_a}_div_{region_b}"] = float(a / (b + eps))
                    feats[f"{space_name}_{channel_name}_{stat_name}_{region_a}_minus_{region_b}"] = float(a - b)

    def add_color_asymmetry(channel_img: np.ndarray, channel_name: str) -> None:
        ys, xs = np.where(lesion_mask)
        vals = channel_img[lesion_mask].astype(np.float32)
        keys = [
            f"asym_{channel_name}_axis0_pct",
            f"asym_{channel_name}_axis0_sum",
            f"asym_{channel_name}_axis90_pct",
            f"asym_{channel_name}_axis90_sum",
        ]
        if vals.size < 5 or np.sum(vals) <= eps:
            for key in keys:
                feats[key] = 0.0
            return

        weights = vals + eps
        cx = np.sum(xs * weights) / np.sum(weights)
        cy = np.sum(ys * weights) / np.sum(weights)
        x0 = xs - cx
        y0 = ys - cy
        cov = np.array([
            [np.sum(weights * x0 * x0) / np.sum(weights), np.sum(weights * x0 * y0) / np.sum(weights)],
            [np.sum(weights * x0 * y0) / np.sum(weights), np.sum(weights * y0 * y0) / np.sum(weights)],
        ], dtype=np.float32)
        _, eigvecs = np.linalg.eigh(cov)
        ux, uy = eigvecs[:, -1]
        u = x0 * ux + y0 * uy
        v = -x0 * uy + y0 * ux

        def folded_difference(coord_a: np.ndarray, coord_b: np.ndarray, values: np.ndarray, fold_axis: str) -> tuple[float, float]:
            qa = np.round(coord_a).astype(int)
            qb = np.round(coord_b).astype(int)
            bins: dict[tuple[int, int], float] = {}
            for a, b, value in zip(qa, qb, values):
                key = (int(a), int(b))
                bins[key] = bins.get(key, 0.0) + float(value)
            visited: set[tuple[int, int]] = set()
            total_abs_diff = 0.0
            for key, value in bins.items():
                if key in visited:
                    continue
                a, b = key
                mirror_key = (-a, b) if fold_axis == "u" else (a, -b)
                total_abs_diff += abs(value - bins.get(mirror_key, 0.0))
                visited.add(key)
                visited.add(mirror_key)
            return float(total_abs_diff / (np.sum(values) + eps)), float(total_abs_diff / (len(values) + eps))

        feats[keys[0]], feats[keys[1]] = folded_difference(u, v, vals, fold_axis="u")
        feats[keys[2]], feats[keys[3]] = folded_difference(u, v, vals, fold_axis="v")

    add_color_asymmetry(rgb[:, :, 0], "R")
    add_color_asymmetry(rgb[:, :, 1], "G")
    add_color_asymmetry(rgb[:, :, 2], "B")

    ys, xs = np.where(lesion_mask)
    geom_cx = np.mean(xs)
    geom_cy = np.mean(ys)
    equiv_diameter = np.sqrt(4.0 * lesion_mask.sum() / np.pi) + eps
    for space_name, (space_img, channel_names) in regional_spaces.items():
        for idx, channel_name in enumerate(channel_names):
            vals = space_img[:, :, idx][lesion_mask].astype(np.float32)
            weights = vals - np.min(vals) + eps
            if np.sum(weights) <= eps:
                feats[f"centroid_dist_{space_name}_{channel_name}"] = 0.0
                continue
            bright_cx = np.sum(xs * weights) / np.sum(weights)
            bright_cy = np.sum(ys * weights) / np.sum(weights)
            dist = np.sqrt((bright_cx - geom_cx) ** 2 + (bright_cy - geom_cy) ** 2)
            feats[f"centroid_dist_{space_name}_{channel_name}"] = float(dist / equiv_diameter)

    def luv_hist(region_mask: np.ndarray) -> np.ndarray:
        vals = luv[region_mask]
        if vals.shape[0] == 0:
            return np.zeros(4 * 8 * 8, dtype=np.float32)
        hist, _ = np.histogramdd(vals, bins=(4, 8, 8), range=((0, 256), (0, 256), (0, 256)))
        hist = hist.astype(np.float32).reshape(-1)
        return hist / (hist.sum() + eps)

    hist_regions = {
        "lesion": luv_hist(lesion_mask),
        "inner": luv_hist(inner_mask),
        "outer": luv_hist(outer_mask),
    }
    for region_a, region_b in [("lesion", "inner"), ("lesion", "outer"), ("inner", "outer")]:
        hist_a = hist_regions[region_a]
        hist_b = hist_regions[region_b]
        feats[f"Luv_hist_L1_{region_a}_{region_b}"] = float(np.sum(np.abs(hist_a - hist_b)))
        feats[f"Luv_hist_L2_{region_a}_{region_b}"] = float(np.sqrt(np.sum((hist_a - hist_b) ** 2)))

    r = rgb[:, :, 0][lesion_mask].astype(np.float32)
    g = rgb[:, :, 1][lesion_mask].astype(np.float32)
    b = rgb[:, :, 2][lesion_mask].astype(np.float32)
    erythema_index = r / (g + b + eps)
    feats["erythema_index_mean"] = float(np.mean(erythema_index))
    feats["erythema_index_var"] = float(np.var(erythema_index))

    h_deg = hsv[:, :, 0][lesion_mask].astype(np.float32)
    red_hue_mask = ((h_deg >= 0) & (h_deg <= 20)) | ((h_deg >= 340) & (h_deg <= 360))
    feats["hue_red_ratio"] = float(np.mean(red_hue_mask))

    return feats


def extract_lesion_feature_color(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Extract color features and return a matrix plus feature names."""
    feature_dicts = [_extract_single_color_features(image, mask) for image, mask in zip(images, masks)]
    feature_names = list(feature_dicts[0].keys()) if feature_dicts else []
    feature_rows = [[feature_dict.get(name, 0.0) for name in feature_names] for feature_dict in feature_dicts]
    return np.asarray(feature_rows, dtype=np.float32), feature_names


# functions for superpixel clinical color proportion features
from skimage.segmentation import slic as _slic


_CLINICAL_COLOR_RGB_REFERENCES = {
    'white': (235, 235, 225),
    'black': (35, 30, 25),
    'light_brown': (175, 125, 75),
    'dark_brown': (85, 50, 30),
    'blue_gray': (80, 105, 130),
    'red': (180, 60, 55),
}
_CLINICAL_COLOR_NAMES = list(_CLINICAL_COLOR_RGB_REFERENCES.keys())
_COLOR_SUPERPIXEL_FEATURE_NAMES = [
    f'color_proportion_{color_name}'
    for color_name in _CLINICAL_COLOR_NAMES
] + [
    'color_proportion_present_count',
    'color_proportion_entropy',
    'color_proportion_dominance',
]


def _clinical_color_lab_references() -> np.ndarray:
    rgb_values = np.asarray([_CLINICAL_COLOR_RGB_REFERENCES[name] for name in _CLINICAL_COLOR_NAMES], dtype=np.uint8)
    rgb_values = rgb_values.reshape(1, -1, 3)
    lab_values = cv2.cvtColor(rgb_values, cv2.COLOR_RGB2Lab).reshape(-1, 3).astype(np.float32)
    return lab_values


def _prepare_rgb_and_lesion_mask(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb_uint8 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    binary = _to_binary_mask(mask)
    if binary.shape[:2] != rgb_uint8.shape[:2]:
        binary = cv2.resize(binary, (rgb_uint8.shape[1], rgb_uint8.shape[0]), interpolation=cv2.INTER_NEAREST)
    lesion_mask = binary > 127
    return rgb_uint8, lesion_mask


def _nearest_clinical_color_indices(lab_values: np.ndarray, reference_lab: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(lab_values[:, None, :] - reference_lab[None, :, :], axis=2)
    return np.argmin(distances, axis=1)


def _pixel_level_clinical_color_proportions(lab_image: np.ndarray, lesion_mask: np.ndarray, reference_lab: np.ndarray) -> np.ndarray:
    lesion_lab = lab_image[lesion_mask].astype(np.float32)
    if lesion_lab.size == 0:
        return np.zeros(len(_CLINICAL_COLOR_NAMES), dtype=np.float32)
    labels = _nearest_clinical_color_indices(lesion_lab, reference_lab)
    counts = np.bincount(labels, minlength=len(_CLINICAL_COLOR_NAMES)).astype(np.float32)
    return counts / (counts.sum() + 1e-8)


def _compute_single_superpixel_color_features(
    image: np.ndarray,
    mask: np.ndarray,
    n_segments: int = 80,
    compactness: float = 10.0,
    min_present_ratio: float = 0.05,
    eps: float = 1e-8,
) -> list[float]:
    rgb_uint8, lesion_mask = _prepare_rgb_and_lesion_mask(image, mask)
    if lesion_mask.sum() == 0:
        return [0.0] * len(_COLOR_SUPERPIXEL_FEATURE_NAMES)

    lab_image = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2Lab).astype(np.float32)
    reference_lab = _clinical_color_lab_references()
    proportions = np.zeros(len(_CLINICAL_COLOR_NAMES), dtype=np.float32)

    try:
        segments = _slic(
            rgb_uint8.astype(np.float32) / 255.0,
            n_segments=max(2, int(n_segments)),
            compactness=float(compactness),
            mask=lesion_mask,
            start_label=0,
            channel_axis=-1,
        )
        segment_ids = np.unique(segments[lesion_mask])
        for segment_id in segment_ids:
            segment_mask = (segments == segment_id) & lesion_mask
            area = float(segment_mask.sum())
            if area <= 0:
                continue
            mean_lab = lab_image[segment_mask].mean(axis=0, keepdims=True)
            color_idx = int(_nearest_clinical_color_indices(mean_lab, reference_lab)[0])
            proportions[color_idx] += area
        if proportions.sum() <= 0:
            proportions = _pixel_level_clinical_color_proportions(lab_image, lesion_mask, reference_lab)
        else:
            proportions = proportions / (proportions.sum() + eps)
    except Exception:
        proportions = _pixel_level_clinical_color_proportions(lab_image, lesion_mask, reference_lab)

    present_count = float(np.sum(proportions >= min_present_ratio))
    entropy = float(-np.sum(proportions * np.log(proportions + eps)) / np.log(len(proportions)))
    dominance = float(np.max(proportions))
    return [float(value) for value in proportions] + [present_count, entropy, dominance]


def extract_lesion_feature_color_superpixel(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    n_segments: int = 80,
    compactness: float = 10.0,
) -> tuple[np.ndarray, list[str]]:
    """Extract SLIC-based clinical color proportion features from lesion regions."""
    rows = [
        _compute_single_superpixel_color_features(
            image,
            mask,
            n_segments=n_segments,
            compactness=compactness,
        )
        for image, mask in zip(images, masks)
    ]
    return np.asarray(rows, dtype=np.float32), _COLOR_SUPERPIXEL_FEATURE_NAMES.copy()


# functions for texture features: GLCM

_GLCM_FEATURE_NAMES = [
    "glcm_ASM",
    "glcm_Contrast",
    "glcm_Correlation",
    "glcm_Homogeneity",
    "glcm_Dissimilarity",
    "glcm_Entropy",
    "glcm_MaxProbability",
    "glcm_Variance",
    "glcm_SumVariance",
    "glcm_SumEntropy",
    "glcm_DifferenceVariance",
    "glcm_DifferenceEntropy",
    "glcm_IMCorr1",
    "glcm_IMCorr2",
]


def _compute_single_glcm_features(
    image: np.ndarray,
    mask: np.ndarray,
    distances: Sequence[int] = (1,),
    angles: Sequence[float] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    levels: int = 64,
) -> list[float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_mask = _to_binary_mask(mask)
    if binary_mask.shape[:2] != gray.shape[:2]:
        binary_mask = cv2.resize(binary_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

    coords = cv2.findNonZero(binary_mask)
    if coords is None:
        return [np.nan] * len(_GLCM_FEATURE_NAMES)

    x, y, w, h = cv2.boundingRect(coords)
    roi_gray = gray[y:y + h, x:x + w]
    roi_gray = (roi_gray / 256 * levels).astype(np.uint8)
    roi_gray[roi_gray >= levels] = levels - 1

    glcm = graycomatrix(
        roi_gray,
        distances=list(distances),
        angles=list(angles),
        levels=levels,
        symmetric=True,
        normed=True,
    )

    values_by_feature = {name: [] for name in _GLCM_FEATURE_NAMES}

    for angle_idx in range(len(angles)):
        glcm_angle = glcm[:, :, 0, angle_idx]
        glcm_4d = glcm_angle.reshape(levels, levels, 1, 1)

        values_by_feature["glcm_ASM"].append(graycoprops(glcm_4d, "ASM")[0, 0])
        values_by_feature["glcm_Contrast"].append(graycoprops(glcm_4d, "contrast")[0, 0])
        values_by_feature["glcm_Correlation"].append(graycoprops(glcm_4d, "correlation")[0, 0])
        values_by_feature["glcm_Homogeneity"].append(graycoprops(glcm_4d, "homogeneity")[0, 0])
        values_by_feature["glcm_Dissimilarity"].append(graycoprops(glcm_4d, "dissimilarity")[0, 0])

        p = glcm_angle + 1e-12
        values_by_feature["glcm_Entropy"].append(-np.sum(p * np.log(p)))
        values_by_feature["glcm_MaxProbability"].append(np.max(glcm_angle))

        px = np.sum(glcm_angle, axis=1)
        py = np.sum(glcm_angle, axis=0)
        mu = np.sum(np.arange(levels) * px)
        values_by_feature["glcm_Variance"].append(np.sum((np.arange(levels)[:, None] - mu) ** 2 * glcm_angle))

        sum_prob = np.zeros(2 * levels + 1)
        diff_prob = np.zeros(levels)
        for i_idx in range(levels):
            for j_idx in range(levels):
                prob = glcm_angle[i_idx, j_idx]
                sum_prob[i_idx + j_idx] += prob
                diff_prob[abs(i_idx - j_idx)] += prob

        k_vals = np.arange(2, 2 * levels + 1)
        sum_prob_valid = sum_prob[2:2 * levels + 1]
        mu_sum = np.sum(k_vals * sum_prob_valid)
        values_by_feature["glcm_SumVariance"].append(np.sum((k_vals - mu_sum) ** 2 * sum_prob_valid))
        values_by_feature["glcm_SumEntropy"].append(-np.sum(sum_prob_valid * np.log(sum_prob_valid + 1e-12)))

        mu_diff = np.sum(np.arange(levels) * diff_prob)
        values_by_feature["glcm_DifferenceVariance"].append(np.sum((np.arange(levels) - mu_diff) ** 2 * diff_prob))
        values_by_feature["glcm_DifferenceEntropy"].append(-np.sum(diff_prob * np.log(diff_prob + 1e-12)))

        hx = -np.sum(px * np.log(px + 1e-12))
        hy = -np.sum(py * np.log(py + 1e-12))
        hxy = -np.sum(glcm_angle * np.log(glcm_angle + 1e-12))
        hxy1 = -np.sum(glcm_angle * np.log(px[:, None] * py[None, :] + 1e-12))
        hxy2 = -np.sum(px[:, None] * py[None, :] * np.log(px[:, None] * py[None, :] + 1e-12))
        values_by_feature["glcm_IMCorr1"].append((hxy - hxy1) / (max(hx, hy) + 1e-12))
        values_by_feature["glcm_IMCorr2"].append(np.sqrt(max(0.0, 1 - np.exp(-2 * (hxy2 - hxy)))))

    return [float(np.mean(values_by_feature[name])) for name in _GLCM_FEATURE_NAMES]


def extract_lesion_feature_glcm(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    distances: Sequence[int] = (1,),
    angles: Sequence[float] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    levels: int = 64,
) -> tuple[np.ndarray, list[str]]:
    """Extract GLCM texture features and return a matrix plus feature names."""
    feature_rows = [
        _compute_single_glcm_features(
            image,
            mask,
            distances=distances,
            angles=angles,
            levels=levels,
        )
        for image, mask in zip(images, masks)
    ]
    return np.asarray(feature_rows, dtype=np.float32), _GLCM_FEATURE_NAMES.copy()


# functions for mask-aware texture features: GLCM_plus

_GLCM_PLUS_FEATURE_NAMES = [
    name.replace('glcm_', 'glcm_plus_', 1)
    for name in _GLCM_FEATURE_NAMES
]


def _quantize_gray_image(gray: np.ndarray, levels: int) -> np.ndarray:
    quantized = np.floor(gray.astype(np.float32) / 256.0 * levels).astype(np.uint8)
    quantized[quantized >= levels] = levels - 1
    return quantized


def _masked_graycomatrix_2d(
    quantized_gray: np.ndarray,
    lesion_mask: np.ndarray,
    distance: int,
    angle: float,
    levels: int,
    symmetric: bool = True,
) -> np.ndarray:
    """Build a GLCM from pixel pairs where both pixels are inside the lesion mask."""
    h, w = quantized_gray.shape
    dx = int(round(np.cos(angle) * distance))
    dy = int(round(np.sin(angle) * distance))
    if dx == 0 and dy == 0:
        return np.zeros((levels, levels), dtype=np.float64)

    y0_start = max(0, -dy)
    y0_end = min(h, h - dy)
    x0_start = max(0, -dx)
    x0_end = min(w, w - dx)
    y1_start = y0_start + dy
    y1_end = y0_end + dy
    x1_start = x0_start + dx
    x1_end = x0_end + dx

    src_vals = quantized_gray[y0_start:y0_end, x0_start:x0_end]
    dst_vals = quantized_gray[y1_start:y1_end, x1_start:x1_end]
    valid_pairs = (
        lesion_mask[y0_start:y0_end, x0_start:x0_end]
        & lesion_mask[y1_start:y1_end, x1_start:x1_end]
    )

    glcm = np.zeros((levels, levels), dtype=np.float64)
    if not np.any(valid_pairs):
        return glcm

    src_flat = src_vals[valid_pairs].ravel()
    dst_flat = dst_vals[valid_pairs].ravel()
    np.add.at(glcm, (src_flat, dst_flat), 1.0)
    if symmetric:
        np.add.at(glcm, (dst_flat, src_flat), 1.0)

    matrix_sum = glcm.sum()
    if matrix_sum > 0:
        glcm /= matrix_sum
    return glcm


def _haralick_features_from_glcm_2d(glcm_angle: np.ndarray, levels: int) -> dict[str, float]:
    glcm_angle = np.nan_to_num(glcm_angle.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if glcm_angle.sum() <= 0:
        return {name: 0.0 for name in _GLCM_FEATURE_NAMES}
    glcm_angle = glcm_angle / glcm_angle.sum()
    glcm_4d = glcm_angle.reshape(levels, levels, 1, 1)

    out: dict[str, float] = {}
    out['glcm_ASM'] = float(graycoprops(glcm_4d, 'ASM')[0, 0])
    out['glcm_Contrast'] = float(graycoprops(glcm_4d, 'contrast')[0, 0])
    out['glcm_Correlation'] = float(np.nan_to_num(graycoprops(glcm_4d, 'correlation')[0, 0], nan=0.0))
    out['glcm_Homogeneity'] = float(graycoprops(glcm_4d, 'homogeneity')[0, 0])
    out['glcm_Dissimilarity'] = float(graycoprops(glcm_4d, 'dissimilarity')[0, 0])

    p = glcm_angle + 1e-12
    out['glcm_Entropy'] = float(-np.sum(p * np.log(p)))
    out['glcm_MaxProbability'] = float(np.max(glcm_angle))

    px = np.sum(glcm_angle, axis=1)
    py = np.sum(glcm_angle, axis=0)
    levels_range = np.arange(levels)
    mu = np.sum(levels_range * px)
    out['glcm_Variance'] = float(np.sum((levels_range[:, None] - mu) ** 2 * glcm_angle))

    sum_prob = np.zeros(2 * levels + 1)
    diff_prob = np.zeros(levels)
    for i_idx in range(levels):
        for j_idx in range(levels):
            prob = glcm_angle[i_idx, j_idx]
            sum_prob[i_idx + j_idx] += prob
            diff_prob[abs(i_idx - j_idx)] += prob

    k_vals = np.arange(2, 2 * levels + 1)
    sum_prob_valid = sum_prob[2:2 * levels + 1]
    mu_sum = np.sum(k_vals * sum_prob_valid)
    out['glcm_SumVariance'] = float(np.sum((k_vals - mu_sum) ** 2 * sum_prob_valid))
    out['glcm_SumEntropy'] = float(-np.sum(sum_prob_valid * np.log(sum_prob_valid + 1e-12)))

    mu_diff = np.sum(np.arange(levels) * diff_prob)
    out['glcm_DifferenceVariance'] = float(np.sum((np.arange(levels) - mu_diff) ** 2 * diff_prob))
    out['glcm_DifferenceEntropy'] = float(-np.sum(diff_prob * np.log(diff_prob + 1e-12)))

    hx = -np.sum(px * np.log(px + 1e-12))
    hy = -np.sum(py * np.log(py + 1e-12))
    hxy = -np.sum(glcm_angle * np.log(glcm_angle + 1e-12))
    hxy1 = -np.sum(glcm_angle * np.log(px[:, None] * py[None, :] + 1e-12))
    hxy2 = -np.sum(px[:, None] * py[None, :] * np.log(px[:, None] * py[None, :] + 1e-12))
    out['glcm_IMCorr1'] = float((hxy - hxy1) / (max(hx, hy) + 1e-12))
    out['glcm_IMCorr2'] = float(np.sqrt(max(0.0, 1 - np.exp(-2 * (hxy2 - hxy)))))

    return out


def _compute_single_glcm_plus_features(
    image: np.ndarray,
    mask: np.ndarray,
    distances: Sequence[int] = (1,),
    angles: Sequence[float] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    levels: int = 64,
) -> list[float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_mask = _to_binary_mask(mask)
    if binary_mask.shape[:2] != gray.shape[:2]:
        binary_mask = cv2.resize(binary_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

    lesion_mask = binary_mask > 127
    if lesion_mask.sum() == 0:
        return [0.0] * len(_GLCM_PLUS_FEATURE_NAMES)

    quantized = _quantize_gray_image(gray, levels=levels)
    values_by_feature = {name: [] for name in _GLCM_FEATURE_NAMES}

    for distance in distances:
        for angle in angles:
            glcm_2d = _masked_graycomatrix_2d(
                quantized_gray=quantized,
                lesion_mask=lesion_mask,
                distance=int(distance),
                angle=float(angle),
                levels=levels,
                symmetric=True,
            )
            feature_dict = _haralick_features_from_glcm_2d(glcm_2d, levels=levels)
            for feature_name in _GLCM_FEATURE_NAMES:
                values_by_feature[feature_name].append(feature_dict[feature_name])

    return [
        float(np.mean(values_by_feature[old_name]))
        for old_name in _GLCM_FEATURE_NAMES
    ]


def extract_lesion_feature_glcm_plus(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    distances: Sequence[int] = (1,),
    angles: Sequence[float] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    levels: int = 64,
) -> tuple[np.ndarray, list[str]]:
    """Extract mask-aware GLCM features without discarding true black lesion pixels."""
    rows = [
        _compute_single_glcm_plus_features(
            image,
            mask,
            distances=distances,
            angles=angles,
            levels=levels,
        )
        for image, mask in zip(images, masks)
    ]
    return np.asarray(rows, dtype=np.float32), _GLCM_PLUS_FEATURE_NAMES.copy()


# functions for local binary pattern texture features: LBP
from skimage.feature import local_binary_pattern as _local_binary_pattern


def _lbp_n_bins(points: int, method: str) -> int:
    if method == 'nri_uniform':
        return points * (points - 1) + 3
    if method == 'uniform':
        return points + 2
    return 2 ** points


def _lbp_feature_names(points: int = 8, method: str = 'nri_uniform') -> list[str]:
    n_bins = _lbp_n_bins(points, method)
    return [f'texture_lbp_bin_{idx:02d}' for idx in range(n_bins)]


def _compute_single_lbp_features(
    image: np.ndarray,
    mask: np.ndarray,
    points: int = 8,
    radius: float = 1.0,
    method: str = 'nri_uniform',
    eps: float = 1e-8,
) -> list[float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    binary = _to_binary_mask(mask)
    if binary.shape[:2] != gray.shape[:2]:
        binary = cv2.resize(binary, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

    lesion_mask = binary > 127
    n_bins = _lbp_n_bins(points, method)
    if lesion_mask.sum() == 0:
        return [0.0] * n_bins

    lbp = _local_binary_pattern(gray, P=points, R=radius, method=method)
    lbp_values = lbp[lesion_mask].astype(np.int32)
    lbp_values = np.clip(lbp_values, 0, n_bins - 1)
    hist = np.bincount(lbp_values, minlength=n_bins).astype(np.float32)
    hist = hist / (hist.sum() + eps)
    return [float(value) for value in hist]


def extract_lesion_feature_lbp(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    points: int = 8,
    radius: float = 1.0,
    method: str = 'nri_uniform',
) -> tuple[np.ndarray, list[str]]:
    """Extract masked LBP histogram texture features from lesion regions."""
    feature_names = _lbp_feature_names(points=points, method=method)
    rows = [
        _compute_single_lbp_features(
            image,
            mask,
            points=points,
            radius=radius,
            method=method,
        )
        for image, mask in zip(images, masks)
    ]
    return np.asarray(rows, dtype=np.float32), feature_names


# functions for concat features and feature names

def concat_features(*feature_matrices: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate feature matrices by columns."""
    if len(feature_matrices) == 1 and isinstance(feature_matrices[0], (list, tuple)):
        feature_matrices = tuple(feature_matrices[0])
    return np.concatenate(feature_matrices, axis=1)


def concat_feature_names(*feature_name_lists: Sequence[str] | Sequence[Sequence[str]]) -> list[str]:
    """Concatenate feature-name lists while preserving order."""
    if len(feature_name_lists) == 1 and isinstance(feature_name_lists[0], (list, tuple)):
        feature_name_lists = tuple(feature_name_lists[0])
    names: list[str] = []
    for feature_name_list in feature_name_lists:
        names.extend(feature_name_list)
    return names


def extract_all_feature_blocks(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Extract every feature block used by the training notebook."""
    return {
        'shape': extract_lesion_feature_shape(images, masks),
        'shape_plus': extract_lesion_feature_shape_plus(images, masks),
        'hu': extract_lesion_feature_hu_moments(images, masks),
        'asymmetry': extract_lesion_feature_asymmetry(images, masks),
        'asymmetry_plus': extract_lesion_feature_asymmetry_plus(images, masks),
        'color': extract_lesion_feature_color(images, masks),
        'color_superpixel': extract_lesion_feature_color_superpixel(images, masks),
        'glcm': extract_lesion_feature_glcm(images, masks),
        'glcm_plus': extract_lesion_feature_glcm_plus(images, masks),
        'lbp': extract_lesion_feature_lbp(images, masks),
    }


def build_feature_df_for_inference(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    filenames: Sequence[str],
) -> pd.DataFrame:
    """Build a feature DataFrame with the same column names as submit.ipynb."""
    feature_blocks = extract_all_feature_blocks(images, masks)
    matrices = [matrix for matrix, _ in feature_blocks.values()]
    name_lists = [names for _, names in feature_blocks.values()]
    all_features = concat_features(matrices)
    all_feature_names = concat_feature_names(name_lists)

    feature_df = pd.DataFrame(all_features, columns=all_feature_names)
    feature_df.insert(0, 'filename', list(filenames))
    feature_df.insert(1, 'image_id', [Path(name).stem for name in filenames])
    feature_df.insert(2, 'base_id', [Path(name).stem.split('_aug')[0] for name in filenames])
    return feature_df


def extract_feature_df_from_directories(
    image_dir: str | Path,
    mask_dir: str | Path,
    max_images: int | None = None,
) -> pd.DataFrame:
    """Load image-mask pairs and return the full inference feature table."""
    images, masks, filenames = load_image_mask_pairs(image_dir, mask_dir, max_images=max_images)
    return build_feature_df_for_inference(images, masks, filenames)

# =============================================================================
# Section 4: model feature-alignment functions
# =============================================================================
# feature alignment helper: infer required feature names from the saved model

def get_two_stage_result_from_bundle(bundle: dict[str, object]) -> dict[str, object]:
    """Extract two_stage_result from an imported model bundle."""
    if 'two_stage_result' not in bundle:
        raise ValueError('Invalid model bundle: missing two_stage_result.')
    return bundle['two_stage_result']


def get_required_model_features(two_stage_result: dict[str, object]) -> list[str]:
    """Return the union of first-stage and second-stage feature names required by the model."""
    first_features = list(two_stage_result['first_feature_names'])
    second_features = list(two_stage_result['second_feature_names'])
    required = list(dict.fromkeys(first_features + second_features))
    print(f'First-stage feature count: {len(first_features)}')
    print(f'Second-stage feature count: {len(second_features)}')
    print(f'Union required feature count: {len(required)}')
    return required

# feature alignment helper: validate feature_df columns against the saved model

def validate_feature_df_for_model(feature_df: pd.DataFrame, two_stage_result: dict[str, object]) -> None:
    """Raise a clear error if model-required features are missing."""
    required_features = get_required_model_features(two_stage_result)
    missing = [feature for feature in required_features if feature not in feature_df.columns]
    if missing:
        raise ValueError(
            f'Feature table is missing {len(missing)} model-required features. '
            f'Example: {missing[:10]}'
        )
    print('[SUCCESS] Feature table contains every model-required feature.')

# feature alignment helper: build model matrices in the exact saved feature order

def prepare_feature_matrix(
    features: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Convert feature values to a finite float32 matrix for model prediction."""
    return np.nan_to_num(
        np.asarray(features, dtype=np.float32),
        nan=fill_value,
        posinf=fill_value,
        neginf=fill_value,
    )


def select_model_feature_matrix(feature_df: pd.DataFrame, feature_names: Sequence[str]) -> np.ndarray:
    """Select feature columns in the exact order stored in the model bundle."""
    return prepare_feature_matrix(feature_df.loc[:, list(feature_names)].values)

# =============================================================================
# Section 5: model import functions
# =============================================================================
# model import functions

def import_two_stage_model(model_path: str | Path) -> dict[str, object]:
    """Load the saved two-stage model bundle."""
    import joblib

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f'Model file does not exist: {model_path}')
    bundle = joblib.load(model_path)
    if 'two_stage_result' not in bundle:
        raise ValueError('Invalid model bundle: missing two_stage_result.')
    print(f'[SUCCESS] Imported two-stage model from: {model_path}')
    metadata = bundle.get('metadata', {})
    if metadata:
        print('Model metadata:')
        print(metadata)
    return bundle

# =============================================================================
# Section 6: prediction functions
# =============================================================================
# model prediction functions

def predict_model_labels(
    model_result: dict[str, object] | object,
    x_test: np.ndarray,
) -> np.ndarray:
    """Predict labels from a model or unified model-result dictionary."""
    if isinstance(model_result, dict):
        model = model_result['model']
        label_encoder = model_result.get('label_encoder')
    else:
        model = model_result
        label_encoder = getattr(model, 'label_encoder_', None)

    raw_pred = model.predict(prepare_feature_matrix(x_test))
    pred_array = np.asarray(raw_pred)

    if label_encoder is not None and np.issubdtype(pred_array.dtype, np.number):
        return label_encoder.inverse_transform(pred_array.astype(int))
    return pred_array.astype(str)


def predict_model_max_probability(
    model_result: dict[str, object] | object,
    x_test: np.ndarray,
) -> np.ndarray:
    """Return max predicted probability when the model exposes predict_proba."""
    model = model_result['model'] if isinstance(model_result, dict) else model_result
    if not hasattr(model, 'predict_proba'):
        return np.full(len(x_test), np.nan, dtype=np.float32)
    try:
        proba = model.predict_proba(prepare_feature_matrix(x_test))
        return np.max(np.asarray(proba, dtype=np.float32), axis=1)
    except Exception:
        return np.full(len(x_test), np.nan, dtype=np.float32)


def predict_two_stage_dataframe(
    two_stage_result: dict[str, object],
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Predict final labels and return a table ready for CSV export."""
    validate_feature_df_for_model(feature_df, two_stage_result)

    first_feature_names = list(two_stage_result['first_feature_names'])
    second_feature_names = list(two_stage_result['second_feature_names'])
    first_model_result = two_stage_result['first_model_result']
    second_model_result = two_stage_result['second_model_result']
    first_positive_label = str(two_stage_result['first_positive_label'])

    first_x = select_model_feature_matrix(feature_df, first_feature_names)
    first_pred = predict_model_labels(first_model_result, first_x).astype(str)
    first_conf = predict_model_max_probability(first_model_result, first_x)

    final_pred = np.full(len(first_pred), first_positive_label, dtype=object)
    second_pred = np.full(len(first_pred), '', dtype=object)
    second_conf = np.full(len(first_pred), np.nan, dtype=np.float32)
    routed_to_second = first_pred != first_positive_label

    if np.any(routed_to_second):
        second_x = select_model_feature_matrix(feature_df, second_feature_names)
        second_pred_values = predict_model_labels(second_model_result, second_x[routed_to_second]).astype(str)
        second_conf_values = predict_model_max_probability(second_model_result, second_x[routed_to_second])
        second_pred[routed_to_second] = second_pred_values
        second_conf[routed_to_second] = second_conf_values
        final_pred[routed_to_second] = second_pred_values

    result_df = feature_df[['filename', 'image_id', 'base_id']].copy()
    result_df['first_stage_prediction'] = first_pred
    result_df['first_stage_confidence'] = first_conf
    result_df['routed_to_second_stage'] = routed_to_second
    result_df['second_stage_prediction'] = second_pred
    result_df['second_stage_confidence'] = second_conf
    result_df['prediction'] = final_pred.astype(str)
    return result_df

# =============================================================================
# Section 7: CSV output helper functions
# =============================================================================
# helper function: save prediction result to CSV

def save_prediction_csv(prediction_df: pd.DataFrame, output_csv: str | Path) -> Path:
    """Save prediction results as a CSV file."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(output_csv, index=False)
    print(f'[SUCCESS] Saved prediction CSV to: {output_csv}')
    return output_csv

# =============================================================================
# Section 8: direct-running entry point
# =============================================================================

def run_inference(
    project_dir: str | Path = ".",
    model_path: str | Path | None = None,
    image_dir: str | Path | None = None,
    mask_dir: str | Path | None = None,
    raw_image_dir: str | Path | None = None,
    preprocessed_output_dir: str | Path | None = None,
    output_csv: str | Path | None = None,
    max_images: int | None = None,
    run_preprocessing: bool = True,
) -> pd.DataFrame:
    """Run the full image-to-CSV prediction workflow."""
    project_dir = Path(project_dir)
    model_path = Path(model_path) if model_path is not None else project_dir / "model.joblib"
    mask_dir = Path(mask_dir) if mask_dir is not None else project_dir / "mask"
    raw_image_dir = Path(raw_image_dir) if raw_image_dir is not None else project_dir / "image"
    preprocessed_output_dir = (
        Path(preprocessed_output_dir)
        if preprocessed_output_dir is not None
        else project_dir / "image_processed_submit2"
    )
    output_csv = Path(output_csv) if output_csv is not None else project_dir / "prediction_results.csv"

    if run_preprocessing:
        preprocess_image_directory(raw_image_dir, preprocessed_output_dir)
        image_dir = preprocessed_output_dir
    else:
        image_dir = Path(image_dir) if image_dir is not None else project_dir / "image_processed"
        if not image_dir.exists():
            image_dir = raw_image_dir
            print(f"[WARNING] Processed image directory not found; using raw image directory: {image_dir}")

    model_bundle = import_two_stage_model(model_path)
    two_stage_result = get_two_stage_result_from_bundle(model_bundle)

    inference_feature_df = extract_feature_df_from_directories(
        image_dir=image_dir,
        mask_dir=mask_dir,
        max_images=max_images,
    )
    prediction_df = predict_two_stage_dataframe(
        two_stage_result=two_stage_result,
        feature_df=inference_feature_df,
    )
    save_prediction_csv(prediction_df, output_csv)
    print(prediction_df.head().to_string(index=False))
    return prediction_df


def main() -> None:
    """Direct-running main function.

    Change the parameters below when using a different folder, model, or output
    path. The defaults match the submit.ipynb calling cell.
    """
    PROJECT_DIR = Path("./")
    MODEL_PATH = PROJECT_DIR / "model.joblib"
    RAW_IMAGE_DIR = PROJECT_DIR / "image"
    IMAGE_DIR = PROJECT_DIR / "image_processed"
    MASK_DIR = PROJECT_DIR / "mask"
    PREPROCESSED_OUTPUT_DIR = PROJECT_DIR / "image_processed"
    OUTPUT_CSV = PROJECT_DIR / "output.csv"

    # Set to an integer such as 10 for a quick test; keep None for all images.
    MAX_IMAGES = None

    # True: run preprocessing from RAW_IMAGE_DIR first.
    # False: use IMAGE_DIR directly; if IMAGE_DIR does not exist, raw images are used.
    RUN_PREPROCESSING = True

    run_inference(
        project_dir=PROJECT_DIR,
        model_path=MODEL_PATH,
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        raw_image_dir=RAW_IMAGE_DIR,
        preprocessed_output_dir=PREPROCESSED_OUTPUT_DIR,
        output_csv=OUTPUT_CSV,
        max_images=MAX_IMAGES,
        run_preprocessing=RUN_PREPROCESSING,
    )


if __name__ == "__main__":
    main()
