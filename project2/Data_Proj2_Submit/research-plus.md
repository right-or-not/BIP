##  可以优化的提升点
### 1.有关特征GLCM提取部分的优化：  
在计算图像的GLCM矩阵中：原计算方式是选择提取了包含病灶的最小矩形，在原图像上切割。在我们的lab7中的确是对提取脑部肿瘤ROI这么干的，但在使用外接矩形框取脑肿瘤时，框内的“背景”并不是真正的无效背景，而是包含了受压迫的脑白质、灰质以及水肿区。  
而在皮肤镜中，我们计算GLCM是为了展现病灶内部微观网络的异常。（区分瘤与痣）因此引入健康皮肤会造成比较严重的噪声污染。  
我们最好用掩膜去掉ROI里面的正常皮肤组织区域，然后在计算GLCM矩阵的时候，把包含0的像素对去除。随后再计算二阶值，这样可以得到更好的结果。  

#####  **参考修改代码为：（我让ai不要破坏工作流，但是不知道是不是真的）**
import cv2
import numpy as np
from typing import Sequence
from skimage.feature import graycomatrix, graycoprops

    #functions for texture features: GLCM

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
    binary_mask = _to_binary_mask(mask) # 假设此内部函数在外部已定义
    if binary_mask.shape[:2] != gray.shape[:2]:
        binary_mask = cv2.resize(binary_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

    coords = cv2.findNonZero(binary_mask)
    if coords is None:
        return [np.nan] * len(_GLCM_FEATURE_NAMES)

    x, y, w, h = cv2.boundingRect(coords)
    roi_gray = gray[y:y + h, x:x + w]
    roi_mask = binary_mask[y:y + h, x:x + w]

    # 【核心修正 1：物理隔离】利用掩膜将矩形区域内的非病灶背景像素彻底清零
    roi_gray_masked = cv2.bitwise_and(roi_gray, roi_gray, mask=roi_mask)

    roi_gray = (roi_gray_masked / 256 * levels).astype(np.uint8)
    roi_gray[roi_gray >= levels] = levels - 1

    # 【核心修正 2：关闭默认归一化】先获取原始像素对的绝对频次计数矩阵
    glcm_raw = graycomatrix(
        roi_gray,
        distances=list(distances),
        angles=list(angles),
        levels=levels,
        symmetric=True,
        normed=False, 
    )

    # 转换为浮点型以支持后续的小数概率归一化
    glcm = glcm_raw.astype(np.float64)

    # 【核心修正 3：矩阵截断】强制切除由背景（灰度级为0）产生的所有无效像素对，防止概率稀释
    glcm[0, :, :, :] = 0.0
    glcm[:, 0, :, :] = 0.0

    # 【核心修正 4：重塑概率空间】手动重新执行严密的联合概率归一化
    for d in range(glcm.shape[2]):
        for a in range(glcm.shape[3]):
            matrix_sum = np.sum(glcm[:, :, d, a])
            if matrix_sum > 0:
                glcm[:, :, d, a] = glcm[:, :, d, a] / matrix_sum

    values_by_feature = {name: [] for name in _GLCM_FEATURE_NAMES}

    for angle_idx in range(len(angles)):
        glcm_angle = glcm[:, :, 0, angle_idx]
        glcm_4d = glcm_angle.reshape(levels, levels, 1, 1)

        # skimage 的 graycoprops 可以完美兼容经过我们手动归一化后的 float64 矩阵
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

### 2.有关形状特征提取部分的优化：（重点提升优化nv和mel的分类）
在形状特征提取部分中部分出现了2个错误和1个可以补充的点：  
##### 2.1错误：不对称参数应该改为“主轴不对称参数”  
在医学图像分析中，不对称指数的计算机制是“确定病灶的质心和主轴方向。随后，将病灶图像沿其主轴进行对折，计算两半区域之间的非重叠面积”。而在我们的这部分代码中，仅对图像在水平（X轴）和垂直（Y轴）方向上进行简单的翻转。  
但皮肤病灶在图像中往往是倾斜生长的。如果一个良性痣呈 45 度角完美对称生长，水平翻转和垂直翻转代码会报告出极高的“不对称度”，从而将良性病灶误判为恶性。必须必须计算图像协方差矩阵求出特征向量（主轴），并将图像旋转对齐到主轴后再进行折叠相减，这才是真正的临床“不对称性”。  

##### 2.2错误：边缘梯度突变率
针对边缘不规则性（Border），除了分形维数，还明确列出了“边缘梯度”。  
但代码在 extract_lesion_feature_shape 中仅对二值化的掩膜（Mask）求了轮廓（cv2.findContours），完全抛弃了原图。  
恶性黑色素瘤的边缘往往表现出“侵袭性褪色”，即病灶与正常皮肤的边界并不是一刀切的，而是渐变的。这要求提取病灶轮廓后，需要回到原始灰度图中。

##### 2.3缺失：边缘分形维数
为了量化临床上的“B (Border)”，分析边缘的复杂性和侵袭性微小分支，必须“使用盒子计数法 (Box-counting Method) 计算边缘的分形维数”。  
但我们的代码仅计算了“紧凑度 (Compactness)”、“坚固度 (Solidity)”和“圆度 (Circularity)”。只能反映病灶宏观上“圆不圆”。
恶性黑色素瘤（MEL）的典型特征是边缘出现极其细小的伪足和锯齿状侵袭。这种微观边界的粗糙度与自相似性，在数字信号处理中只能通过盒子计数法的分形维数来精准捕获，目前的常规几何指标无法替代。

#####  **参考修改代码为：（不保证一定对QWQ）**
import os
import glob
import cv2
import numpy as np
from typing import Sequence

#functions for Shape features and asymmetry features

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
    """
    Shape Features
    [严格维持 12 维输出与原有命名体系]
    底层算法升级：基于二阶中心矩与特征值分解的椭圆拟合，提升对恶性病变破碎边缘的鲁棒性。
    """
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
        
        # 1. 圆度
        circularity = (4.0 * np.pi * area) / (perimeter ** 2 + eps)

        # 2. 最大直径 (优化求取方式避免 N^2 内存爆炸)
        hull_pts = cv2.convexHull(contour).reshape(-1, 2)
        if len(hull_pts) > 1:
            diff = hull_pts[:, np.newaxis, :] - hull_pts[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=-1)
            max_diameter = float(np.sqrt(np.max(dist_sq)))
        else:
            max_diameter = 0.0

        # 3. 等效直径
        equiv_diameter = float(np.sqrt(4.0 * area / np.pi)) if area > 0 else 0.0

        # 4 & 5. 长宽比与偏心率 (放弃 cv2.fitEllipse，采用严谨的图像协方差矩阵特征值分解)
        moments = cv2.moments(thresh)
        if moments["m00"] != 0:
            mu20 = moments["mu20"] / moments["m00"]
            mu02 = moments["mu02"] / moments["m00"]
            mu11 = moments["mu11"] / moments["m00"]
            
            # 构建协方差矩阵并计算特征值
            cov_matrix = np.array([[mu20, mu11], [mu11, mu02]])
            eigenvalues, _ = np.linalg.eigh(cov_matrix)
            
            # 确保特征值非负并排序 (lambda_1 >= lambda_2)
            eigenvalues = np.maximum(eigenvalues, 0.0)
            lambda_minor, lambda_major = np.sort(eigenvalues)
            
            major_axis = 4.0 * np.sqrt(lambda_major)
            minor_axis = 4.0 * np.sqrt(lambda_minor)
            
            aspect_ratio = major_axis / (minor_axis + eps)
            eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor_axis ** 2) / (major_axis ** 2 + eps))))
        else:
            aspect_ratio = 1.0
            eccentricity = 0.0

        # 6. 紧凑度
        compactness = equiv_diameter / (max_diameter + eps)
        
        # 7 & 8. 坚固度与凸包缺陷比
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / (hull_area + eps)
        defect_ratio = max(0.0, hull_area - area) / (area + eps)

        # 9 & 10. 矩形度与伸长率
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
    """
    Asymmetry Features
    [严格维持 5 维输出与原有命名体系]
    底层算法升级：asymmetry_rotation 变量实质上承载了高度校准的“沿物理主轴对折的不对称指数”。
    """
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
            mu20 = moments["mu20"] / moments["m00"]
            mu02 = moments["mu02"] / moments["m00"]
            mu11 = moments["mu11"] / moments["m00"]
        else:
            cx, cy = thresh.shape[1] // 2, thresh.shape[0] // 2
            mu20, mu02, mu11 = 1.0, 1.0, 0.0

        # 保留原有的直接轴向翻转（维护数据特征意义的一致性）
        diff_x = cv2.absdiff(thresh, cv2.flip(thresh, 1))
        diff_y = cv2.absdiff(thresh, cv2.flip(thresh, 0))
        area_diff_x = float(np.sum(diff_x == 255))
        area_diff_y = float(np.sum(diff_y == 255))

        asymmetry_x_axis = (area_diff_x / total_area) * 100.0
        asymmetry_xy_sum = ((area_diff_x + area_diff_y) / total_area) * 100.0

        left_area = float(np.sum(thresh[:, :cx] == 255))
        right_area = float(np.sum(thresh[:, cx:] == 255))
        asymmetry_area_ratio = (abs(left_area - right_area) / total_area) * 100.0

        # --- 将核心优化无缝嵌入 asymmetry_rotation 变量 ---
        # 计算物理主轴角度
        theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02) * (180 / np.pi)
        rot_mat = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
        aligned_mask = cv2.warpAffine(thresh, rot_mat, (thresh.shape[1], thresh.shape[0]), flags=cv2.INTER_NEAREST)
        
        # 沿物理主轴翻转相减
        flipped_aligned = cv2.flip(aligned_mask, 1)
        diff_principal = cv2.absdiff(aligned_mask, flipped_aligned)
        
        # 将主轴不对称指数映射到 asymmetry_rotation 输出接口
        asymmetry_rotation = (float(np.sum(diff_principal == 255)) / total_area) * 100.0

        # 饱满度
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

#concat_features, concat_feature_names, pipeline_extract_features 保持绝对原样，不做展示以避免冗余

### 3.结合相关paper后选择加入的新特征量：（补充缺失的微观特征）

#### 3.1引入Hu不对称矩作为不对称性的指标补充
Hu矩是专门适用于机器学习中描述不对称度的一种特征值，其相较于一般的asymmetry_rotation 沿主轴对折、asymmetry_fullness 饱满度。对于图像中的局部的微小毛刺具有极强的低通滤波（抗噪）特性，关注病灶整体质量分布的宏观倾斜。  
Hu矩具有“三不变性”：平移不变、缩放不变、旋转不变。对增强后的病灶图片具有很好的识别度。  
Hu矩的计算方法就不详细说明了，但它会输出7个值：  
$\phi_1$：衡量病灶整体像素的扩散度（Spread）。良性痣通常较圆润，扩散度低；恶性病变往往呈细长或不规则放射状生长，扩散度高。  
$\phi_2$：衡量病灶的细长程度（Slenderness）和方差分布。  
$\phi_3$ 与 $\phi_4$：极度关键的非对称性/偏度指标（Skewness）。 这两个值对恶性黑色素瘤向一侧不对称侵袭生长的“偏重”极其敏感。它们在数学上完美映射了 ABCD 规则中的 A（Asymmetry）。  
$\phi_5, \phi_6, \phi_7$：高阶拓扑复杂性。数值通常极小，用于捕捉非常复杂的非对称卷曲和边缘凹凸交错。  
    
#####  **参考修改代码为：（直接新建一个cell加入特征提取工作流即可）**
import cv2
import numpy as np
from typing import Sequence

def extract_lesion_feature_hu_moments(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    eps: float = 1e-8,
) -> tuple[np.ndarray, list[str]]:
    """
    Hu Moment Invariants Features
    [新增模块：严格输出 N * 7 维矩阵，与原有工作流完美兼容]
    功能：计算具有平移、缩放、旋转不变性的全局拓扑不对称性特征。
    """
    feature_rows: list[list[float]] = []
    # 生成 7 个标准的特征列名
    feature_names = [f"shape_hu_moment_{i}" for i in range(1, 8)]

    for mask in masks:
        # 复用你原有的二值化逻辑
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        total_area = float(np.sum(thresh == 255))

        # 异常拦截：如果掩膜为空，输出 7 个 0.0
        if total_area == 0:
            feature_rows.append([0.0] * 7)
            continue

        # 1. 计算原始矩与中心矩
        moments = cv2.moments(thresh)
        
        # 2. 计算 7 个 Hu 矩不变量
        hu_moments = cv2.HuMoments(moments).flatten()

        # 3. 极值对数转换 (Log Transform)
        # Hu 矩的数值范围跨度极大，且常常极小（如 1e-12）。
        # 必须使用 -sign(h) * log10(|h|) 进行数值稳定性映射，否则传统机器学习模型无法收敛。
        log_hu = []
        for h in hu_moments:
            if abs(h) < eps:
                log_hu.append(0.0)
            else:
                # 经典的 Hu 矩工程映射公式
                transformed_h = -1.0 * np.sign(h) * np.log10(abs(h))
                log_hu.append(float(transformed_h))

        feature_rows.append(log_hu)

    return np.asarray(feature_rows, dtype=np.float32), feature_names

#### 3.2 引入LBP“局部二值模式”补充纹理特征值（在多个文献中提到，能针对黑色素瘤有效提高分辨准确率。）
现有的14个GLCM虽然区分度很好，但只能量化宏观纹理。但恶性黑色素瘤中普遍存在微观的高频纹理突变。因此引入LBP捕捉微观尺度的组织纹理，能针对黑色素瘤有效提高分辨准确率。（计算方法略）  
#####  **参考修改代码为：（直接新建一个cell加入特征提取工作流即可）**
import cv2
import numpy as np
from typing import Sequence
from skimage.feature import local_binary_pattern

def extract_lesion_feature_lbp(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    P: int = 8,
    R: int = 1,
    method: str = 'uniform',
    eps: float = 1e-8
) -> tuple[np.ndarray, list[str]]:
    """
    Local Binary Pattern (LBP) Texture Features
    [新增模块：输出 N * 59 维纹理特征直方图，与原有工作流完美兼容]
    功能：统计病灶内部微观纹理分布，捕捉恶性病变导致的色素网络崩塌与高频突变。
    """
    # 根据 Uniform 模式计算特征维度 (当 P=8 时，n_bins = 59)
    if method == 'uniform':
        n_bins = P * (P - 1) + 3
    elif method == 'default':
        n_bins = 2 ** P
    else:
        n_bins = P + 2

    feature_rows: list[list[float]] = []
    # 生成标准的特征列名
    feature_names = [f"texture_lbp_bin_{i}" for i in range(n_bins)]

    for img, mask in zip(images, masks):
        # 图像灰度化预处理
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # 掩膜二值化预处理
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 异常拦截：如果掩膜为空，输出全 0
        if np.sum(thresh == 255) == 0:
            feature_rows.append([0.0] * n_bins)
            continue

        # 1. 计算整张图像的 LBP 矩阵
        # 使用 skimage 的高效 C 语言底层实现
        lbp_matrix = local_binary_pattern(gray, P, R, method)

        # 2. 掩膜截断提取：仅保留真实病灶区域内的 LBP 编码
        # 这一步极其关键，剔除了背景和健康皮肤带来的庞大噪音
        lbp_masked_pixels = lbp_matrix[thresh == 255]

        # 3. 统计频次直方图
        hist, _ = np.histogram(lbp_masked_pixels, bins=n_bins, range=(0, n_bins))

        # 4. 直方图归一化 (概率分布)
        # 将频次转换为占比，消除不同病灶大小对特征绝对数值的干扰
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + eps)

        feature_rows.append(hist.tolist())

    return np.asarray(feature_rows, dtype=np.float32), feature_names

#### 3.3对于色彩特征，引入“超像素聚类”（补充Color）
我们现有的 399 个颜色特征全是全局统计特征，这无法告诉分类器“病灶内是否存在孤立的蓝灰色团块”，而mel常表现出多色性斑块，因此引入K-Means 或 SLIC 将病灶划分为几个色块并计算这些特定颜色域的面积占比。  
(计算方式略)  
计算会输出多个超像素斑块后，我们将这些色块其与皮肤科临床公认的 6 种主导颜色（浅棕、深棕、黑、白、蓝灰、红）的理想中心点进行欧氏距离匹配 。最后，统计这 6 种颜色在病灶总面积中的百分比。  
这 6 个最终输出的百分比特征，具有极高的医学可解释性，完美映射了 ABCD 规则中的 C (Color Variegation, 颜色异质性)：  
**蓝灰色占比（Blue-Gray %）**： 直接量化“蓝白面纱”与“深层黑色素吞噬”现象，这是恶性黑色素瘤（MEL）最危险的特异性指标。  
**白色占比（White %）**： 映射“退化结构（Regression structures）”，即免疫系统攻击肿瘤后留下的瘢痕区。
**红色占比（Red %）**： 捕捉红蓝腔隙（Red-blue lacunae），这是将血管病变（VASC）从色素性病变中绝对剥离出来的黄金特征 。  
**深/浅棕色失衡：** 良性痣（NV）通常由单一颜色的棕色主导（占比通常 $>90\%$），而恶性病变的多种棕/黑色比例往往呈现混杂状态。  

#####  **参考修改代码为：（直接新建一个cell加入特征提取工作流即可）**  
import cv2
import numpy as np
from typing import Sequence
from skimage.segmentation import slic
from skimage.color import rgb2lab

def extract_lesion_feature_superpixel_color(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    n_segments: int = 400,  # 控制超像素的精细度，数值越大斑点越小
    compactness: float = 10.0,
    eps: float = 1e-8
) -> tuple[np.ndarray, list[str]]:
    """
    Superpixel Color Proportion Features (SLIC)
    [新增模块：输出 N * 6 维特征，与原有工作流完美兼容]
    功能：基于 SLIC 超像素聚类，统计皮肤镜 6 种经典临床颜色的面积占比。
    """
    feature_rows: list[list[float]] = []
    
    # 严格映射临床的 6 种核心颜色
    color_keys = ["white", "black", "light_brown", "dark_brown", "blue_gray", "red"]
    feature_names = [f"color_proportion_{color}" for color in color_keys]

    # 预定义 6 种临床颜色的 CIELAB 经验中心点 (L: 0~100, a: -128~127, b: -128~127)
    clinical_lab_centers = np.array([
        [85.0,   0.0,   0.0],  # White (退化结构)
        [15.0,   0.0,   0.0],  # Black (浅表高密度黑色素)
        [65.0,  15.0,  30.0],  # Light Brown (正常色素网)
        [35.0,  15.0,  20.0],  # Dark Brown (致密色素)
        [50.0,  -5.0, -15.0],  # Blue-Gray (蓝白面纱/深层黑色素)
        [50.0,  50.0,  20.0]   # Red (血管病变 VASC 的腔隙)
    ])

    for img, mask in zip(images, masks):
        # 掩膜二值化处理
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        valid_area = np.sum(thresh == 255)
        if valid_area == 0:
            feature_rows.append([0.0] * len(feature_names))
            continue

        # 1. 对原始 RGB 图像执行 SLIC 超像素聚类
        # 注意：SLIC 需要 RGB 顺序的图像，确保输入图像通道正确
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        segments = slic(img_rgb, n_segments=n_segments, compactness=compactness, start_label=1)
        
        # 将图像转换至 LAB 空间用于颜色距离计算
        img_lab = rgb2lab(img_rgb)
        
        # 初始化当前图像的颜色计数器
        color_area_counts = np.zeros(len(color_keys), dtype=np.float32)

        # 2. 遍历每一个生成的超像素斑块
        unique_segments = np.unique(segments)
        for seg_id in unique_segments:
            # 找到当前超像素斑块与病灶掩膜的交集
            seg_mask = (segments == seg_id) & (thresh == 255)
            seg_area = np.sum(seg_mask)
            
            # 如果该斑块完全在背景中，则跳过
            if seg_area == 0:
                continue
                
            # 计算该超像素斑块在 LAB 空间的均值颜色
            mean_l = np.mean(img_lab[:, :, 0][seg_mask])
            mean_a = np.mean(img_lab[:, :, 1][seg_mask])
            mean_b = np.mean(img_lab[:, :, 2][seg_mask])
            mean_color = np.array([mean_l, mean_a, mean_b])
            
            # 3. 寻找欧氏距离最近的临床颜色中心
            distances = np.linalg.norm(clinical_lab_centers - mean_color, axis=1)
            closest_color_idx = np.argmin(distances)
            
            # 将该斑块的面积累加到对应的临床颜色统计中
            color_area_counts[closest_color_idx] += seg_area

        # 4. 归一化：转换为病灶内部的面积百分比
        color_proportions = color_area_counts / (valid_area + eps)
        feature_rows.append(color_proportions.tolist())

    return np.asarray(feature_rows, dtype=np.float32), feature_names  