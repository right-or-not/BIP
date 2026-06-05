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
