# Project 2: Skin Lesion Classification

本项目实现了一个基于传统图像特征和机器学习模型的皮肤病灶分类流程。整体代码集中在 `submit.ipynb` 中，按照以下流程组织：

1. 图像预处理
2. 特征提取
3. 三分类模型训练
4. 特征评估与筛选
5. 两阶段二分类策略
6. 综合分类结果评估

数据包含 200 张原始皮肤病图像和 400 张增强图像。文件名中相同数字前缀表示同一个原始病灶或患者，例如 `12.jpg`、`12_aug1.jpg`、`12_aug2.jpg` 属于同一组。因此模型训练和测试划分均使用 `base_id` 分组，避免增强图像泄漏到不同集合。

## 1. 图像预处理

本部分对原始图像进行颜色校正、去噪和毛发去除，输出预处理后的图像。

主要函数：

| 函数 | 任务 | 输入 | 输出 |
|---|---|---|---|
| `shades_of_gray` | Shades of Gray 颜色恒常性校正 | 单张 BGR 图像 `np.ndarray` | 校正后的 BGR 图像 `np.ndarray` |
| `dull_razor_hair_removal` | 使用 Dull-Razor 方法检测并修复毛发区域 | 单张 BGR 图像 `np.ndarray` | 去毛发后的 BGR 图像 `np.ndarray` |
| `pipeline_preview` | 批量执行颜色校正、中值滤波、毛发去除 | 输入图像目录、输出目录 | 将处理后的 `.jpg` 保存到 `image_processed/` |

输出数据格式：

- 图像文件保存为 `.jpg`
- 输出目录为 `image_processed/`
- 文件名保持和原始图像一致，便于和 `mask/`、`label.csv` 对齐

## 2. 特征提取

本部分从预处理图像和病灶 mask 中提取形态、不对称、颜色和纹理特征，并合并为统一的特征表。

### 2.1 形态特征与不对称特征

主要函数：

| 函数 | 任务 | 输出 |
|---|---|---|
| `_to_binary_mask` | 将 mask 转为二值图 | 二值 mask `np.ndarray` |
| `extract_lesion_feature_shape` | 提取面积、周长、圆度、直径、长宽比、偏心率、紧致度等形态特征 | `(features, feature_names)` |
| `extract_lesion_feature_asymmetry` | 提取左右/上下/旋转不对称特征 | `(features, feature_names)` |

输出格式：

```python
features: np.ndarray          # shape = (n_samples, n_features)
feature_names: list[str]      # 每一列特征对应的名称
```

### 2.2 颜色特征

主要函数：

| 函数 | 任务 |
|---|---|
| `_normalized_rgb` | 计算归一化 RGB |
| `_ohta_color_space` | 转换到 Ohta 颜色空间 |
| `_gevers_l123` | 计算 Gevers L1/L2/L3 颜色不变量 |
| `_build_peripheral_regions` | 构建病灶周围内环、外环区域 |
| `_extract_single_color_features` | 提取单张图像的颜色统计特征 |
| `extract_lesion_feature_color` | 批量提取颜色特征 |

颜色特征覆盖 RGB、HSV、Lab、Luv、YCbCr、Ohta、Gevers 等空间。统计量包括 `min`、`max`、`mean`、`std`、`var`、`skew`，并包含病灶区域与周边区域之间的差值和比值特征。

输出格式同样为：

```python
features: np.ndarray
feature_names: list[str]
```

### 2.3 纹理特征：GLCM

主要函数：

| 函数 | 任务 |
|---|---|
| `_compute_single_glcm_features` | 对单张图像的病灶 ROI 计算 GLCM 特征 |
| `extract_lesion_feature_glcm` | 批量提取 GLCM 纹理特征 |

主要 GLCM 特征包括：

- `ASM`
- `Contrast`
- `Correlation`
- `Homogeneity`
- `Dissimilarity`
- `Entropy`
- `MaxProbability`
- `Variance`
- `SumVariance`
- `SumEntropy`
- `DifferenceVariance`
- `DifferenceEntropy`
- `IMCorr1`
- `IMCorr2`

### 2.4 特征合并与特征表

主要函数：

| 函数 | 任务 | 输出 |
|---|---|---|
| `concat_features` | 横向合并多组特征矩阵 | `np.ndarray` |
| `concat_feature_names` | 合并多组特征名 | `list[str]` |
| `show_features_information` | 展示特征数量、缺失值、统计摘要 | `pd.DataFrame` |

最终特征表格式：

```python
all_features: np.ndarray
all_feature_names: list[str]

feature_df: pd.DataFrame
```

`feature_df` 包含以下元信息列：

| 列名 | 含义 |
|---|---|
| `filename` | 图像文件名 |
| `image_id` | 去掉扩展名后的图像 ID，例如 `12_aug1` |
| `base_id` | 原始病灶 ID，例如 `12` |

其余列为提取出的全部图像特征。

## 3. 学习分类

本部分实现三类标签 `mel`、`nv`、`vasc` 的传统机器学习分类。模型包括 SVM、随机森林和 XGBoost。

### 3.0 辅助函数

主要函数：

| 函数 | 任务 | 输出 |
|---|---|---|
| `prepare_feature_matrix` | 将特征转成有限的 `float32` 矩阵，并处理 NaN/Inf | `np.ndarray` |
| `_validate_and_prepare_groups` | 检查分组长度和同组标签一致性 | `(labels, groups)` |
| `split_classification_dataset` | 按 `base_id` 分组划分训练集和测试集 | `x_train, x_test, y_train, y_test` |
| `attach_labels_from_csv` | 将 `label.csv` 标签合并到 `feature_df` | `labeled_feature_df: pd.DataFrame` |
| `build_training_result` | 统一封装 SVM、随机森林、XGBoost 的训练结果 | `dict` |
| `predict_model_labels` | 使用统一训练结果或裸模型进行预测 | `np.ndarray` |
| `evaluate_model` | 统一计算准确率、混淆矩阵和分类报告 | `dict` |
| `print_model_evaluation` | 统一打印模型评估结果 | 控制台输出 |
| `print_top_features` | 统一打印前 N 个特征重要性 | 控制台输出 |

训练/测试集格式：

```python
x_train: np.ndarray
x_test: np.ndarray
y_train: np.ndarray
y_test: np.ndarray
```

其中，如果传入 `groups=base_ids`，同一个 `base_id` 的原图和增强图会被整体放入训练集或测试集。

### 3.1-3.3 三个模型的统一函数模板

SVM、随机森林和 XGBoost 都按同一个函数模板组织。不同模型只替换模型名称，输入输出保持一致。

统一模板如下：

| 模板函数 | SVM | 随机森林 | XGBoost | 任务 | 输出 |
|---|---|---|---|---|---|
| `build_*_classifier` | `build_svm_classifier` | `build_random_forest_classifier` | `build_xgboost_classifier` | 创建未训练模型 | 模型对象 |
| `train_*_model` | `train_svm_model` | `train_random_forest_model` | `train_xgboost_model` | 训练模型 | 训练后的模型对象 |
| `get_*_feature_importance` | `get_svm_feature_importance` | `get_random_forest_feature_importance` | `get_xgboost_feature_importance` | 获取特征重要性 | `list[tuple[str, float]]` 或 `None` |
| `run_*_training_pipeline` | `run_svm_training_pipeline` | `run_random_forest_training_pipeline` | `run_xgboost_training_pipeline` | 分组划分、训练、封装结果 | 统一训练结果 `dict` |
| `optimize_*_hyperparameters` | `optimize_svm_hyperparameters` | `optimize_random_forest_hyperparameters` | `optimize_xgboost_hyperparameters` | GridSearchCV 超参数搜索 | 调参结果 `dict` |

其中评估和打印不再按模型分别实现一套新逻辑，而是统一使用 3.0 中的通用函数：

| 通用函数 | 任务 | 输出 |
|---|---|---|
| `evaluate_model` | 对任意统一训练结果或裸模型进行预测和评估 | 评估结果 `dict` |
| `print_model_evaluation` | 打印 hit rate、混淆矩阵、classification report | 控制台输出 |
| `print_top_features` | 打印前 N 个特征重要性 | 控制台输出 |

因此后续统一写法为：

```python
evaluation = evaluate_model(
    model_result,
    model_result['x_test'],
    model_result['y_test'],
    label_order=['mel', 'nv', 'vasc'],
)
print_model_evaluation(evaluation, title='Model evaluation')
print_top_features(model_result['feature_importances'], top_n=20)
```

### 自动调参与最终测试

三个模型的自动调参 cell 都分为两步：

1. 使用 `optimize_*_hyperparameters(...)` 通过 `GridSearchCV` 搜索最优超参数。
2. 使用搜索得到的 `best_params` 重新调用 `run_*_training_pipeline(...)`，在统一的 hold-out 测试集上输出最终 `Hit rate`。

需要区分两个指标：

| 指标 | 来源 | 含义 |
|---|---|---|
| `Best CV score` | `GridSearchCV` 的交叉验证平均分 | 用于选择哪组超参数更好 |
| `Hit rate` | `evaluate_model` 在 hold-out 测试集上的准确率 | 用于观察该组超参数在最终测试集上的表现 |

因此 `Best CV score` 不一定等于最终 `Hit rate`。前者来自交叉验证，后者来自一次固定的训练/测试划分。

自动调参时，如果传入 `groups=*_groups`，SVM、随机森林和 XGBoost 都会使用 `StratifiedGroupKFold`，避免同一个 `base_id` 的原图和增强图出现在不同 CV fold 中。

### 3.1 SVM

SVM 使用 `StandardScaler + SVC` 的 `Pipeline`。函数接口遵循上面的统一模板：

```python
svm_model = train_svm_model(x_train, y_train, ...)
svm_result = run_svm_training_pipeline(features, labels, feature_names, groups=base_ids, ...)
```

SVM 的特征重要性说明：

- 当 `kernel='linear'` 时，`get_svm_feature_importance` 可根据 `coef_` 计算特征重要性。
- 当使用 `rbf` 等非线性核时，没有直接可解释的线性系数，因此 `feature_importances` 为 `None`。

### 3.2 随机森林

随机森林使用 `RandomForestClassifier`，函数接口遵循统一模板：

```python
rf_model = train_random_forest_model(x_train, y_train, ...)
rf_result = run_random_forest_training_pipeline(features, labels, feature_names, groups=base_ids, ...)
```

随机森林原生支持 `feature_importances_`，因此：

```python
rf_result['feature_importances']  # list[tuple[str, float]]
```

自动优化超参数时使用 `optimize_random_forest_hyperparameters`。该函数中：

- `n_jobs` 控制 `GridSearchCV` 的外层并行，用来并行测试不同参数组合。
- `estimator_n_jobs` 控制单个 `RandomForestClassifier` 内部并行，默认设为 `1`。

这样可以避免 `GridSearchCV(n_jobs=-1)` 和 `RandomForestClassifier(n_jobs=-1)` 同时并行导致的嵌套并行 warning 和 CPU 资源抢占。一般保持 `estimator_n_jobs=1` 即可。

### 3.3 XGBoost

XGBoost 使用 `XGBClassifier`，函数接口遵循统一模板：

```python
xgb_model = train_xgboost_model(x_train, y_train, ...)
xgb_result = run_xgboost_training_pipeline(features, labels, feature_names, groups=base_ids, ...)
```

XGBoost 的标签处理和目标函数：

- `train_xgboost_model` 内部使用 `LabelEncoder` 将字符串标签转成数字标签。
- 训练后的 `label_encoder` 会通过统一结果字典保存到 `xgb_result['label_encoder']`。
- 二分类自动使用 `binary:logistic`。
- 多分类自动使用 `multi:softprob`。

XGBoost 原生支持 `feature_importances_`，因此：

```python
xgb_result['feature_importances']  # list[tuple[str, float]]
```

XGBoost 可以尝试使用 GPU 加速。当前函数支持以下参数：

| 参数 | 含义 | 常用取值 |
|---|---|---|
| `device` | 选择 XGBoost 训练设备。`'cpu'` 使用 CPU，`'cuda'` 尝试使用支持 CUDA 的 GPU。 | `'cpu'`, `'cuda'` |
| `tree_method` | 树构建算法。XGBoost 2.x/3.x 推荐配合 GPU 使用 `tree_method='hist'`。 | `'hist'` |
| `n_jobs` | 单个 XGBoost 模型的 CPU 线程数。即使使用 GPU，也仍可能有 CPU 侧开销。 | `-1`, `1`, `4` |

在 XGBoost 自动调参 cell 中：

```python
XGB_DEVICE = 'cpu'
XGB_TREE_METHOD = 'hist'
XGB_GRID_N_JOBS = 1 if XGB_DEVICE == 'cuda' else -1
```

如果已经确认当前 Python 环境可以看到 CUDA GPU，可以改为：

```python
XGB_DEVICE = 'cuda'
```

注意：GPU 加速只对 XGBoost 有直接帮助。当前使用的 scikit-learn `SVC` 和 `RandomForestClassifier` 不会因为机器有独显就自动使用 GPU。并且在 GPU 调参时，建议保持 `GridSearchCV` 的 `n_jobs=1`，避免同时启动多个 XGBoost 训练任务抢同一块 GPU。

### 3.4 手动调参测试代码

每个模型在自动搜索超参数的测试代码之外，都提供了一个手动调参测试 cell。手动调参时，主要修改两类参数：

1. `*_MANUAL_PARAMS`：模型自身的超参数。
2. `run_*_training_pipeline(...)`：训练流程参数，包括特征、标签、分组划分比例和随机种子。

#### SVM 手动调参参数

示例：

```python
SVM_MANUAL_PARAMS = {
    'c': 10.0,
    'kernel': 'rbf',
    'gamma': 'scale',
}
```

| 参数 | 含义 | 常用取值 |
|---|---|---|
| `c` | SVM 惩罚系数。越大越强调训练集分类正确，可能过拟合；越小越保守，可能欠拟合。 | `0.1`, `1.0`, `10.0`, `100.0` |
| `kernel` | 核函数，决定分类边界形状。 | `'linear'`, `'rbf'`, `'poly'`, `'sigmoid'` |
| `gamma` | 样本影响范围，主要用于 `'rbf'`、`'poly'`、`'sigmoid'`。越大边界越复杂，越小边界越平滑。 | `'scale'`, `'auto'`, `0.001`, `0.01`, `0.1`, `1.0` |

注意：当 `kernel='linear'` 时，`gamma` 基本不起作用；只有线性 SVM 可以直接输出基于系数的 `feature_importances`。

#### 随机森林手动调参参数

示例：

```python
RF_MANUAL_PARAMS = {
    'n_estimators': 800,
    'max_depth': None,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'max_features': 'sqrt',
}
```

| 参数 | 含义 | 常用取值 |
|---|---|---|
| `n_estimators` | 森林中决策树数量。越大通常越稳定，但训练更慢。 | `200`, `500`, `800`, `1000` |
| `max_depth` | 单棵树最大深度。限制深度可以减少过拟合。 | `None`, `3`, `5`, `8`, `10`, `15` |
| `min_samples_leaf` | 叶子节点至少包含的样本数。越大越平滑，越不容易过拟合。 | `1`, `2`, `4`, `8` |
| `min_samples_split` | 内部节点继续分裂所需的最小样本数。 | `2`, `4`, `8`, `10` |
| `max_features` | 每次分裂时可使用的最大特征数。 | `'sqrt'`, `'log2'`, `None`, `0.3`, `0.5` |

随机森林一般先尝试增大 `n_estimators`，再调小 `max_depth` 或增大 `min_samples_leaf` 来控制过拟合。

#### XGBoost 手动调参参数

示例：

```python
XGB_MANUAL_PARAMS = {
    'n_estimators': 500,
    'max_depth': 2,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 3.0,
    'reg_alpha': 0.0,
}
```

| 参数 | 含义 | 常用取值 |
|---|---|---|
| `n_estimators` | boosting 轮数，即树的数量。数量越多模型越复杂，训练更慢。 | `100`, `300`, `500`, `800` |
| `max_depth` | 单棵树最大深度。小数据集上通常不宜过大。 | `2`, `3`, `4`, `5`, `6` |
| `learning_rate` | 学习率。越小训练越稳，通常需要更多树。 | `0.01`, `0.03`, `0.05`, `0.1` |
| `subsample` | 每棵树训练时采样的样本比例。小于 1 可减少过拟合。 | `0.6`, `0.8`, `0.9`, `1.0` |
| `colsample_bytree` | 每棵树训练时采样的特征比例。小于 1 可减少过拟合。 | `0.6`, `0.8`, `0.9`, `1.0` |
| `reg_lambda` | L2 正则化系数。越大模型越保守。 | `0.5`, `1.0`, `3.0`, `5.0`, `10.0` |
| `reg_alpha` | L1 正则化系数。可促使模型更稀疏。 | `0.0`, `0.1`, `0.5`, `1.0` |

XGBoost 在当前代码中会根据标签数量自动选择目标函数：二分类使用 `binary:logistic`，多分类使用 `multi:softprob`。

#### `run_*_training_pipeline` 共用关键参数

三个模型的手动调参 cell 都调用对应的 `run_*_training_pipeline(...)`。这些参数不是模型超参数本身，而是控制训练数据、测试集划分、随机种子和模型参数传入方式。

典型调用方式如下：

```python
svm_manual_result = run_svm_training_pipeline(
    features=svm_manual_features,
    labels=svm_manual_labels,
    feature_names=all_feature_names,
    groups=svm_manual_groups,
    test_size=0.3,
    split_random_state=42,
    model_random_state=42,
    **SVM_MANUAL_PARAMS,
)
```

pipeline 参数说明：

| 参数 | 含义 | 常用取值 |
|---|---|---|
| `features` | 输入模型训练的特征矩阵，格式为 `np.ndarray`，形状通常是 `(n_samples, n_features)`。如果想切换为筛选后的特征，需要同时修改 `features` 和 `feature_names`。 | `svm_manual_features`, `rf_manual_features`, `xgb_manual_features`, `selected_feature_matrix` |
| `labels` | 每个样本对应的标签数组，长度必须和 `features` 的样本数一致。三分类为 `mel/nv/vasc`，二分类可以是 `vasc/other` 或 `mel/nv`。 | `svm_manual_labels`, `rf_manual_labels`, `xgb_manual_labels` |
| `feature_names` | 特征名列表，顺序必须和 `features` 的列顺序完全一致。用于保存训练结果和输出特征重要性。 | `all_feature_names`, `selected_feature_names` |
| `groups` | 分组 ID，当前使用 `base_id`。相同 `base_id` 的原图和增强图会被整体划分到训练集或测试集，避免数据泄漏。 | `svm_manual_groups`, `rf_manual_groups`, `xgb_manual_groups` |
| `test_size` | 测试集占比。值越大，测试集更多但训练集更少；值越小，训练集更多但测试结果波动可能更大。 | `0.2`, `0.25`, `0.3`, `0.35` |
| `split_random_state` | 训练集/测试集划分的随机种子。修改它会改变哪些 `base_id` 被分到训练集或测试集，可用于检查模型结果是否稳定。 | `0`, `1`, `42`, `2024` |
| `model_random_state` | 模型训练本身的随机种子。随机森林和 XGBoost 对它更敏感；SVM 一般影响较小。固定该值可以让结果更容易复现。 | `0`, `1`, `42`, `2024` |
| `**SVM_MANUAL_PARAMS` / `**RF_MANUAL_PARAMS` / `**XGB_MANUAL_PARAMS` | 将手动设置的模型超参数展开后传入对应 pipeline。这里面的键必须和对应 `run_*_training_pipeline` 函数支持的参数名一致。 | `c`, `kernel`, `gamma`; `n_estimators`, `max_depth`; `learning_rate`, `subsample` 等 |

通常建议保持 `groups=*_manual_groups`，避免同一 `base_id` 的增强图片泄漏到训练集和测试集两侧。

如果要从全部特征切换到筛选后特征，应该成对修改：

```python
features=selected_feature_matrix
feature_names=selected_feature_names
```

如果只想观察不同划分带来的结果变化，优先修改：

```python
split_random_state=0
```

如果只想调模型能力，保持划分参数不变，只修改：

```python
**SVM_MANUAL_PARAMS
**RF_MANUAL_PARAMS
**XGB_MANUAL_PARAMS
```

### 统一训练结果格式

三个模型的 `run_*_training_pipeline` 输出完全一致：

```python
model_result = {
    'model': model,
    'label_encoder': label_encoder,
    'x_train': x_train,
    'x_test': x_test,
    'y_train': y_train,
    'y_test': y_test,
    'feature_names': list(feature_names),
    'feature_importances': feature_importances,
}
```

字段含义：

| 键 | 含义 |
|---|---|
| `model` | 训练后的模型对象。SVM 为 `Pipeline`，随机森林为 `RandomForestClassifier`，XGBoost 为 `XGBClassifier`。 |
| `label_encoder` | 标签编码器。用于保持三个模型结果格式一致，也用于 XGBoost 数字预测转回字符串标签。 |
| `x_train` / `x_test` | 训练集和测试集特征矩阵，类型为 `np.ndarray`。 |
| `y_train` / `y_test` | 训练集和测试集真实标签。 |
| `feature_names` | 当前模型训练使用的特征名，顺序与特征矩阵列顺序一致。 |
| `feature_importances` | 特征重要性列表。树模型为原生重要性；线性 SVM 为系数重要性；非线性 SVM 为 `None`。 |

### 统一评估结果格式

三个模型统一使用 `evaluate_model`，返回格式为：

```python
evaluation = {
    'hit_rate': float,
    'confusion_matrix': np.ndarray,
    'classification_report': str,
    'classification_report_dict': dict,
    'labels': list[str],
    'y_pred': np.ndarray,
}
```

字段含义：

| 键 | 含义 |
|---|---|
| `hit_rate` | 准确率，即预测正确样本数 / 总测试样本数。 |
| `confusion_matrix` | 混淆矩阵。 |
| `classification_report` | 文本格式的 precision、recall、F1-score 报告。 |
| `classification_report_dict` | 字典格式的 classification report，便于后续程序读取。 |
| `labels` | 评估时使用的标签顺序。 |
| `y_pred` | 模型对测试集的预测标签。 |

## 4. 特征评估与筛选

本部分用于从全部特征中筛选出更适合 `vasc` vs `other` 的特征集合。

### 4.0 辅助函数

主要函数：

| 函数 | 任务 |
|---|---|
| `feature_tuple_to_dict` | `(features, feature_names)` 转为 `{feature_name: values}` |
| `feature_dict_to_tuple` | `{feature_name: values}` 转回矩阵和名称 |
| `feature_tuple_to_dataframe` | 特征矩阵转为 DataFrame |
| `feature_dataframe_to_tuple` | DataFrame 转回特征矩阵 |
| `build_group_labels_from_base_ids` | 构造 grouped CV 使用的 group 标签 |
| `get_feature_category` | 判断特征类别：`shape`、`texture`、`color` |

### 4.1 特征评估与筛选工作流

主要函数：

| 函数 | 任务 | 输出 |
|---|---|---|
| `filter_low_variance_features` | 去除低方差特征 | `(features, names, variance_df)` |
| `bh_fdr_correction` | Benjamini-Hochberg FDR 校正 | `(adjusted_pvalues, reject)` |
| `compute_binary_feature_metrics` | 计算单变量特征指标 | `scores_df: pd.DataFrame` |
| `list_high_corr_pairs` | 找高相关特征对 | `list[tuple]` |
| `remove_high_corr_features_by_score` | 按综合分数删除高相关冗余特征 | `(selected_features, removed_pairs)` |
| `evaluate_top_k_feature_sets` | 使用 grouped CV 评估不同 top-k 特征集合 | `pd.DataFrame` |
| `choose_final_feature_set` | 选择最终特征集合 | `pd.Series` |
| `run_feature_evaluation_selection_pipeline` | 完整特征筛选流程 | `dict` |
| `print_feature_selection_summary` | 打印筛选摘要 | 控制台输出 |

单变量评估表 `scores_df` 包含：

```python
feature
category
variance
F_score
F_pvalue
MI
AUC
Cohens_d
MW_pvalue
KS_stat
positive_mean
negative_mean
composite_score
MW_pvalue_FDR
FDR_significant
is_strong_univariate
```

最终特征筛选结果格式：

```python
feature_selection_result = {
    "variance_df": pd.DataFrame,
    "scores_df": pd.DataFrame,
    "pearson_high_corr": list,
    "spearman_high_corr": list,
    "k_results_df": pd.DataFrame,
    "selected_features": list[str],
    "selected_feature_matrix": np.ndarray,
    "filtered_feature_names": list[str],
}
```

后续训练中使用：

```python
selected_feature_names = feature_selection_result["selected_features"]
```

## 5. 逐步分类

本节已重构为通用的两阶段二分类框架。总体目标是把一个三分类问题拆成两个二分类问题：

1. 第一阶段：从三类中先分出一个指定标签，例如 `vasc` vs `other`
2. 第二阶段：对剩余两个指定标签继续二分类，例如 `mel` vs `nv`
3. 综合阶段：如果第一阶段预测为指定标签，最终结果直接取该标签；否则进入第二阶段得到另一个标签

两个阶段都可以独立配置：

- 第一阶段分出的标签是哪一个
- 第二阶段继续分类的两个标签是哪两个
- 第一阶段和第二阶段各自使用哪些特征
- 第一阶段和第二阶段各自使用什么模型
- 第一阶段和第二阶段各自的模型参数和调参网格

### 5.0 两阶段数据划分函数

本部分提供两个核心划分函数：

| 函数 | 任务 | 输出 |
|---|---|---|
| `prepare_first_stage_split` | 按 `base_id` 分组划分训练/测试集，并把标签转成 `first_positive_label` vs `negative_label` | 第一阶段 `dict` |
| `prepare_second_stage_split` | 复用第一阶段的 train/test group，只保留第二阶段的两个真实标签 | 第二阶段 `dict` |

第一阶段输出结构：

```python
first_stage_split = {
    "x_train": np.ndarray,
    "x_test": np.ndarray,
    "y_train": np.ndarray,           # first_positive_label / negative_label
    "y_test": np.ndarray,
    "y_train_original": np.ndarray,  # 原始三分类标签
    "y_test_original": np.ndarray,
    "train_indices": np.ndarray,
    "test_indices": np.ndarray,
    "train_groups": np.ndarray,
    "test_groups": np.ndarray,
    "train_group_ids": np.ndarray,
    "test_group_ids": np.ndarray,
    "feature_names": list[str],
    "positive_label": str,
    "negative_label": str,
}
```

第二阶段输出结构：

```python
second_stage_split = {
    "x_train": np.ndarray,
    "x_test": np.ndarray,
    "y_train": np.ndarray,       # second_labels 中的两个标签
    "y_test": np.ndarray,
    "train_indices": np.ndarray,
    "test_indices": np.ndarray,
    "train_groups": np.ndarray,
    "test_groups": np.ndarray,
    "train_group_ids": np.ndarray,
    "test_group_ids": np.ndarray,
    "feature_names": list[str],
    "second_labels": tuple[str, str],
}
```

调试重点：

```python
len(set(first_stage_split["train_groups"]) & set(first_stage_split["test_groups"]))
len(set(second_stage_split["train_groups"]) & set(second_stage_split["test_groups"]))
```

这两个值都应该为 `0`，表示同一原始病灶及其增强图没有同时进入训练集和测试集。代码中也提供了 `assert_no_group_leakage`，一旦某个 `base_id` 同时出现在训练集和测试集，会直接抛出错误。

### 5.1 两阶段模型训练函数

本部分提供一个统一训练入口：

| 函数 | 任务 |
|---|---|
| `_normalize_model_type` | 将 `svm`、`rf`、`xgb` 等别名归一化 |
| `train_stage_classifier` | 训练单个二分类模型，并封装成统一训练结果 |
| `run_two_stage_training_pipeline` | 一次性完成两个阶段的数据划分、模型训练和结果封装 |

支持的模型类型：

```python
'svm'
'random_forest'
'xgboost'
```

统一训练示例：

```python
two_stage_result = run_two_stage_training_pipeline(
    feature_df=labeled_feature_df,
    first_positive_label='vasc',
    second_labels=('mel', 'nv'),
    first_feature_names=first_stage_feature_names,
    second_feature_names=second_stage_feature_names,
    first_model_type='random_forest',
    second_model_type='random_forest',
    first_model_params=FIRST_MODEL_PARAMS,
    second_model_params=SECOND_MODEL_PARAMS,
    negative_label='other',
    test_size=0.3,
    split_random_state=42,
    model_random_state=42,
)
```

`two_stage_result` 的主要结构：

```python
two_stage_result = {
    "first_stage_split": dict,
    "second_stage_split": dict,
    "first_model_result": dict,
    "second_model_result": dict,
    "first_model_type": str,
    "second_model_type": str,
    "first_model_params": dict,
    "second_model_params": dict,
    "first_positive_label": str,
    "negative_label": str,
    "second_labels": tuple[str, str],
    "first_feature_names": list[str],
    "second_feature_names": list[str],
}
```

### 5.2 两阶段评估函数

本部分提供两个评估函数：

| 函数 | 任务 | 输出 |
|---|---|---|
| `evaluate_two_stage_models` | 分别评估第一阶段、第二阶段和最终串联三分类结果 | `dict` |
| `print_two_stage_evaluation_results` | 打印三部分 hit rate、混淆矩阵、classification report 和特征重要性 | 控制台输出 |

综合预测逻辑：

```python
if first_stage_prediction == first_positive_label:
    final_prediction = first_positive_label
else:
    final_prediction = second_stage_model.predict(second_stage_features)
```

评估输出结构：

```python
two_stage_evaluation = {
    "first_stage": dict,   # 第一阶段二分类 evaluate_model 结果
    "second_stage": dict,  # 第二阶段二分类 evaluate_model 结果
    "combined": {
        "hit_rate": float,
        "confusion_matrix": np.ndarray,
        "classification_report": str,
        "classification_report_dict": dict,
        "labels": list[str],
        "y_true": np.ndarray,
        "y_pred": np.ndarray,
        "first_stage_positive_count": int,
        "second_stage_routed_count": int,
    },
}
```

建议每次调试都同时查看三块结果：

- 第一阶段是否把指定标签分出来
- 第二阶段在真实剩余两类上是否足够稳定
- 最终三分类是否因为第一阶段误分而影响整体结果

### 5.3 两阶段联合超参数优化函数

本部分提供统一调参入口：

| 函数 | 任务 |
|---|---|
| `optimize_stage_hyperparameters` | 对单个阶段调用对应模型的 `GridSearchCV` 调参函数 |
| `_convert_stage_search_params_to_training_params` | 将搜索参数名转换为训练函数参数名，主要用于 SVM |
| `optimize_two_stage_hyperparameters` | 同时优化两个阶段，重新训练，并输出最终三分类评估 |

调参示例：

```python
two_stage_optimization = optimize_two_stage_hyperparameters(
    feature_df=labeled_feature_df,
    first_positive_label='vasc',
    second_labels=('mel', 'nv'),
    first_feature_names=first_stage_feature_names,
    second_feature_names=second_stage_feature_names,
    first_model_type='random_forest',
    second_model_type='random_forest',
    first_param_grid=FIRST_PARAM_GRID,
    second_param_grid=SECOND_PARAM_GRID,
    cv=5,
    scoring='accuracy',
)
```

输出结构：

```python
two_stage_optimization = {
    "first_optimization": dict,
    "second_optimization": dict,
    "first_best_params": dict,
    "second_best_params": dict,
    "two_stage_result": dict,
    "evaluation": dict,
}
```

如果使用 XGBoost GPU，建议：

```python
FIRST_DEVICE = 'cuda'
SECOND_DEVICE = 'cuda'
FIRST_N_JOBS = 1
SECOND_N_JOBS = 1
```

如果 GPU 显存仍然很空，可以小步尝试把对应的 `*_N_JOBS` 调到 `2`、`3` 或 `4`。

### 5.4 基础两阶段实验入口

基础实验 cell 通过配置变量控制整个流程：

```python
FIRST_STAGE_LABEL = 'vasc'
SECOND_STAGE_LABELS = ('mel', 'nv')
FIRST_FEATURE_MODE = 'selected'
SECOND_FEATURE_MODE = 'all'
FIRST_MODEL_TYPE = 'random_forest'
SECOND_MODEL_TYPE = 'random_forest'
```

特征模式支持：

| 模式 | 含义 |
|---|---|
| `all` | 使用全部特征 |
| `selected` | 使用第 4 节筛选后的 `selected_feature_names` |
| `category` | 使用指定特征类别，例如 `color`、`shape`、`asymmetry`、`glcm` |
| `custom` | 使用手动列出的特征名 |

常见调试方法：

1. 先固定两个阶段都用 `random_forest`，确认流程可运行
2. 检查两个阶段的 group overlap 是否都为 `0`
3. 改 `FIRST_STAGE_LABEL` 测试先分出 `mel`、`nv` 或 `vasc` 哪个更好
4. 分别调整 `FIRST_FEATURE_MODE` 和 `SECOND_FEATURE_MODE`，比较全特征与筛选特征
5. 只改一个阶段模型，观察第一阶段、第二阶段和最终三分类各自变化
6. 最后再打开联合调参，避免一开始就运行大网格

### 5.5 两阶段联合调参实验入口

联合调参 cell 默认：

```python
RUN_TWO_STAGE_OPTIMIZATION = False
```

需要调参时改为：

```python
RUN_TWO_STAGE_OPTIMIZATION = True
```

默认网格较小，适合先验证流程。确认无误后，再扩大 `FIRST_PARAM_GRID` 和 `SECOND_PARAM_GRID`。

## 6. 综合两步分类策略

第 5 节已经在 `evaluate_two_stage_models` 中完成两个二分类模型的串联预测，因此第 6 节可以用于汇总和比较多组两阶段实验结果。当前综合评估主要依赖：

```python
evaluate_two_stage_models
print_two_stage_evaluation_results
```

可以分别比较：

- 不同第一阶段标签：先分出 `vasc`、`mel` 或 `nv`
- 不同模型组合：SVM + 随机森林、随机森林 + XGBoost 等
- 不同特征组合：第一阶段用筛选特征，第二阶段用全特征或自定义特征
- 不同调参策略：手动参数和 `optimize_two_stage_hyperparameters` 的自动调参结果

最终输出均包含：

- 第一阶段二分类 hit rate、混淆矩阵和 classification report
- 第二阶段二分类 hit rate、混淆矩阵和 classification report
- 两阶段串联后的三分类 hit rate、混淆矩阵和 classification report
- `y_true`
- `y_pred`

## 主要中间数据总览

| 变量 | 类型 | 含义 |
|---|---|---|
| `feature_df` | `pd.DataFrame` | 全部样本的特征表，含 `filename`、`image_id`、`base_id` |
| `labeled_feature_df` | `pd.DataFrame` | 合并 `label.csv` 后的特征表 |
| `all_features` | `np.ndarray` | 全部提取特征矩阵 |
| `all_feature_names` | `list[str]` | 全部特征名 |
| `selected_feature_names` | `list[str]` | 第 4 节筛选后的特征名 |
| `first_stage_feature_names` | `list[str]` | 第一阶段使用的特征名 |
| `second_stage_feature_names` | `list[str]` | 第二阶段使用的特征名 |
| `two_stage_result` | `dict` | 两阶段划分、模型、参数和特征配置 |
| `two_stage_evaluation` | `dict` | 第一阶段、第二阶段和综合三分类评估结果 |
| `two_stage_optimization` | `dict` | 两阶段联合调参结果，仅在打开调参 cell 后生成 |
| `*_feature_importances` | `list[tuple[str, float]]` | 树模型或线性 SVM 特征重要性 |

## 复现实验时的运行顺序

建议按 notebook 顺序执行：

1. 运行导入和预处理函数
2. 运行特征提取，生成 `feature_df`
3. 运行第 3 节三分类模型
4. 运行第 4 节特征筛选，生成 `selected_feature_names`
5. 运行第 5 节两阶段分类
6. 根据需要在第 6 节汇总结果

如果只想复用已经提取好的特征，可以从生成 `feature_df` 之后继续运行模型和筛选部分。
