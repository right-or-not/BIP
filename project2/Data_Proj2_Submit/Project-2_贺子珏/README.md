# BIP Project 2 皮肤病图像分类提交说明

本文件夹是 Project 2 的最终提交版本。项目目标是对皮肤病灶图像进行三分类预测：

```text
mel
nv
vasc
```

最终模型采用双阶段分类结构：

1. 第一阶段：判断样本是否为 `vasc`
2. 第二阶段：对非 `vasc` 样本继续判断 `mel` 或 `nv`

预测结果会保存为 CSV 文件。

## 1. 文件内容

提交文件夹结构如下：

```text
submit/
├── main.py                 # 可直接运行的 Python 预测脚本
├── submit.ipynb            # Notebook 版本预测流程
├── model.joblib            # 已训练并保存好的双阶段模型
├── pyproject.toml          # 项目依赖
├── README.md               # 项目说明
├── image/                  # 输入图像
├── mask/                   # 病灶 mask
└── output.csv  			# 默认预测输出文件
```

当前保存的模型信息：

- 第一阶段模型：SVM
- 第二阶段模型：SVM
- 第一阶段使用特征数：16
- 第二阶段使用特征数：45
- 模型文件：`model.joblib` 

模型文件中已经保存了每一阶段所需的特征名，因此运行预测时会自动按训练时的特征顺序对齐。

## 2. 准备工作

建议使用 Python `>=3.13`。

依赖写在 `pyproject.toml` 中。进入 `submit/` 文件夹后，可以使用以下方式安装：

```bash
python -m pip install .
```

如果使用 `uv`，也可以运行：

```bash
uv sync
```

运行前需要确认以下文件和文件夹存在：

```text
model.joblib
image/
mask/
```

mask 文件建议使用如下命名方式：

```text
mask_<图像文件名>
```

例如：

```text
image/1.jpg
mask/mask_1.jpg
```

程序也兼容以下 mask 命名形式：

```text
mask_<image_name>
<image_name>
<image_stem>.png
mask_<image_stem>.png
<image_stem>.jpg
mask_<image_stem>.jpg
```

## 3. 运行方法一：使用 submit.ipynb

打开并从上到下运行：

```text
submit.ipynb
```

Notebook 中包含：

1. 图像读取与预处理函数
2. 特征提取函数
3. 模型特征对齐函数
4. 模型导入与预测函数
5. 调用接口

最后一个调用 cell 中可以修改路径参数，例如：

```python
PROJECT_DIR = Path('./')
MODEL_PATH = PROJECT_DIR / 'model.joblib'
IMAGE_DIR = PROJECT_DIR / 'image_processed'
MASK_DIR = PROJECT_DIR / 'mask'
OUTPUT_CSV = PROJECT_DIR / 'output.csv'
RUN_PREPROCESSING = True
RAW_IMAGE_DIR = PROJECT_DIR / 'image'
```

运行后会生成预测 CSV。

## 4. 运行方法二：使用 main.py

`main.py` 是可直接运行的 Python 脚本。进入 `submit/` 文件夹后运行：

```bash
python main.py
```

脚本中的参数集中写在 `main()` 函数中：

```python
PROJECT_DIR = Path("./")
MODEL_PATH = PROJECT_DIR / "model.joblib"
RAW_IMAGE_DIR = PROJECT_DIR / "image"
IMAGE_DIR = PROJECT_DIR / "image_processed"
MASK_DIR = PROJECT_DIR / "mask"
PREPROCESSED_OUTPUT_DIR = PROJECT_DIR / "image_processed"
OUTPUT_CSV = PROJECT_DIR / "output.csv"
MAX_IMAGES = None
RUN_PREPROCESSING = True
```

如果只想快速测试几张图片，可以把：

```python
MAX_IMAGES = None
```

改成：

```python
MAX_IMAGES = 5
```

如果已经有处理好的图像，不想重新预处理，可以改成：

```python
RUN_PREPROCESSING = False
IMAGE_DIR = PROJECT_DIR / "image_processed"
```

## 5. 参数含义

`PROJECT_DIR`

项目根目录。默认是当前 `submit/` 文件夹。

`MODEL_PATH`

保存好的模型文件路径。默认是：

```text
model.joblib
```

`RAW_IMAGE_DIR`

原始图像文件夹。默认是：

```text
image/
```

`IMAGE_DIR`

已经预处理好的图像文件夹。如果 `RUN_PREPROCESSING=False`，程序会从这里读取图像。

`MASK_DIR`

mask 文件夹。默认是：

```text
mask/
```

`PREPROCESSED_OUTPUT_DIR`

当 `RUN_PREPROCESSING=True` 时，预处理后的图像会保存到该文件夹。

`OUTPUT_CSV`

预测结果 CSV 的输出路径。默认是：

```text
prediction_results.csv
```

`MAX_IMAGES`

最多处理多少张图像。  
如果为 `None`，则处理所有图像。

`RUN_PREPROCESSING`

是否先执行预处理。

- `True`：从 `RAW_IMAGE_DIR` 读取原图，预处理后再提取特征
- `False`：直接从 `IMAGE_DIR` 读取图像

## 6. 输出结果

输出 CSV 包含以下列：

```text
image_id
dx
```

其中最终分类结果在：

```text
dx
```

可能的预测结果为：

```text
mel
nv
vasc
```
