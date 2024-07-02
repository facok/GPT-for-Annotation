
# Florence-2-api: Local OpenAI API Interface

## 简介
Florence-2-api 是一个用于生成图像描述的本地 OpenAI API 接口，依托于 `Florence-2` 项目。通过简单的 API 调用，您可以为图像生成高质量的文本描述。Florence-2 提供多个型号和微调版本，可根据个人需求下载使用。

[Florence-2-SD3-Captioner](https://huggingface.co/gokaygokay/Florence-2-SD3-Captioner)

[Florence-2-large](https://huggingface.co/microsoft/Florence-2-large)

[Florence-2-large-ft](https://huggingface.co/microsoft/Florence-2-large-ft)

[Florence-2-base](https://huggingface.co/microsoft/Florence-2-base)

[Florence-2-base-ft](https://huggingface.co/microsoft/Florence-2-base-ft)


## 目录
- [安装](#安装)
  - [安装依赖环境](#安装依赖环境)
  - [下载打标模型](#下载打标模型)
  - [复制脚本](#复制脚本)
  - [修改脚本](#修改脚本)
  - [运行 API 服务](#运行-api-服务)
- [用法](#用法)

## 安装

### 安装依赖环境
Tips: Florence2的环境需求和混元DiT基本重合，如果你已有混元DiT环境，可以跳过依赖安装直接使用。
```bash
pip install -r requirements_florence2.txt
```


### 下载打标模型
使用 `huggingface-cli` 工具下载打标模型到本地目录：
```bash
huggingface-cli download gokaygokay/Florence-2-SD3-Captioner --local-dir ./Florence-2-SD3-Captioner
```

### 复制脚本
手动复制 `Florence-2_api.py` 到本地目录：
```bash
cp Florence-2_api.py ./
```

### 修改脚本
如有需要，修改 `Florence-2_api.py` 中本地打标模型的目录路径：
```python
# 修改第 55 行，设置默认模型目录路径
default_model_dir = "/mnt/e/public_model/Florence-2-SD3-Captioner" # 修改为你的本地打标模型目录路径。

# 修改第 63 行到第 69 行，设置模型目录映射
model_dir_mapping = {
    "Florence-2-large-ft": "/mnt/e/public_model/Florence-2-large-ft", # 修改为你的本地打标模型目录路径
    "Florence-2-large": "/mnt/e/public_model/Florence-2-large",
    "Florence-2-base-ft": "/mnt/e/public_model/Florence-2-base-ft",
    "Florence-2-base": "/mnt/e/public_model/Florence-2-base",
    "Florence-2-SD3-Captioner": "/mnt/e/public_model/Florence-2-SD3-Captioner"
}
```

### 运行 API 服务
启动 API 服务：
```bash
python Florence-2_api.py
```

## 用法
当看到以下信息时，代表本地 API 服务启动成功：
```plaintext
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:5000 (Press CTRL+C to quit)
```
现在可以通过 GPT-for-Annotation 插件连接 API 进行打标了。


