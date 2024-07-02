# HunyuanCaptioner: Local OpenAI API Interface

## 简介
HunyuanCaptioner 是一个用于生成图像描述的本地 OpenAI API 接口，依托于 HunyuanDiT 项目。通过简单的 API 调用，您可以为图像生成高质量的文本描述。

## 目录
- [安装](#安装)
  - [克隆 HunyuanDiT 最新仓库](#克隆-hunyuandit-最新仓库)
  - [下载打标模型](#下载打标模型)
  - [复制脚本](#复制脚本)
  - [修改脚本](#修改脚本)
  - [运行 API 服务](#运行-api-服务)
- [用法](#用法)

## 安装

### 克隆 HunyuanDiT 最新仓库
首先，克隆 HunyuanDiT 最新仓库：
```bash
git clone https://github.com/Tencent/HunyuanDiT
```

### 下载打标模型
使用 `huggingface-cli` 工具下载打标模型到本地目录：
```bash
huggingface-cli download Tencent-Hunyuan/HunyuanCaptioner --local-dir ./ckpts/captioner
```

### 复制脚本
手动复制 `caption_api_demo.py` 到 HunyuanDiT 的 `mllm` 目录：
```bash
cp caption_api_demo.py ./HunyuanDiT/mllm
```

### 修改脚本
如有需要，修改 `caption_api_demo.py` 文件第 141 行和第 151 行的本地打标模型目录路径。
```bash
"HunyuanCaptioner": "./ckpts/captioner" #修改为你的本地打标模型目录路径
```

### 运行 API 服务
启动 API 服务：
```bash
python mllm/caption_api_demo.py
```
### 用法
```bash
#当看见以下信息，代表本地api服务启动成功
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:5000 (Press CTRL+C to quit)
```
现在可以通过GPT-for-Annotation插件连接api进行打标了
