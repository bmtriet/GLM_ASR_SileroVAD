# GLM-ASR

[Readme in English](README.md)

<div align="center">
<img src=resources/logo.svg width="20%"/>
</div>
<p align="center">
    👋 加入我们的 <a href="resources/WECHAT.md" target="_blank">微信</a> 社区
</p>

## 模型介绍

**GLM-ASR-Nano-2512** 是一款鲁棒的开源语音识别模型，参数量为 **1.5B**。
该模型专为应对真实世界的复杂场景而设计，在多项基准测试中超越 OpenAI Whisper V3，同时保持紧凑的模型规模。

核心能力包括：

* **卓越的方言支持**
  除标准普通话和英语外，模型针对**粤语**及其他方言进行了深度优化，有效填补了方言语音识别领域的空白。

* **低音量语音鲁棒性**
  专门针对**"低语/轻声"**场景进行训练，能够捕捉并准确转录传统模型难以识别的极低音量音频。

* **SOTA 性能**
  在同类开源模型中实现**最低平均错误率 (4.10)**，在中文基准测试（Wenet Meeting、Aishell-1 等）中展现出显著优势。

## 基准测试

我们将 GLM-ASR-Nano 与主流开源和闭源模型进行了对比评测。结果表明，**GLM-ASR-Nano (1.5B)** 表现优异，尤其在复杂声学环境下优势明显。

![bench](resources/bench.png)

说明：

- Wenet Meeting 反映了包含噪声和语音重叠的真实会议场景。
- Aishell-1 是标准普通话基准测试集。

## 推理

`GLM-ASR-Nano-2512` 可通过 `transformers` 库轻松集成。  
我们将支持 `transformers 5.x` 以及 `vLLM`、`SGLang` 等推理框架。

### 环境依赖

```bash
pip install -r requirements.txt
sudo apt install ffmpeg
```

### 示例代码

```shell
python inference.py --checkpoint_dir zai-org/GLM-ASR-Nano-2512 --audio examples/example_en.wav # 英文
python inference.py --checkpoint_dir zai-org/GLM-ASR-Nano-2512 --audio examples/example_zh.wav # 中文
```

对于上述两段示例音频，模型能够生成准确的转录结果：

```shell
be careful not to allow fabric to become too hot which can cause shrinkage or in extreme cases scorch
我还能再搞一个，就算是非常小的声音也能识别准确
```