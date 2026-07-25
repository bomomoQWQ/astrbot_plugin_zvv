# 张维为表情包

> "对，你要自信。" —— 张维为

让 AI 学会用张维为语录怼人。输入"美国不行了"，输出"美国已经竞争不过中国了.png"。

## ✨ 怎么玩

### AI 自动发

聊天中 AI 判断氛围到了，自己搜图自己发，不用你管：

```
用户: 我觉得日本新首相这人不行
AI: 调用 search_zvv("什么都敢说") → 候选 → send_zvv(1)
QQ: [动画表情] 他什么都敢说.png
```

### 手动发

```
/zvv 无语
/zvv 美国不行了
/zvv 对方太自信了
```

## 🧠 技术

- 嵌入模型：[bge-base-zh-v1.5 Q8](https://huggingface.co/moyangzhan/bge-base-zh-v1.5-onnx)（量化到 98MB）
- 推理引擎：ONNX Runtime，纯 CPU
- 416 张张维为语录，语义搜索跨关键词匹配
- 双核 2GB 服务器就能跑

> ⚠️ 首次启动需要 `pip install onnxruntime tokenizers numpy`，然后自动下载模型和图库（约 100MB），并预计算 416 张图向量（约 2-3 分钟）。之后秒启。

## 🙏 致谢

- 模型：[moyangzhan/bge-base-zh-v1.5-onnx](https://huggingface.co/moyangzhan/bge-base-zh-v1.5-onnx)
- 表情包：[MemeMeow](https://github.com/MemeMeow-Studio/MemeMeow)
