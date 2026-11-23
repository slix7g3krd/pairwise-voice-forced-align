# pairwise-voice-forced-align

## 领域

游戏、图形与多媒体

## 主题

配音强制对齐

## 说明

静音阈值、自适应窗长与 DP 回溯。

初始状态下 `media/voice_forced_align.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
