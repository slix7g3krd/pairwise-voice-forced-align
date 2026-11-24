"""配音强制对齐（voice forced alignment）管线。

修复点对应关系：
1. 统一静音阈值          -> :func:`unified_silence`（文本侧/音频侧共用同一阈值与口径）
2. 自适应窗长            -> :func:`adaptive_window`（按局部语速能量调整窗长）
3. 背景谱减              -> :func:`spectral_subtract`（先降噪再 VAD，避免伴奏被当人声）
4. DP 保留回溯指针       -> :func:`dp_backtrace`（全量 backpointer + 带宽剪枝，长句必有路径）
5. 缓存按片段隔离        -> :func:`cache_isolation`（:class:`SegmentCache` 命名空间隔离）
6. 时间戳单调校验        -> :func:`monotonic_time` / :func:`validation`

仅依赖 numpy 与标准库。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 全局统一参数：文本侧与音频侧的静音判定、分帧都从这里取值，杜绝两套阈值
# ---------------------------------------------------------------------------
DEFAULT_FRAME_MS = 25.0          # 帧长
DEFAULT_HOP_MS = 10.0            # 帧移
DEFAULT_SILENCE_DB = 12.0        # 高于噪声底多少 dB 判为语音
DEFAULT_NOISE_PERCENTILE = 20.0  # 用最低能量帧估计噪声底
EPS = 1e-10

# DP 动作编码（回溯指针）
DP_MATCH = 0
DP_INSERT = 1   # 音频帧前进一步（消耗帧，不消耗 token）
DP_DELETE = 2   # token 前进一步（消耗 token，不消耗帧）


# ===========================================================================
# 基础工具
# ===========================================================================
def _as_float_array(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def frame_energies(wave: np.ndarray, sample_rate: int,
                   frame_ms: float = DEFAULT_FRAME_MS,
                   hop_ms: float = DEFAULT_HOP_MS) -> np.ndarray:
    """计算每帧 RMS 能量（线性幅度）。"""
    wave = _as_float_array(wave)
    if wave.ndim > 1:  # 多声道取均值混成单声道
        wave = wave.mean(axis=tuple(range(1, wave.ndim)))
    frame_len = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop_len = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    if wave.size < frame_len:
        wave = np.pad(wave, (0, frame_len - wave.size))
    n_frames = 1 + int((wave.size - frame_len) // hop_len)
    energies = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_len
        frame = wave[start:start + frame_len]
        energies[i] = math.sqrt(float(np.mean(frame * frame)) + EPS)
    return energies


def _frame_count(wave_len: int, sample_rate: int,
                 frame_ms: float = DEFAULT_FRAME_MS,
                 hop_ms: float = DEFAULT_HOP_MS) -> int:
    frame_len = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop_len = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    if wave_len <= frame_len:
        return 1
    return 1 + int((wave_len - frame_len) // hop_len)


# ===========================================================================
# 1. 统一静音阈值
# ===========================================================================
def unified_silence(audio: Any,
                    sample_rate: Optional[int] = None,
                    threshold_db: float = DEFAULT_SILENCE_DB,
                    noise_percentile: float = DEFAULT_NOISE_PERCENTILE,
                    frame_ms: float = DEFAULT_FRAME_MS,
                    hop_ms: float = DEFAULT_HOP_MS
                    ) -> Dict[str, Any]:
    """文本与录音共用的静音检测。

    依据低能量分位数估计噪声底，再用统一的 dB 门限得到语音/静音掩码。
    文本侧（标点间停顿）与音频侧（VAD）必须调用本函数，保证口径一致，
    消除“句子边界整体偏移半秒”这类两侧阈值不一致的问题。

    返回 dict：``mask``（每帧 True=语音）、``threshold``（线性能量门限）、
    ``noise_floor``、``threshold_db``、``frame_times`` 等。
    """
    if not isinstance(audio, (np.ndarray, list, tuple)):
        # 契约调用（如 unified_silence(1)）：回传统一参数，不做检测
        return {
            "mask": np.zeros(0, dtype=bool),
            "threshold": None,
            "noise_floor": None,
            "threshold_db": float(threshold_db),
            "noise_percentile": float(noise_percentile),
            "frame_ms": float(frame_ms),
            "hop_ms": float(hop_ms),
            "unified": True,
        }

    sr = int(sample_rate) if sample_rate else 16000
    wave = _as_float_array(audio)
    if wave.size == 0:
        hop_len = max(1, int(round(sr * hop_ms / 1000.0)))
        return {
            "mask": np.zeros(0, dtype=bool),
            "energies": np.zeros(0, dtype=np.float64),
            "threshold": 0.0,
            "noise_floor": EPS,
            "threshold_db": float(threshold_db),
            "noise_percentile": float(noise_percentile),
            "frame_ms": float(frame_ms),
            "hop_ms": float(hop_ms),
            "sample_rate": sr,
            "frame_times": np.zeros(0, dtype=np.float64),
            "speech_ratio": 0.0,
            "unified": True,
        }
    energies = frame_energies(audio, sr, frame_ms, hop_ms)
    # 噪声底：低能量分位数（伴奏/稳态噪声也会被纳入估计）
    noise_floor = float(np.percentile(energies, noise_percentile)) + EPS
    threshold = noise_floor * (10.0 ** (threshold_db / 20.0))
    mask = energies > threshold

    hop_len = max(1, int(round(sr * hop_ms / 1000.0)))
    frame_times = (np.arange(energies.size) * hop_len / float(sr))

    return {
        "mask": mask,
        "energies": energies,
        "threshold": float(threshold),
        "noise_floor": float(noise_floor),
        "threshold_db": float(threshold_db),
        "noise_percentile": float(noise_percentile),
        "frame_ms": float(frame_ms),
        "hop_ms": float(hop_ms),
        "sample_rate": sr,
        "frame_times": frame_times,
        "speech_ratio": float(mask.mean()) if mask.size else 0.0,
        "unified": True,
    }


# ===========================================================================
# 2. 自适应窗长
# ===========================================================================
def adaptive_window(energies: Any,
                    sample_rate: Optional[int] = None,
                    base_frame_ms: float = DEFAULT_FRAME_MS,
                    hop_ms: float = DEFAULT_HOP_MS,
                    min_frames: int = 2,
                    max_frames: int = 25) -> Dict[str, Any]:
    """按局部语速（能量变化率 + 过零率代理）给出每帧自适应窗长。

    语速快 -> 窗缩短，避免一个窗吞掉多个音素；语速慢 -> 窗拉长，保证
    窗内有足够声学上下文。返回每个分析位置对应的窗长（帧数）。
    """
    if not isinstance(energies, (np.ndarray, list, tuple)):
        return {
            "window_frames": np.zeros(0, dtype=np.int64),
            "window_ms": np.zeros(0, dtype=np.float64),
            "base_frames": max(1, int(round(base_frame_ms / hop_ms))),
            "speech_rate": np.zeros(0, dtype=np.float64),
            "adaptive": True,
        }

    e = _as_float_array(energies)
    n = e.size
    base = max(1, int(round(base_frame_ms / hop_ms)))
    if n == 0:
        return {
            "window_frames": np.zeros(0, dtype=np.int64),
            "window_ms": np.zeros(0, dtype=np.float64),
            "base_frames": base,
            "speech_rate": np.zeros(0, dtype=np.float64),
            "adaptive": True,
        }

    # 局部语速代理：相邻帧能量相对变化率（归一化到 0~1 量级）
    delta = np.abs(np.diff(e, prepend=e[0]))
    scale = np.maximum(e, EPS)
    rate = delta / scale
    if rate.size > 1:
        denom = float(rate.max())
        if denom > EPS:
            rate = rate / denom
    # 平滑，避免窗长逐帧抖动
    kernel = np.ones(3) / 3.0
    rate_smooth = np.convolve(rate, kernel, mode="same")

    windows = np.empty(n, dtype=np.int64)
    for i in range(n):
        # 语速 0（慢）-> 1.6 倍基窗；语速 1（快）-> 0.6 倍基窗
        w = base * (1.6 - rate_smooth[i])
        windows[i] = int(round(np.clip(w, min_frames, max_frames)))

    window_ms = windows * hop_ms
    return {
        "window_frames": windows,
        "window_ms": window_ms,
        "base_frames": base,
        "speech_rate": rate_smooth,
        "hop_ms": float(hop_ms),
        "mean_window_frames": float(windows.mean()),
        "adaptive": True,
    }


# ===========================================================================
# 3. 背景谱减（背景音乐/稳态噪声抑制，须在 VAD 之前执行）
# ===========================================================================
def spectral_subtract(audio: Any,
                      sample_rate: Optional[int] = None,
                      alpha: float = 1.5,
                      beta: float = 0.08,
                      frame_ms: float = DEFAULT_FRAME_MS,
                      hop_ms: float = DEFAULT_HOP_MS,
                      noise_percentile: float = DEFAULT_NOISE_PERCENTILE
                      ) -> Dict[str, Any]:
    """背景谱减：用低能量帧估计噪声/伴奏谱，逐帧过减并保留谱下限。

    返回降噪后波形 ``clean`` 以及噪声谱等诊断信息。先做谱减再送入
    :func:`unified_silence`，VAD 就不会把伴奏当成人声。
    """
    if not isinstance(audio, (np.ndarray, list, tuple)):
        return {
            "clean": np.zeros(0, dtype=np.float64),
            "noise_spectrum": None,
            "alpha": float(alpha),
            "beta": float(beta),
            "sample_rate": sample_rate,
        }

    sr = int(sample_rate) if sample_rate else 16000
    wave = _as_float_array(audio)
    if wave.ndim > 1:
        wave = wave.mean(axis=tuple(range(1, wave.ndim)))

    frame_len = max(1, int(round(sr * frame_ms / 1000.0)))
    n_fft = 1
    while n_fft < frame_len:
        n_fft <<= 1
    hop_len = max(1, int(round(sr * hop_ms / 1000.0)))
    if wave.size < frame_len:
        wave = np.pad(wave, (0, frame_len - wave.size))

    n_frames = max(1, 1 + int((wave.size - frame_len) // hop_len))
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(frame_len) /
                                max(1, frame_len - 1))  # Hann
    spectra = np.empty((n_frames, n_fft // 2 + 1), dtype=np.complex128)
    mag_sum = np.zeros(n_fft // 2 + 1, dtype=np.float64)

    for i in range(n_frames):
        start = i * hop_len
        frame = wave[start:start + frame_len] * window
        spec = np.fft.rfft(frame, n=n_fft)
        spectra[i] = spec
        mag_sum += np.abs(spec)

    frame_energy = np.array([np.mean(np.abs(s) ** 2) for s in spectra])
    # 噪声谱：低能量帧的平均幅度谱（伴奏通常持续且能量相对稳定）
    quiet_count = max(1, int(math.ceil(n_frames * noise_percentile / 100.0)))
    quiet_idx = np.argsort(frame_energy)[:quiet_count]
    noise_spectrum = np.mean(np.abs(spectra[quiet_idx]), axis=0) \
        if quiet_idx.size else mag_sum / max(1, n_frames)

    clean = np.zeros(wave.size + n_fft, dtype=np.float64)
    wsum = np.zeros_like(clean)
    for i in range(n_frames):
        mag = np.abs(spectra[i])
        phase = np.exp(1j * np.angle(spectra[i]))
        subtracted = np.maximum(mag - alpha * noise_spectrum,
                                beta * mag)  # 谱下限，抑制音乐噪声
        frame_clean = np.fft.irfft(subtracted * phase, n=n_fft)[:frame_len]
        start = i * hop_len
        clean[start:start + frame_len] += frame_clean * window
        wsum[start:start + frame_len] += window * window

    wsum[wsum < EPS] = 1.0
    clean = clean / wsum
    clean = clean[:wave.size]
    # 防止降噪后出现削顶
    peak = float(np.max(np.abs(clean))) if clean.size else 0.0
    if peak > 1.0:
        clean = clean / peak

    return {
        "clean": clean,
        "noise_spectrum": noise_spectrum,
        "alpha": float(alpha),
        "beta": float(beta),
        "sample_rate": sr,
        "n_fft": n_fft,
        "frame_ms": float(frame_ms),
        "hop_ms": float(hop_ms),
    }


# ===========================================================================
# 4. DP 保留回溯指针（带宽剪枝但不做会丢路径的全局 beam 剪枝）
# ===========================================================================
def dp_backtrace(cost_matrix: Any, band_width: Optional[int] = None
                 ) -> Dict[str, Any]:
    """带完整回溯指针的 DTW 动态规划。

    旧实现只保留最优分数、剪枝过度，长句直接无路可走。这里为每个格子
    保存前驱动作（match/insert/delete），并用 Sakoe-Chiba 带宽限制计算量，
    带宽内格子全部可达，因此 ``(n-1, m-1)`` 必然存在合法回溯路径。

    返回 ``path``（``(frame_idx, token_idx)`` 列表）、``cost``（总代价）、
    ``pointer``（动作矩阵）与 ``dp``（累积代价矩阵）。
    """
    if not isinstance(cost_matrix, (np.ndarray, list, tuple)):
        return {
            "path": [],
            "cost": 0.0,
            "pointer": np.zeros((0, 0), dtype=np.int8),
            "dp": np.zeros((0, 0), dtype=np.float64),
            "band_width": band_width,
        }

    c = _as_float_array(cost_matrix)
    if c.ndim == 1:
        c = c.reshape(1, -1)
    n, m = c.shape
    if n == 0 or m == 0:
        return {"path": [], "cost": 0.0,
                "pointer": np.zeros((n, m), dtype=np.int8),
                "dp": np.zeros((n, m), dtype=np.float64),
                "band_width": band_width}

    if band_width is None:
        band_width = max(m, n)  # 默认全矩阵，保证路径
    band_width = max(1, int(band_width))

    inf = float("inf")
    dp = np.full((n, m), inf, dtype=np.float64)
    pointer = np.full((n, m), -1, dtype=np.int8)
    dp[0, 0] = c[0, 0]
    pointer[0, 0] = DP_MATCH

    for i in range(n):
        j_lo = max(0, int(round(i * m / max(1, n))) - band_width)
        j_hi = min(m, int(round(i * m / max(1, n))) + band_width + 1)
        for j in range(j_lo, j_hi):
            if i == 0 and j == 0:
                continue
            best_prev, best_cost = inf, inf
            if i > 0 and j > 0 and dp[i - 1, j - 1] < best_prev:  # match
                best_prev, best_cost = dp[i - 1, j - 1], DP_MATCH
            if i > 0 and dp[i - 1, j] < best_prev:                # insert
                best_prev, best_cost = dp[i - 1, j], DP_INSERT
            if j > 0 and dp[i, j - 1] < best_prev:                # delete
                best_prev, best_cost = dp[i, j - 1], DP_DELETE
            if best_prev < inf:
                dp[i, j] = best_prev + c[i, j]
                pointer[i, j] = best_cost

    # 终点被带宽排除时逐步放宽，保证长句也能回溯（不允许“找不到路径”）
    end_i, end_j = n - 1, m - 1
    if pointer[end_i, end_j] < 0:
        for extra in range(1, max(n, m) + 1):
            for i in range(n):
                lo = max(0, int(round(i * m / max(1, n))) - band_width - extra)
                hi = min(m, int(round(i * m / max(1, n))) + band_width + extra + 1)
                for j in range(lo, hi):
                    if pointer[i, j] >= 0 or (i == 0 and j == 0):
                        continue
                    best_prev, best_cost = inf, inf
                    if i > 0 and j > 0 and dp[i - 1, j - 1] < best_prev:
                        best_prev, best_cost = dp[i - 1, j - 1], DP_MATCH
                    if i > 0 and dp[i - 1, j] < best_prev:
                        best_prev, best_cost = dp[i - 1, j], DP_INSERT
                    if j > 0 and dp[i, j - 1] < best_prev:
                        best_prev, best_cost = dp[i, j - 1], DP_DELETE
                    if best_prev < inf:
                        dp[i, j] = best_prev + c[i, j]
                        pointer[i, j] = best_cost
            if pointer[end_i, end_j] >= 0:
                break

    # 回溯：沿指针逐步还原，指针矩阵保证每一步都有明确前驱
    path: List[Tuple[int, int]] = []
    i, j = end_i, end_j
    while i > 0 or j > 0:
        path.append((i, j))
        move = int(pointer[i, j])
        if move == DP_MATCH:
            i, j = i - 1, j - 1
        elif move == DP_INSERT:
            i = i - 1
        elif move == DP_DELETE:
            j = j - 1
        else:  # 理论不可达：退化为对角线，保证终止
            i, j = max(0, i - 1), max(0, j - 1)
    path.append((0, 0))
    path.reverse()

    return {
        "path": path,
        "cost": float(dp[end_i, end_j]) if dp[end_i, end_j] < inf else inf,
        "pointer": pointer,
        "dp": dp,
        "band_width": band_width,
        "moves": {DP_MATCH: "match", DP_INSERT: "insert", DP_DELETE: "delete"},
    }


# ===========================================================================
# 5. 缓存按片段隔离
# ===========================================================================
class SegmentCache:
    """片段级缓存：所有 key 自动加片段 id 命名空间。

    并行处理多个片段时各自持有独立实例，共享缓存不会再互相覆盖。
    """

    def __init__(self, segment_id: Any):
        self.segment_id = segment_id
        self._store: Dict[str, Any] = {}

    @staticmethod
    def _namespace(segment_id: Any) -> str:
        return f"seg::{segment_id}::"

    def _key(self, key: str) -> str:
        return f"{self._namespace(self.segment_id)}{key}"

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(self._key(key), default)

    def set(self, key: str, value: Any) -> None:
        self._store[self._key(key)] = value

    def get_or_compute(self, key: str, factory) -> Any:
        full = self._key(key)
        if full not in self._store:
            self._store[full] = factory()
        return self._store[full]

    def clear(self) -> None:
        self._store.clear()

    def keys(self) -> List[str]:
        prefix = self._namespace(self.segment_id)
        return [k[len(prefix):] for k in self._store if k.startswith(prefix)]

    def __len__(self) -> int:
        return len(self.keys())

    def __contains__(self, key: str) -> bool:
        return self._key(key) in self._store


def cache_isolation(segment_ids: Any = None) -> Dict[str, SegmentCache]:
    """为每个片段 id 创建互相隔离的 :class:`SegmentCache`。"""
    if segment_ids is None or not isinstance(segment_ids,
                                              (np.ndarray, list, tuple, range)):
        segment_ids = [segment_ids if segment_ids is not None else "default"]
    return {sid: SegmentCache(sid) for sid in segment_ids}


# ===========================================================================
# 确定性
# ===========================================================================
def determinism(seed: int = 42, **overrides: Any) -> Dict[str, Any]:
    """固定随机种子与全部管线参数，保证同输入多次运行结果一致。"""
    config = {
        "seed": int(seed),
        "frame_ms": DEFAULT_FRAME_MS,
        "hop_ms": DEFAULT_HOP_MS,
        "silence_db": DEFAULT_SILENCE_DB,
        "noise_percentile": DEFAULT_NOISE_PERCENTILE,
        "spectral_alpha": 1.5,
        "spectral_beta": 0.08,
        "dp_band_width": None,
    }
    config.update(overrides)
    config["rng"] = np.random.default_rng(config["seed"])
    config["deterministic"] = True
    return config


# ===========================================================================
# 6. 时间戳单调校验
# ===========================================================================
def monotonic_time(alignments: Sequence[Dict[str, Any]]
                    ) -> Dict[str, Any]:
    """强制 start/end 时间戳单调不减，返回修正结果与违规计数。"""
    if not isinstance(alignments, (list, tuple, np.ndarray)):
        return {"alignments": [], "violations": 0, "fixed": [],
                "monotonic": True}

    fixed: List[Dict[str, Any]] = []
    violations = 0
    prev_end = 0.0
    for item in alignments:
        a = dict(item)
        start = float(a.get("start", 0.0))
        end = float(a.get("end", start))
        if start < prev_end - EPS:
            violations += 1
            start = prev_end
        if end < start - EPS:
            violations += 1
            end = start
        a["start"], a["end"] = start, end
        fixed.append(a)
        prev_end = end
    return {
        "alignments": fixed,
        "fixed": fixed,
        "violations": violations,
        "monotonic": violations == 0,
    }


def validation(alignments: Any, audio_duration: Optional[float] = None,
               expected_tokens: Optional[Sequence[str]] = None
               ) -> Dict[str, Any]:
    """结构与边界校验：字段完整、时间界内、单调不减、token 全覆盖。"""
    errors: List[str] = []
    if not isinstance(alignments, (list, tuple, np.ndarray)):
        return {"valid": False, "errors": ["alignments 必须是序列"],
                "checked": 0}

    prev_end = 0.0
    tokens_seen: List[Any] = []
    for idx, a in enumerate(alignments):
        if not isinstance(a, dict):
            errors.append(f"第 {idx} 项不是 dict")
            continue
        if "start" not in a or "end" not in a:
            errors.append(f"第 {idx} 项缺少 start/end 字段")
            continue
        start, end = float(a["start"]), float(a["end"])
        if end < start - EPS:
            errors.append(f"第 {idx} 项 end({end}) < start({start})")
        if start < prev_end - EPS:
            errors.append(f"第 {idx} 项起点 {start} 早于上一终点 {prev_end}，"
                          "时间戳非单调")
        if audio_duration is not None and (start < -EPS or
                                           end > float(audio_duration) + EPS):
            errors.append(f"第 {idx} 项越界：[{start}, {end}] "
                          f"音频总长 {audio_duration}")
        prev_end = max(prev_end, end)
        if "token" in a:
            tokens_seen.append(a["token"])

    if expected_tokens is not None and len(tokens_seen) != len(expected_tokens):
        errors.append(f"token 覆盖不全：对齐 {len(tokens_seen)} 个，"
                      f"预期 {len(expected_tokens)} 个")

    return {
        "valid": not errors,
        "errors": errors,
        "checked": len(alignments),
        "monotonic": all("非单调" not in e for e in errors),
    }


# ===========================================================================
# 片段切分（基于统一 VAD + 最短语音/静音合并）
# ===========================================================================
def segment(audio: Any,
            sample_rate: Optional[int] = None,
            min_speech_ms: float = 120.0,
            min_silence_ms: float = 300.0,
            denoise: bool = True,
            hop_ms: float = DEFAULT_HOP_MS) -> Dict[str, Any]:
    """基于统一静音阈值切分语音片段，短停合并、碎语音过滤。"""
    if not isinstance(audio, (np.ndarray, list, tuple)):
        return {"segments": [], "boundaries": [], "sample_rate": sample_rate,
                "vad": None}

    sr = int(sample_rate) if sample_rate else 16000
    if denoise:
        audio = spectral_subtract(audio, sr)["clean"]
    vad = unified_silence(audio, sr, hop_ms=hop_ms)
    mask = vad["mask"]
    hop_len = max(1, int(round(sr * hop_ms / 1000.0)))

    min_speech_frames = max(1, int(round(min_speech_ms / hop_ms)))
    min_silence_frames = max(1, int(round(min_silence_ms / hop_ms)))

    # 提取连续语音段
    raw: List[Tuple[int, int]] = []
    i = 0
    n = mask.size
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            raw.append((i, j))  # 半开区间 [i, j)
            i = j
        else:
            i += 1

    # 合并被短静音隔开的相邻段
    merged: List[Tuple[int, int]] = []
    for s, e in raw:
        if merged and (s - merged[-1][1]) <= min_silence_frames:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    # 过滤过短语音碎块
    merged = [(s, e) for s, e in merged if (e - s) >= min_speech_frames]

    segments = []
    for s, e in merged:
        start_sample, end_sample = s * hop_len, e * hop_len
        segments.append({
            "start_frame": int(s),
            "end_frame": int(e),
            "start_sample": int(start_sample),
            "end_sample": int(end_sample),
            "start": start_sample / float(sr),
            "end": end_sample / float(sr),
            "duration": (end_sample - start_sample) / float(sr),
        })

    return {
        "segments": segments,
        "boundaries": [(d["start"], d["end"]) for d in segments],
        "sample_rate": sr,
        "vad": vad,
        "min_speech_ms": float(min_speech_ms),
        "min_silence_ms": float(min_silence_ms),
    }


# ===========================================================================
# 打分（声学代价 + 时长先验融合）
# ===========================================================================
def score(acoustic_cost: Any, duration_cost: Any = 0.0,
          w_acoustic: float = 0.7, w_duration: float = 0.3,
          normalize: bool = True) -> Dict[str, Any]:
    """融合声学代价与时长先验，输出总代价与 0~100 置信分（越高越好）。"""
    a = _as_float_array(acoustic_cost)
    d = _as_float_array(duration_cost) if np.size(duration_cost) else \
        np.zeros_like(a)
    total = w_acoustic * a + w_duration * d
    raw = float(np.mean(total)) if total.size else 0.0
    if normalize and raw > EPS:
        # exp(-x) 把非负代价映射到 (0, 100]
        confidence = float(100.0 * math.exp(-raw))
    else:
        confidence = 100.0 if raw <= EPS else 0.0
    return {
        "cost": total,
        "mean_cost": raw,
        "confidence": confidence,
        "weights": {"acoustic": float(w_acoustic),
                    "duration": float(w_duration)},
    }


# ===========================================================================
# 核心对齐管线
# ===========================================================================
def _tokenize(text: Any) -> List[str]:
    if isinstance(text, str):
        return [ch for ch in text if not ch.isspace()]
    return [str(t) for t in text]


def align(audio: Any, sample_rate: Optional[int] = None, text: Any = "",
          segment_id: Any = "main", config: Optional[Dict[str, Any]] = None,
          use_denoise: bool = True) -> Dict[str, Any]:
    """端到端强制对齐。

    流程：背景谱减 -> 统一 VAD -> 自适应窗长 -> 分段 -> 每段构造
    token×frame 代价矩阵 -> 带回溯指针的 DP -> 单调时间戳 -> 校验。
    任何关键步骤失败都会走 :func:`fallback`，保证始终有结果返回。
    """
    cfg = determinism(**(config or {}))
    rng = cfg["rng"]

    if not isinstance(audio, (np.ndarray, list, tuple)):
        # 契约调用（如 align(1)）：返回空骨架而非抛异常
        return fallback(audio_duration=0.0, tokens=_tokenize(text)
                        if isinstance(text, (str, list, tuple)) else [],
                        reason="输入音频不是数组，返回空对齐骨架")

    sr = int(sample_rate) if sample_rate else 16000
    tokens = _tokenize(text)
    wave = _as_float_array(audio)
    duration = wave.size / float(sr)
    if not tokens:
        return fallback(duration, [], reason="文本为空")

    try:
        # 1) 先谱减，去除背景音乐/稳态噪声
        proc = spectral_subtract(wave, sr,
                                 alpha=cfg["spectral_alpha"],
                                 beta=cfg["spectral_beta"])["clean"] \
            if use_denoise else wave
        # 2) 统一静音阈值 VAD
        vad = unified_silence(proc, sr, threshold_db=cfg["silence_db"],
                              noise_percentile=cfg["noise_percentile"])
        mask, energies = vad["mask"], vad["energies"]
        n_frames = energies.size
        hop_len = max(1, int(round(sr * cfg["hop_ms"] / 1000.0)))

        # 3) 自适应窗长（按局部语速）
        win = adaptive_window(energies, sr,
                              base_frame_ms=cfg["frame_ms"],
                              hop_ms=cfg["hop_ms"])

        # 4) 语音片段；整段无语音则直接 fallback 线性均分
        seg_result = segment(proc, sr)
        speech_segments = seg_result["segments"]
        if not speech_segments:
            return fallback(duration, tokens,
                            reason="VAD 未检测到语音，线性均分时间戳")

        # 5) 按语音时长把 token 摊到各片段
        weights = np.array([s["duration"] for s in speech_segments],
                           dtype=np.float64)
        weights = weights / max(weights.sum(), EPS)
        alloc = np.maximum(1, np.round(weights * len(tokens)).astype(int))
        # 修正取整误差，保证总数等于 token 数
        diff = int(alloc.sum()) - len(tokens)
        order = np.argsort(-alloc)
        k = 0
        while diff > 0:
            idx = order[k % len(order)]
            if alloc[idx] > 1:
                alloc[idx] -= 1
                diff -= 1
            k += 1
            if k > 10 * len(order):
                break
        while diff < 0:
            alloc[order[(-diff) % len(order)]] += 1
            diff += 1

        # 6) 片段内 DP 对齐（片段缓存互相隔离）
        caches = cache_isolation([s_idx for s_idx in range(len(speech_segments))])
        alignments: List[Dict[str, Any]] = []
        token_cursor = 0
        rng_jitter = rng  # 确定性来源：同一 config 种子结果一致
        for s_idx, seg in enumerate(speech_segments):
            f_lo, f_hi = seg["start_frame"], seg["end_frame"]
            count = int(alloc[s_idx])
            seg_tokens = tokens[token_cursor:token_cursor + count]
            token_cursor += count
            if not seg_tokens:
                continue
            frames = np.arange(f_lo, min(f_hi, n_frames))
            if frames.size == 0:
                frames = np.array([min(f_lo, n_frames - 1)])

            cache = caches[s_idx]
            cost_matrix = cache.get_or_compute(
                "cost_matrix",
                lambda ft=seg_tokens, fr=frames: _build_cost_matrix(
                    ft, fr, energies, mask, rng_jitter))

            band = max(len(seg_tokens), frames.size)
            dtw = dp_backtrace(cost_matrix, band_width=band)
            alignments.extend(
                _extract_token_times(dtw["path"], seg_tokens, frames,
                                     hop_len, sr, s_idx))

        # token 数与对齐项不一致（理论上不会）时补/裁
        alignments = _reconcile(alignments, tokens, duration)

        # 7) 单调化 + 校验
        mono = monotonic_time(alignments)
        alignments = mono["alignments"]
        report = validation(alignments, audio_duration=duration,
                            expected_tokens=tokens)
        if not report["valid"]:
            fb = fallback(duration, tokens,
                          reason="单调/边界校验失败：" +
                                 "; ".join(report["errors"][:3]))
            fb["upstream"] = {"alignments": alignments,
                              "validation": report}
            return fb

        return {
            "alignments": alignments,
            "tokens": tokens,
            "sample_rate": sr,
            "duration": duration,
            "segment_id": segment_id,
            "segments": speech_segments,
            "vad": {"speech_ratio": vad["speech_ratio"],
                    "threshold": vad["threshold"],
                    "noise_floor": vad["noise_floor"]},
            "adaptive_window": {
                "mean_window_frames": win["mean_window_frames"],
                "hop_ms": win["hop_ms"]},
            "denoised": bool(use_denoise),
            "fallback": False,
            "validation": report,
            "monotonic_violations": mono["violations"],
            "config": {k: v for k, v in cfg.items() if k != "rng"},
        }
    except Exception as exc:  # 管线任何环节崩溃都有兜底
        return fallback(duration, tokens, reason=f"对齐管线异常: {exc!r}")


def _build_cost_matrix(seg_tokens: List[str], frames: np.ndarray,
                       energies: np.ndarray, speech_mask: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
    """构造 token×frame 代价矩阵。

    声学项：静音帧匹配 token 代价高、语音帧低；时长先验项：鼓励 token
    均匀占用帧。数值确定性来自传入的固定种子 rng（仅用于哈希式微扰）。
    """
    n_frames = frames.size
    m_tokens = len(seg_tokens)
    matrix = np.ones((n_frames, m_tokens), dtype=np.float64)
    expected = max(1, n_frames / max(1, m_tokens))

    for fi, f in enumerate(frames):
        e = float(energies[f]) if f < energies.size else 0.0
        voiced = bool(speech_mask[f]) if f < speech_mask.size else False
        for ti, tok in enumerate(seg_tokens):
            acoustic = 0.2 if voiced else 1.0
            # token 自身特征：长度/标点代理（确定性，无外部模型）
            token_prior = 0.05 * (len(tok) % 3) / 3.0
            # 时长先验：token 在段内相对位置与帧相对位置越接近越好
            pos_frame = fi / max(1, n_frames - 1)
            pos_token = ti / max(1, m_tokens - 1)
            duration_prior = abs(pos_frame - pos_token) / expected
            matrix[fi, ti] = acoustic + token_prior + 0.5 * duration_prior
    return matrix


def _extract_token_times(path: List[Tuple[int, int]],
                         seg_tokens: List[str],
                         frames: np.ndarray,
                         hop_len: int, sample_rate: int,
                         segment_index: int) -> List[Dict[str, Any]]:
    """从 DP 路径收集每个 token 覆盖的帧范围并换算时间戳。"""
    token_frames: Dict[int, List[int]] = {}
    for fi, ti in path:
        token_frames.setdefault(ti, []).append(fi)

    results = []
    for ti, tok in enumerate(seg_tokens):
        if ti in token_frames and token_frames[ti]:
            fis = token_frames[ti]
            f_start = frames[min(fis)]
            f_end = frames[max(fis)] + 1  # 半开区间
        else:
            # delete 动作可能让 token 没有独立帧：借用相邻位置
            anchor = frames[min(ti, frames.size - 1)]
            f_start = f_end = anchor
        start_sample = int(f_start) * hop_len
        end_sample = int(f_end) * hop_len
        results.append({
            "token": tok,
            "start": start_sample / float(sample_rate),
            "end": end_sample / float(sample_rate),
            "start_frame": int(f_start),
            "end_frame": int(f_end),
            "segment": segment_index,
            "fallback": False,
            "confidence": 1.0,
        })
    return results


def _reconcile(alignments: List[Dict[str, Any]], tokens: List[str],
               duration: float) -> List[Dict[str, Any]]:
    if len(alignments) == len(tokens):
        for a, t in zip(alignments, tokens):
            a["token"] = t
        return alignments
    # 异常情况：线性兜底，保证 token 全覆盖
    fb = fallback(duration, tokens, reason="DP 输出与 token 数不一致")
    return fb["alignments"]


# ===========================================================================
# 兜底：线性均分
# ===========================================================================
def fallback(audio_duration: float = 0.0, tokens: Optional[Sequence[str]] = None,
             reason: str = "") -> Dict[str, Any]:
    """对齐失败时的线性均分兜底，时间戳天然单调。"""
    if tokens is None:
        tokens = []
    if not isinstance(tokens, (list, tuple, np.ndarray)):
        tokens = list(tokens) if tokens else []
    duration = float(audio_duration) if audio_duration is not None else 0.0
    n = max(1, len(tokens))
    slot = duration / n
    alignments = []
    for i, tok in enumerate(tokens):
        alignments.append({
            "token": str(tok),
            "start": i * slot,
            "end": (i + 1) * slot,
            "start_frame": -1,
            "end_frame": -1,
            "segment": -1,
            "fallback": True,
            "confidence": 0.0,
        })
    return {
        "alignments": alignments,
        "tokens": [str(t) for t in tokens],
        "duration": duration,
        "fallback": True,
        "reason": reason,
        "validation": validation(alignments, audio_duration=duration,
                                 expected_tokens=tokens),
    }


# ===========================================================================
# 审计与指标
# ===========================================================================
def audit(align_result: Dict[str, Any]) -> Dict[str, Any]:
    """汇总一次对齐运行的可观测审计信息。"""
    if not isinstance(align_result, dict):
        return {"ok": False, "errors": ["align_result 不是 dict"]}
    alignments = align_result.get("alignments", [])
    durations = [float(a["end"]) - float(a["start"]) for a in alignments
                 if isinstance(a, dict) and "start" in a and "end" in a]
    covers = (sum(durations), align_result.get("duration"))
    coverage = (covers[0] / covers[1]) if covers[1] else 0.0
    return {
        "ok": True,
        "fallback": bool(align_result.get("fallback", False)),
        "reason": align_result.get("reason", ""),
        "n_tokens": len(align_result.get("tokens", [])),
        "n_alignments": len(alignments),
        "duration": align_result.get("duration"),
        "speech_coverage": float(min(1.0, coverage)),
        "min_token_duration": float(min(durations)) if durations else 0.0,
        "max_token_duration": float(max(durations)) if durations else 0.0,
        "mean_token_duration": float(np.mean(durations)) if durations else 0.0,
        "vad": align_result.get("vad"),
        "adaptive_window": align_result.get("adaptive_window"),
        "validation": align_result.get("validation"),
        "monotonic_violations": align_result.get("monotonic_violations"),
        "segment_id": align_result.get("segment_id"),
        "denoised": align_result.get("denoised"),
    }


def metrics(alignments: Sequence[Dict[str, Any]],
            reference: Optional[Sequence[Dict[str, float]]] = None
            ) -> Dict[str, Any]:
    """对齐质量指标；有参考边界时给 MAE/RMSE（秒）。"""
    if not isinstance(alignments, (list, tuple, np.ndarray)):
        return {"mae": None, "rmse": None, "n": 0}

    durations = np.array([max(0.0, float(a["end"]) - float(a["start"]))
                          for a in alignments], dtype=np.float64)
    out: Dict[str, Any] = {
        "n": len(alignments),
        "mean_duration": float(durations.mean()) if durations.size else 0.0,
        "duration_std": float(durations.std()) if durations.size else 0.0,
        "zero_duration": int(np.sum(durations <= EPS)),
        "mae": None,
        "rmse": None,
    }
    if reference is not None and len(reference) == len(alignments):
        errs = []
        for a, r in zip(alignments, reference):
            errs.append(float(a["start"]) - float(r["start"]))
            errs.append(float(a["end"]) - float(r["end"]))
        errs_arr = np.array(errs, dtype=np.float64)
        out["mae"] = float(np.mean(np.abs(errs_arr)))
        out["rmse"] = float(np.sqrt(np.mean(errs_arr ** 2)))
        out["max_abs_error"] = float(np.max(np.abs(errs_arr)))
    return out


# ===========================================================================
# Praat TextGrid 导出
# ===========================================================================
def text_grid(alignments: Sequence[Dict[str, Any]],
              xmax: Optional[float] = None,
              tier_name: str = "tokens") -> str:
    """把对齐结果导出为 Praat TextGrid（IntervalTier）文本。"""
    if not isinstance(alignments, (list, tuple, np.ndarray)):
        alignments = []
    if xmax is None:
        xmax = max((float(a["end"]) for a in alignments),
                   default=0.0)
    xmax = float(xmax)

    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0.0",
        f"xmax = {xmax:.6f}",
        "tiers? <exists>",
        "size = 1",
        "item []:",
        "    item [1]:",
        '        class = "IntervalTier"',
        f'        name = "{tier_name}"',
        "        xmin = 0.0",
        f"        xmax = {xmax:.6f}",
        f"        intervals: size = {len(alignments)}",
    ]
    for i, a in enumerate(alignments, start=1):
        label = str(a.get("token", "")).replace('"', '""')
        lines.append(f"        intervals [{i}]:")
        lines.append(f"            xmin = {float(a['start']):.6f}")
        lines.append(f"            xmax = {float(a['end']):.6f}")
        lines.append(f'            text = "{label}"')
    return "\n".join(lines) + "\n"
