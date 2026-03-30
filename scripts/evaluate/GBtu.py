"""
Memory Trace Visualization
==========================
按论文 Figure 17 风格绘制三种后端的内存堆叠面积图。
每种后端统一包含：weight / reserve / kv_used(各类型) / wasted / free

用法：
  python plot_memory_trace.py
"""

import json, os, re, glob, tempfile, math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family":       "DejaVu Serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── 颜色（与论文图风格对应）────────────────────────────────────────────────
#   绿  = weight
#   蓝  = reserve（激活预留）
#   青  = kv_used / used_trans / used_full
#   黄  = used_window / used_swa
#   紫  = used_state（Mamba）
#   红  = wasted（已分配但无用）
#   灰  = free（未分配）
COLORS = {
    "weight":       "#6DBF67",   # 绿
    "reserve":      "#7EC8E3",   # 蓝
    "kv_used":      "#40C4AA",   # 青  (vLLM 纯注意力)
    "used_full":    "#40C4AA",   # 青  (full attention)
    "used_window":  "#FFD166",   # 黄  (sliding window)
    "used_swa":     "#FFD166",   # 黄  (vattn swa)
    "used_trans":   "#40C4AA",   # 青  (vattn trans)
    "used_state":   "#A78BFA",   # 紫  (mamba)
    "kv_wasted":    "#E05C5C",   # 红  (wasted)
    "free":         "#D1D5DB",   # 浅灰
}

# ════════════════════════════════════════════════════════════════════════════
# 文件发现
# ════════════════════════════════════════════════════════════════════════════

def extract_type(path):
    m = re.search(r"_attn_(.+?)_cl", path)
    return m.group(1) if m else os.path.basename(os.path.dirname(path))

def find_traces(base_dir):
    paths = glob.glob(os.path.join(base_dir, "**", "memory_trace.json"),
                      recursive=True)
    result = {}
    for p in sorted(paths):
        label = extract_type(p)
        result[label] = p
        print(f"  ✓ [{label}]  {p}")
    return result

# ════════════════════════════════════════════════════════════════════════════
# 数据解析
# ════════════════════════════════════════════════════════════════════════════

def load_trace(path):
    with open(path) as f:
        data = json.load(f)
    return sorted(data, key=lambda x: x.get("step", 0))

def _arr(records, key, default=0.0):
    return np.array([r.get(key, default) for r in records], dtype=float)

def steps_arr(records):
    return np.array([r.get("step", i) for i, r in enumerate(records)])

def detect_backend(records, label=""):
    if not records:
        return "vllm"
    r0 = records[0]
    t  = r0.get("type", "")
    if t:
        return t
    if "kv_used_gb"    in r0: return "vllm"
    if "used_full_gb"  in r0: return "vllm_hybrid"
    if "used_trans_gb" in r0: return "vattn_hybrid"
    lbl = label.lower()
    if "vattn"  in lbl:                      return "vattn_hybrid"
    if "hybrid" in lbl or "hybird" in lbl:   return "vllm_hybrid"
    return "vllm"

# ── 每种后端构建统一结构 series ──────────────────────────────────────────────
# series 统一字段：
#   steps, weight, reserve
#   以及若干 kv_* 字段（随后端类型变化）
#   kv_wasted, free

VLLM_TYPES   = ("vllm", "fi_paged", "fa_paged",
                "fi_serial_paged", "fi_unpaged")
HYBRID_TYPES = ("vllm_hybrid", "fi_paged_hybird", "fa_paged_hybird")


def build_series_vllm(records):
    """
    vLLM PagedAttention：
      weight | reserve | kv_used | kv_wasted | free
    """
    return {
        "steps":     steps_arr(records),
        "weight":    _arr(records, "weight_gb"),
        "reserve":   _arr(records, "reserve_gb"),
        "kv_used":   _arr(records, "kv_used_gb"),
        "kv_wasted": _arr(records, "wasted_gb"),
        "free":      _arr(records, "kv_free_gb"),
    }


def build_series_hybrid(records):
    """
    vLLM Hybrid：
      weight | reserve | used_full | used_window | used_state | kv_wasted | free
    """
    return {
        "steps":       steps_arr(records),
        "weight":      _arr(records, "weight_gb"),
        "reserve":     _arr(records, "reserve_gb"),
        "used_full":   _arr(records, "used_full_gb"),
        "used_window": _arr(records, "used_window_gb"),
        "used_state":  _arr(records, "used_state_gb"),
        "kv_wasted":   _arr(records, "wasted_gb"),
        "free":        _arr(records, "kv_free_gb"),
    }


def build_series_vattn(records):
    """
    vAttention Hybrid：
      weight | reserve | used_trans | used_swa | used_state | kv_wasted | free
    """
    return {
        "steps":      steps_arr(records),
        "weight":     _arr(records, "weight_gb"),
        "reserve":    _arr(records, "reserve_gb"),
        "used_trans": _arr(records, "used_trans_gb"),
        "used_swa":   _arr(records, "used_swa_gb"),
        "used_state": _arr(records, "used_state_gb"),
        "kv_wasted":  _arr(records, "wasted_gb"),
        "free":       _arr(records, "kv_free_gb"),
    }


def build_series(records, backend):
    if backend in VLLM_TYPES:   return build_series_vllm(records)
    if backend in HYBRID_TYPES: return build_series_hybrid(records)
    return build_series_vattn(records)


# ── 每种后端的绘图层定义（从下到上）─────────────────────────────────────────
def get_layers(series, backend):
    """返回 [(values_array, color, legend_label), ...]，从下到上堆叠。"""
    base = [
        (series["weight"],  COLORS["weight"],  "weight"),
        (series["reserve"], COLORS["reserve"], "reserve"),
    ]

    if backend in VLLM_TYPES:
        mid = [
            (series["kv_used"],   COLORS["kv_used"],   "used"),
        ]
    elif backend in HYBRID_TYPES:
        mid = [
            (series["used_full"],   COLORS["used_full"],   "used-full"),
            (series["used_window"], COLORS["used_window"], "used-window"),
            (series["used_state"],  COLORS["used_state"],  "used-mamba"),
        ]
    else:  # vattn
        mid = [
            (series["used_trans"], COLORS["used_trans"], "used-full"),
            (series["used_swa"],   COLORS["used_swa"],   "used-window"),
            (series["used_state"], COLORS["used_state"], "used-mamba"),
        ]

    top = [
        (series["kv_wasted"], COLORS["kv_wasted"], "wasted"),
        (series["free"],      COLORS["free"],      "unallocated"),
    ]
    return base + mid + top

# ════════════════════════════════════════════════════════════════════════════
# 绘图
# ════════════════════════════════════════════════════════════════════════════

def plot_panel(ax, series, backend, title, ylabel=True, ylim=None):
    x      = series["steps"]
    layers = get_layers(series, backend)
    bottom = np.zeros(len(x))

    for vals, color, label in layers:
        vals = np.nan_to_num(vals, nan=0.0)
        ax.fill_between(x, bottom, bottom + vals,
                        color=color, alpha=0.88,
                        label=label, linewidth=0)
        bottom += vals

    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Forward Step", fontsize=10)
    if ylabel:
        ax.set_ylabel("Memory (GB)", fontsize=10)

    top = ylim if ylim else max(bottom.max() * 1.08, 1.0)
    ax.set_ylim(0, top)
    ax.set_xlim(x[0], x[-1])
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.6)
    ax.set_axisbelow(True)

# ════════════════════════════════════════════════════════════════════════════
# 演示数据（找不到真实文件时使用）
# ════════════════════════════════════════════════════════════════════════════

def make_demo_traces():
    tmpdir = tempfile.mkdtemp()
    demo   = {}
    N      = 120
    rng    = np.random.default_rng(42)

    def save(name, recs):
        p = os.path.join(tmpdir, f"model_attn_{name}_cl_test", "memory_trace.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(recs, open(p, "w"))
        demo[name] = p

    # ── vLLM PagedAttention ──────────────────────────────────────────────
    recs = []
    for i in range(N):
        t = i / N
        kv_used   = 12 + 30 * t + rng.uniform(-1, 1)
        kv_used   = max(kv_used, 0)
        allocated = kv_used + 6 + 4 * t + rng.uniform(0, 1)  # 总分配 > used
        wasted    = max(allocated - kv_used, 0)
        reserved  = max(65 - kv_used - wasted, 0)
        recs.append({
            "step": i, "type": "vllm",
            "weight_gb":   4.2,
            "reserve_gb":  3.1,           # profiling 时记录的激活预留
            "kv_used_gb":  kv_used,
            "kv_wasted_gb":wasted,
            "kv_free_gb":  reserved,
            "timestamp": float(i),
        })
    save("fi_paged", recs)

    # ── vLLM Hybrid ──────────────────────────────────────────────────────
    recs = []
    for i in range(N):
        t = i / N
        full   = 10 + 15 * t + rng.uniform(-0.5, 0.5)
        window = 6  +  8 * t + rng.uniform(-0.3, 0.3)
        state  = 1.5 + 1 * t
        total_used = full + window + state
        # 分配略多于 used（block 粒度导致浪费）
        wasted = 2 + 3 * t * rng.uniform(0.5, 1.5)
        free   = max(45 - total_used - wasted, 0)
        recs.append({
            "step": i, "type": "vllm_hybrid",
            "weight_gb":      4.2,
            "reserve_gb":     3.1,
            "used_full_gb":   max(full,   0),
            "used_window_gb": max(window, 0),
            "used_state_gb":  max(state,  0),
            "kv_wasted_gb":   max(wasted, 0),
            "kv_free_gb":     free,
            "timestamp": float(i),
        })
    save("fi_paged_hybird", recs)

    # ── vAttention Hybrid ────────────────────────────────────────────────
    recs = []
    for i in range(N):
        t = i / N
        trans  = 12 + 18 * t + rng.uniform(-0.4, 0.4)
        swa    = 5  +  6 * t + rng.uniform(-0.3, 0.3)
        state  = 1.5 + 0.5 * t
        total_used = trans + swa + state
        # vAttention 以 2MB 为粒度，浪费较小
        wasted = 1 + 1.5 * t * rng.uniform(0.3, 0.8)
        free   = max(48 - total_used - wasted, 0)
        recs.append({
            "step": i, "type": "vattn_hybrid",
            "weight_gb":    4.2,
            "reserve_gb":   3.1,
            "used_trans_gb":max(trans, 0),
            "used_swa_gb":  max(swa,   0),
            "used_state_gb":max(state, 0),
            "kv_wasted_gb": max(wasted,0),
            "kv_free_gb":   free,
            "timestamp": float(i),
        })
    save("fi_vattn_2mb", recs)

    print(f"  [演示模式] 临时数据：{tmpdir}")
    return demo

# ════════════════════════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════════════════════════

def main():
    # ------------------------------------------------------------------ #
    #  ★ 修改 BASE_DIR 为你的实验目录                                        #
    # ------------------------------------------------------------------ #
    BASE_DIR    = "/workspace/vatten_hybrid/experiments/e2e_static_eval_test_GB_500"
    OUTPUT_DIR  = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_NAME = "memory_comparison"
    YLIM        = None   # 统一 y 上限(GB)，None = 自动

    BASE_DIR = os.path.abspath(BASE_DIR)
    print(f"\n搜索 JSON：{BASE_DIR}")

    if not os.path.isdir(BASE_DIR):
        print(f"⚠  目录不存在，使用演示数据。")
        traces = make_demo_traces()
    else:
        traces = find_traces(BASE_DIR)
        if not traces:
            print("⚠  未找到 memory_trace.json，使用演示数据。")
            traces = make_demo_traces()

    n = len(traces)
    if n == 0:
        print("无数据，退出。")
        return
    print(f"共 {n} 个后端，开始绘图…")

    # ── 自动 y 上限 ───────────────────────────────────────────────────────
    if YLIM is None:
        peaks = []
        for label, path in traces.items():
            recs    = load_trace(path)
            backend = detect_backend(recs, label)
            series  = build_series(recs, backend)
            total   = sum(np.nan_to_num(v, nan=0.0)
                          for v, _, _ in get_layers(series, backend))
            peaks.append(total.max())
        YLIM = max(peaks) * 1.1 if peaks else 80.0

    # ── 画布 ──────────────────────────────────────────────────────────────
    fig_w = max(5 * n, 12)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 5))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("#F8F8F6")

    all_handles, all_labels = [], []

    for idx, (label, path) in enumerate(sorted(traces.items())):
        recs    = load_trace(path)
        backend = detect_backend(recs, label)
        series  = build_series(recs, backend)

        ax = axes[idx]
        ax.set_facecolor("#FCFCFA")
        letter = chr(ord('a') + idx)
        plot_panel(ax, series, backend,
                   title=f"({letter}) {label}",
                   ylabel=(idx == 0),
                   ylim=YLIM)

        for hi, li in zip(*ax.get_legend_handles_labels()):
            if li not in all_labels:
                all_handles.append(hi)
                all_labels.append(li)

    # ── 图例（顶部，按固定顺序排列）──────────────────────────────────────
    # 期望顺序：weight reserve used used-full used-window used-mamba wasted unallocated
    order = ["weight", "reserve", "used", "used-full",
             "used-window", "used-mamba", "wasted", "unallocated"]
    ordered_h, ordered_l = [], []
    for name in order:
        if name in all_labels:
            i = all_labels.index(name)
            ordered_h.append(all_handles[i])
            ordered_l.append(all_labels[i])
    # 补上未预期的标签
    for h, l in zip(all_handles, all_labels):
        if l not in ordered_l:
            ordered_h.append(h)
            ordered_l.append(l)

    fig.legend(ordered_h, ordered_l,
               loc="upper center", bbox_to_anchor=(0.5, 1.03),
               ncol=min(len(ordered_l), 8),
               frameon=True, framealpha=0.92,
               edgecolor="#CCCCCC", fontsize=10)

    fig.suptitle(
        "Memory Usage Timeline — Gemma-3-4B  (prefill : decode = 500 : 1)",
        y=1.10, fontsize=13, fontweight="bold", color="#222222")

    plt.tight_layout(rect=[0, 0, 1, 1])

    # ── 保存 ─────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"✓ 保存：{out}")

    plt.close(fig)
    print("完成。")


if __name__ == "__main__":
    main()

