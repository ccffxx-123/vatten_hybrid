import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

helpers = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(helpers)
logs = os.path.join(root, 'logs/figure_6/')

plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'font.family': 'Sans Serif'})

configs = ["FA_Paged", "FI_Paged", "FA_vAttention", "FI_vAttention"]

config_labels = {
    'FA_Paged': 'FA_Paged',
    'FI_Paged': 'FI_Paged',
    'FA_vAttention': 'FA_vAttention',
    'FI_vAttention': 'FI_vAttention',
}

colors = {
    'FA_Paged': 'chocolate',
    'FI_Paged': 'green',
    'FA_vAttention': 'chocolate',
    'FI_vAttention': 'green',
}

linestyles = {
    'FA_Paged': '-',
    'FI_Paged': '-',
    'FA_vAttention': '--',
    'FI_vAttention': '--',
}

markerstyles = {
    'FA_Paged': 'o',
    'FI_Paged': 's',
    'FA_vAttention': 'D',
    'FI_vAttention': '^'
}

def plot_figure(df, figname, title):
    plt.figure(figsize=(10, 5))
    y_max = -1
    context_lengths = df['cl'].unique()
    x_ticks = np.arange(len(context_lengths))
    
    for i, cfg in enumerate(configs):
        if cfg not in df.columns:
            print(f"Warning: {cfg} not found in data, skipping...")
            continue
        plt.plot(x_ticks, df[cfg], label=config_labels[cfg], color=colors[cfg], linestyle=linestyles[cfg], linewidth=2, marker=markerstyles[cfg], markersize=7)
        y_max = max(y_max, df[cfg].max())

    if y_max <= 0:
        print(f"Warning: No valid data for {title}")
        plt.close()
        return

    plt.xlabel('Context Length', fontsize=24, fontweight='bold')
    plt.ylabel('Tokens/second', fontsize=24, fontweight='bold')
    x_labels = [str(cl//1024) + "K" for cl in context_lengths]
    plt.xticks(x_ticks, x_labels, fontsize=18)
    plt.yticks(np.arange(0, y_max*1.1, 3000), fontsize=18)
    plt.grid()
    plt.tight_layout()
    plt.title(title, fontsize=18, fontweight='bold')
    plt.legend(loc='lower left', ncols=1, fontsize=22)
    plt.savefig(figname)
    plt.close()

record = {}

def get_substring(string, start, end):
    start_idx = string.find(start)
    if start_idx == -1:
        return None
    start_idx += len(start)
    end_idx = string.find(end, start_idx)
    if end_idx == -1:
        return None
    return string[start_idx:end_idx]

def prettify_model_name(model):
    return 'Yi-6B' if model == 'yi-6b' else \
            'Yi-34B' if model == 'yi-34b' else \
            'Llama-3-8B' if model == 'llama-3-8b' else model

def prettify_attn_name(attn):
    return 'FA_Paged' if attn == 'fa_paged' else \
            'FI_Paged' if attn == 'fi_paged' else \
            'FA_vAttention' if attn == 'fa_vattn' else \
            'FI_vAttention' if attn == 'fi_vattn' else attn

def read_perf_record(path):
    # 适配新的目录格式: model_{model}_tp_{tp}_attn_{attn}_cl_{cl}_pd_{pd}_reqs_{reqs}
    model = get_substring(path, 'figure_6/model_', '_tp_')
    cl_str = get_substring(path, '_cl_', '_pd_')
    attn = get_substring(path, '_attn_', '_cl_')
    pd_str = get_substring(path, '_pd_', '_reqs_')
    
    if model is None or cl_str is None or attn is None or pd_str is None:
        print(f"Warning: Failed to parse path: {path}")
        return
    
    cl = int(cl_str)
    pd_ratio = int(pd_str)
    attn = prettify_attn_name(attn)
    
    try:
        df_csv = pd.read_csv(path)
    except Exception as e:
        print(f"Warning: Failed to read {path}: {e}")
        return
    
    if 'cdf' not in df_csv.columns or 'prefill_time_execution_plus_preemption' not in df_csv.columns:
        print(f"Warning: Required columns not found in {path}")
        return
    
    df_filtered = df_csv[df_csv['cdf'] >= 0.5].sort_values(by='cdf')
    if len(df_filtered) == 0:
        print(f"Warning: No data with cdf >= 0.5 in {path}")
        return
    
    latency_sorted = df_filtered['prefill_time_execution_plus_preemption'].sort_values()
    latency = latency_sorted.iloc[0]
    if len(latency_sorted) >= 5:
        latency = latency_sorted.iloc[0:5].mean()
    
    if latency <= 0:
        print(f"Warning: Invalid latency in {path}")
        return
    
    # 使用 (model, pd_ratio) 作为第一层 key
    key = (model, pd_ratio)
    if key not in record:
        record[key] = {}
    if cl not in record[key]:
        record[key][cl] = {}
    if attn not in record[key][cl]:
        record[key][cl][attn] = int(cl / latency)

def read_logs():
    for dirpath, dirs, files in os.walk(logs):
        for file in files:
            if file == 'prefill_time_execution_plus_preemption.csv':
                filepath = os.path.join(dirpath, file)
                read_perf_record(filepath)

read_logs()

if not record:
    print("Error: No data found in logs directory")
    exit(1)

# 按 (model, pd_ratio) 分组处理
for (model, pd_ratio), cl_data in record.items():
    # 构建 DataFrame
    df = pd.DataFrame.from_dict(cl_data, orient='index')
    df = df.reset_index().rename(columns={'index': 'cl'})
    df.fillna(0, inplace=True)
    df = df.sort_values(by='cl')
    
    print(f"\nData summary for model={model}, pd={pd_ratio}:")
    print(df)
    
    title = f"{prettify_model_name(model)} (P:D = {pd_ratio}:1)"
    figname = os.path.join(root, f'plots/figure_6_{prettify_model_name(model)}_pd_{pd_ratio}.pdf')
    os.makedirs(os.path.dirname(figname), exist_ok=True)
    plot_figure(df, figname, title)
    print(f"Generated: {figname}")