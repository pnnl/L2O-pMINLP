import re
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from scipy import stats
import matplotlib.pyplot as plt
import os

def read_log_file(file_path):
    encodings = ['utf-8', 'gb18030', 'latin1']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ValueError("无法自动检测文件编码")

def extract_tables(log_content):
    """提取日志文件中的表格内容"""

    header_pattern = r"""
        Nodes\s+\|\s+Current\s+Node\s+\|\s+Objective\s+Bounds\s+\|\s+Work\s+
        Expl\s+Unexpl\s+\|\s+Obj\s+Depth\s+IntInf\s+\|\s+Incumbent\s+BestBd\s+Gap\s+\|\s+It/Node\s+Time
    """
    end_pattern = r"""
            Explored
        """
    # 匹配表头及其后续内容
    start_pattern = re.compile(header_pattern.strip(), re.VERBOSE | re.MULTILINE)
    end_pattern = re.compile(end_pattern.strip(), re.VERBOSE | re.MULTILINE)
    start_matches = list(start_pattern.finditer(log_content))
    end_matches = list(end_pattern.finditer(log_content))
    tables = []

    for i in range(len(start_matches)):
        start = start_matches[i].end()
        end = end_matches[i].start() if i < len(end_matches) else len(log_content)
        table_content = log_content[start:end].strip()
        tables.append(table_content)
    return tables

def parse_table(table_str):
    data = []
    lines = [ln.strip() for ln in table_str.split('\n') if ln.strip() and not ln.startswith(('+-', '|--'))]
    if len(lines) >= 2:
        for line in lines:
            cols = re.split(r'\s+', line.strip())
            time_str = cols[-1].replace('s', '').strip()
            time = float(time_str) if '.' in time_str else int(time_str)

            gap_str = cols[-3]
            if gap_str == "-":
                gap = 10000
            else:
                gap_str = gap_str.strip('%')
                gap = float(gap_str) / 100

            data.append([time, gap])

    return pd.DataFrame(data, columns=['time', 'gap'])

def process_timedata(raw_df):
    """处理时间序列：插值、平滑、对齐到0-10800秒"""
    try:
        #raw_df['gap'] = raw_df['gap'].interpolate(method='linear', limit_direction='both')
        #raw_df = raw_df.drop_duplicates('time', keep='last')
        raw_df = raw_df.groupby('time')['gap'].mean().reset_index()
        full_time = np.arange(0, 10801)
        # 三次样条插值
        interp_gap = np.interp(full_time, raw_df['time'], raw_df['gap'])
        return pd.DataFrame({
            'time': full_time,
            'gap': interp_gap})
    except Exception as e:
        print(f"数据处理异常: {str(e)}")
        return pd.DataFrame()

def main(log_folder, plot_save):
    plt.figure(figsize=(12, 6))

    colors = ['#2A6EBC', '#FF6F61', '#4DB6AC', '#FFA726', '#957DAD', '#FF8A65', '#AED581', '#FFD54F', '#81C784', '#9E9E9E']
    all_combined_df = []
    q50_mins, q50_maxs = [], []
    for i, filename in enumerate(os.listdir(log_folder)):
        if filename.endswith(".log"):
            file_path = os.path.join(log_folder, filename)
            log_content = read_log_file(file_path)
            tables = extract_tables(log_content)[:5]
            processed_dfs = []
            for j, table in enumerate(tables, 1):
                print(f"处理表格 {j}/5")
                raw_df = parse_table(table)
                if raw_df.empty:
                    print(f"警告：表格{j}无有效数据")
                    continue
                # 数据处理
                aligned_df = process_timedata(raw_df)
                if not aligned_df.empty:
                    processed_dfs.append(aligned_df)

            combined_df = pd.concat(processed_dfs,ignore_index = False)
            all_combined_df.append(combined_df)
            combined_df.to_csv(os.path.join(log_folder, f"combined_gap_data" + filename+".csv"), index=False)

            # aggregated = combined_df.groupby('time')['gap'].agg(
            #     mean_gap='mean',
            #     std_gap='std',
            #     count='count'
            # ).reset_index()
            # aggregated = aggregated[aggregated['mean_gap'] <= 1]
            # # 计算90%置信区间
            # aggregated['ci'] = stats.t.ppf(0.95, aggregated['count'] - 1) * \
            #                    aggregated['std_gap'] / np.sqrt(aggregated['count'])
            aggregated = combined_df.groupby('time')['gap'].agg(
                q25=lambda x: np.percentile(x, 25),
                q50=lambda x: np.percentile(x, 50),  # 中位数
                q75=lambda x: np.percentile(x, 75)
            ).reset_index()
            aggregated = aggregated[aggregated['q50'] <= 1]
            # # 绘制平均曲线
            # color = colors[i % len(colors)]  # 循环使用颜色
            # plt.plot(aggregated['time'], aggregated['mean_gap'] * 100,  # 转换为百分比
            #          color=color, lw=2, label=f'File: {filename}')
            #
            # # 绘制置信区间带
            # plt.fill_between(
            #     aggregated['time'],
            #     (aggregated['mean_gap'] - aggregated['ci']) * 100,
            #     (aggregated['mean_gap'] + aggregated['ci']) * 100,
            #     color=color, alpha=0.3
            # )
            # 绘制中位数曲线（q50）
            q50_mins.append(aggregated['q50'].min())
            q50_maxs.append(aggregated['q50'].max())
            color = colors[i % len(colors)]
            plt.fill_between(
                aggregated['time'],
                aggregated['q25'] * 100,
                aggregated['q75'] * 100,
                color=color, alpha=0.6
            )
            plt.plot(aggregated['time'], aggregated['q50'] * 100,
                     color=color, lw=2, label=f'File: {filename}')

            # 绘制 [25%, 75%] 区间带

    if all_combined_df:
        total_df = pd.concat(all_combined_df, ignore_index=True)
        total_df.to_csv(os.path.join(log_folder, "all_combined.csv"), index=False)
    plt.title("Optimality Gap Evolution\n", fontsize=14)
    plt.xlabel("Optimization Time (seconds)", fontsize=12)
    plt.ylabel("Optimality Gap (%)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')

    plt.xlim(0, 10800)
    #plt.ylim(0, 100)
    min_q50 = min(q50_mins)
    max_q50 = max(q50_maxs)
    plt.ylim(max(0, min_q50 * 100 * 0.9), min(120, max_q50 * 100 * 1.3))
    #plt.ylim(max(0, aggregated['q50'].min() * 100 * 0.9), min(100, aggregated['q50'].max() * 100 * 1.1))  # 自动调整上界
    #plt.ylim(0, 50)
    plt.savefig(plot_save, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {plot_save}")
    plt.show()

if __name__ == "__main__":
    LOG_FOLDER = "D:/TLY/2025/operation althorigm/Shenzhen_L2O_MINLP/timegapplot/1000/gurobi"
    PLOT_SAVE = "D:/TLY/2025/operation althorigm/Shenzhen_L2O_MINLP/timegapplot/1000/gurobi/combined_plot.png"
    main(LOG_FOLDER, PLOT_SAVE)