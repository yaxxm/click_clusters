import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import seaborn as sns  # 添加seaborn用于更美观的分布可视化
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
import time
from tqdm import tqdm

# 设置matplotlib中文字体
import matplotlib.font_manager as fm

# 本地字体文件路径
LOCAL_FONT_PATH = "/data1/yammyjiang/81814/y_test/点击聚类/font/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"

# 尝试加载本地字体文件
try:
    if os.path.exists(LOCAL_FONT_PATH):
        # 添加本地字体到matplotlib字体管理器
        fm.fontManager.addfont(LOCAL_FONT_PATH)
        # 获取字体名称
        font_prop = fm.FontProperties(fname=LOCAL_FONT_PATH)
        font_name = font_prop.get_name()
        plt.rcParams['font.sans-serif'] = [font_name, 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
        print(f"[成功] 已加载本地字体: {font_name}")
    else:
        # 如果本地字体文件不存在，使用系统字体
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        if 'SimHei' in available_fonts:
            plt.rcParams['font.sans-serif'] = ['SimHei']
            print("[成功] 使用系统字体: SimHei")
        elif 'Microsoft YaHei' in available_fonts:
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
            print("[成功] 使用系统字体: Microsoft YaHei")
        else:
            plt.rcParams['font.sans-serif'] = ['sans-serif']
            print("[警告] 未找到中文字体，可能无法正确显示中文字符")
except Exception as e:
    print(f"[警告] 加载字体时出错: {e}，使用默认字体")
    plt.rcParams['font.sans-serif'] = ['sans-serif']

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# --- 参数设置 ---
DATA_PATH = "/data1/yammyjiang/81814/y_test/点击聚类/20250824.txt"
EPS = 0.01
MIN_SAMPLES = 10
OUTPUT_DIR = "/data1/yammyjiang/81814/y_test/点击聚类/output"

# --- 性能优化参数 ---
ENABLE_PARALLEL = True  # 启用并行处理
MAX_WORKERS = min(cpu_count(), 8)  # 最大工作进程数
GENERATE_PLOTS = True  # 是否生成可视化图像
USE_OPTIMIZED_DBSCAN = True  # 使用优化的DBSCAN算法
OPTIMIZED_PLOTS = True  # 是否使用优化的图像生成（更快但质量略低）

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 数据加载函数 ---
def load_data():
    if not os.path.exists(DATA_PATH):
        print(f"[表情] 错误: 文件 {DATA_PATH} 不存在")
        return None
    
    print(f"[表情] 正在加载数据文件: {DATA_PATH}")
    parsed_data = []
    
    with open(DATA_PATH, 'r') as f:
        for line in f:
            # 跳过空行
            if not line.strip():
                continue
                
            # 分割账号、坐标和点击次数
            parts = line.strip().split('|')
            if len(parts) < 3: 
                print(f"[表情] 跳过格式错误行: {line.strip()}")
                continue
            
            # 解析坐标
            coords = parts[1].split('#')
            if len(coords) < 2: 
                print(f"[表情] 跳过坐标格式错误行: {line.strip()}")
                continue
            
            try:
                # 解析账号、坐标和点击次数
                user_id = parts[0].strip()
                x = float(coords[0])
                y = float(coords[1])
                click_count = int(parts[2])
                
                # 存储数据点（包含点击次数）
                parsed_data.append({
                    'user_id': user_id,
                    'x': x,
                    'y': y,
                    'count': click_count
                })
                    
            except ValueError as e:
                print(f"[表情] 解析错误: {e} - 行内容: {line.strip()}")
                continue
    
    return pd.DataFrame(parsed_data)

# --- 优化的聚类算法 ---
def optimized_dbscan(coords, weights, eps, min_samples):
    """
    优化的DBSCAN实现，使用scikit-learn的高效算法
    """
    if USE_OPTIMIZED_DBSCAN:
        # 使用scikit-learn的优化DBSCAN
        # 为了模拟加权效果，我们复制权重较高的点
        weighted_coords = []
        original_indices = []
        
        for i, (coord, weight) in enumerate(zip(coords, weights)):
            # 根据权重复制点（最少1次，最多10次以避免过度膨胀）
            repeat_count = max(1, min(int(weight), 10))
            for _ in range(repeat_count):
                weighted_coords.append(coord)
                original_indices.append(i)
        
        weighted_coords = np.array(weighted_coords)
        
        # 使用scikit-learn的DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1)
        weighted_labels = dbscan.fit_predict(weighted_coords)
        
        # 将结果映射回原始点
        labels = np.full(len(coords), -1, dtype=np.intp)
        for i, orig_idx in enumerate(original_indices):
            if weighted_labels[i] != -1:
                if labels[orig_idx] == -1:
                    labels[orig_idx] = weighted_labels[i]
        
        return labels
    else:
        # 使用原始加权DBSCAN实现
        return weighted_dbscan_original(coords, weights, eps, min_samples)

def weighted_dbscan_original(coords, weights, eps, min_samples):
    """
    原始加权DBSCAN实现（保留作为备选）
    """
    n_samples = len(coords)
    labels = np.full(n_samples, -1, dtype=np.intp)
    
    neigh = NearestNeighbors(radius=eps)
    neigh.fit(coords)
    neighborhoods = neigh.radius_neighbors(coords, return_distance=False)
    
    core_samples = []
    for i in range(n_samples):
        weighted_count = weights[i]
        for neighbor in neighborhoods[i]:
            if neighbor != i:
                weighted_count += weights[neighbor]
        
        if weighted_count >= min_samples:
            core_samples.append(i)
    
    core_samples = np.asarray(core_samples)
    
    cluster_id = 0
    for i in core_samples:
        if labels[i] == -1:
            stack = [i]
            labels[i] = cluster_id
            
            while stack:
                current = stack.pop()
                
                for neighbor in neighborhoods[current]:
                    if labels[neighbor] == -1:
                        labels[neighbor] = cluster_id
                        
                        if neighbor in core_samples:
                            stack.append(neighbor)
            
            cluster_id += 1
    
    return labels

# --- 优化后的聚类可视化函数 ---
def plot_clusters(df, labels, user_id, filename, show_noise=True):
    # 创建图形，使用较小的DPI以加快生成速度
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    
    # 计算总点击次数
    total_clicks = df['count'].sum()
    
    # 预计算颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_clusters))) if n_clusters > 0 else []
    cluster_color_index = 0
    
    # 反转Y轴，使0.0在顶部，1.1在底部
    ax.invert_yaxis()
    
    # 批量处理数据以减少绘图调用次数
    for label in unique_labels:
        if label == -1 and show_noise:
            # 噪声点 (灰色)
            noise_data = df[df['cluster'] == label]
            if not noise_data.empty:
                noise_clicks = noise_data['count'].sum()
                ax.scatter(
                    noise_data['x'], noise_data['y'],
                    c='gray', alpha=0.4, s=20,  # 减小点的大小
                    label=f'噪声点 ({noise_clicks}次点击)',
                    rasterized=True  # 栅格化以加快渲染
                )
        elif label != -1:
            # 簇点
            cluster_data = df[df['cluster'] == label]
            if not cluster_data.empty:
                cluster_clicks = cluster_data['count'].sum()
                
                ax.scatter(
                    cluster_data['x'], cluster_data['y'],
                    color=colors[cluster_color_index], 
                    s=60,  # 减小点的大小
                    alpha=0.8,
                    label=f'簇 {label} ({cluster_clicks}次点击)',
                    rasterized=True  # 栅格化以加快渲染
                )
                
                # 簇中心（简单平均）
                center_x = cluster_data['x'].mean()
                center_y = cluster_data['y'].mean()
                
                ax.scatter(
                    center_x, center_y,
                    color='black', s=150,  # 减小中心点大小
                    marker='*', edgecolors='yellow',
                    zorder=10  # 确保中心点在最上层
                )
                
                # 簇标签
                ax.annotate(
                    f'簇{label}', (center_x, center_y),
                    xytext=(0, 10),  # 减小偏移
                    textcoords='offset points',
                    ha='center', fontsize=10,  # 减小字体
                    weight='bold'
                )
                cluster_color_index += 1
    
    # 图表设置（简化）
    noise_text = "含噪声" if show_noise else "去噪声"
    ax.set_title(f'用户 {user_id} 聚类 ({noise_text}) - {total_clicks}次点击, {n_clusters}簇', fontsize=12)
    ax.set_xlabel('X 坐标', fontsize=10)
    ax.set_ylabel('Y 坐标', fontsize=10)
    
    # 设置坐标轴范围
    ax.set_xlim(0.0, 1.1)
    ax.set_ylim(1.1, 0.0)
    
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc='upper right')  # 减小图例字体
    
    # 保存图像（优化参数）
    plt.savefig(filename, dpi=100, bbox_inches='tight', 
                facecolor='white', edgecolor='none',
                format='png')  # 移除optimize参数以兼容旧版matplotlib
    plt.close(fig)  # 明确关闭图形以释放内存

# --- 并行处理单个用户的函数 ---
def process_single_user(user_data):
    """
    处理单个用户的聚类和检测（用于并行处理）
    """
    user_id, user_df = user_data
    
    try:
        total_clicks = user_df['count'].sum()
        
        if total_clicks < MIN_SAMPLES:
            return {
                'user_id': user_id,
                'status': 'skipped',
                'reason': f'点击次数不足 ({total_clicks} < {MIN_SAMPLES})'
            }
        
        # 准备坐标和权重数据
        coords = user_df[['x', 'y']].values
        weights = user_df['count'].values
        
        # DBSCAN聚类
        labels = optimized_dbscan(coords, weights, eps=EPS, min_samples=MIN_SAMPLES)
        user_df_copy = user_df.copy()
        user_df_copy['cluster'] = labels
        
        # 检测是否为脚本
        detection_result = detect_script(user_df_copy, labels)
        detection_result['user_id'] = user_id
        detection_result['status'] = 'processed'
        detection_result['user_df'] = user_df_copy
        
        return detection_result
        
    except Exception as e:
        return {
            'user_id': user_id,
            'status': 'error',
            'error': str(e)
        }

# --- 脚本检测函数 ---
def detect_script(user_df, labels):
    """
    根据聚类结果检测是否为脚本用户
    规则: 
      1. 噪音点占比 <= 10%
      2. 最大的三个簇点击次数总占比 >= 80%
    """
    # 计算总点击次数
    total_clicks = user_df['count'].sum()
    
    # 计算噪音点比例
    noise_mask = (labels == -1)
    noise_clicks = user_df.loc[noise_mask, 'count'].sum()
    noise_ratio = noise_clicks / total_clicks if total_clicks > 0 else 0
    
    # 计算簇大小分布
    cluster_clicks = defaultdict(int)
    for idx, label in enumerate(labels):
        if label != -1:  # 排除噪音点
            cluster_clicks[label] += user_df.iloc[idx]['count']
    
    # 获取最大的三个簇
    sorted_clusters = sorted(cluster_clicks.items(), key=lambda x: x[1], reverse=True)
    top3_clusters = sorted_clusters[:3]
    top3_count = sum(count for _, count in top3_clusters)
    top3_ratio = top3_count / total_clicks if total_clicks > 0 else 0
    
    # 应用检测规则
    is_script = (noise_ratio <= 0.1) or (top3_ratio >= 0.8)
    
    # 返回检测结果和详细统计
    return {
        'is_script': is_script,
        'total_clicks': total_clicks,
        'noise_clicks': noise_clicks,
        'noise_ratio': noise_ratio,
        'cluster_count': len(cluster_clicks),
        'top3_ratio': top3_ratio,
        'top3_clusters': [label for label, _ in top3_clusters]
    }

# --- 新增功能：分布可视化 ---
def plot_distribution(results_df, distribution_type, output_dir):
    """
    绘制噪音点比例或前三个簇占比的分布直方图和箱线图
    
    参数:
        results_df: 包含所有用户结果的数据框
        distribution_type: 'noise_ratio' 或 'top3_ratio'
        output_dir: 输出目录路径
    """
    # 确保结果不为空
    if len(results_df) == 0:
        print(f"[表情] 没有用户数据可用于绘制 {distribution_type} 分布")
        return
    
    # 根据类型设置标题和文件名
    if distribution_type == 'noise_ratio':
        title = "用户噪音点比例分布"
        x_label = "噪音点比例"
        filename = "noise_ratio_distribution.png"
        desc = "噪音点比例 = 噪音点点击次数 / 总点击次数"
    elif distribution_type == 'top3_ratio':
        title = "用户前三个簇占比分布"
        x_label = "前三个簇占比"
        filename = "top3_ratio_distribution.png"
        desc = "前三个簇占比 = (前三个最大簇的点击次数之和) / 总点击次数"
    else:
        raise ValueError("无效的分布类型")
    
    plt.figure(figsize=(15, 6))
    
    # 绘制直方图 (左上)
    plt.subplot(1, 2, 1)
    sns.histplot(results_df[distribution_type], bins=20, kde=True, color='skyblue')
    plt.axvline(results_df[distribution_type].median(), color='red', linestyle='dashed', linewidth=1)
    plt.text(results_df[distribution_type].median(), plt.ylim()[1]*0.9, 
             f'中位数: {results_df[distribution_type].median():.2f}', color='red')
    plt.title(f"{title} (直方图)", fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel("用户数量", fontsize=12)
    plt.grid(alpha=0.3)
    
    # 绘制箱线图 (右上)
    plt.subplot(1, 2, 2)
    sns.boxplot(x=results_df[distribution_type], color='lightgreen')
    plt.title(f"{title} (箱线图)", fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    
    plt.tight_layout()
    
    # 添加整体描述 (底部)
    plt.figtext(0.5, 0.01, desc, ha='center', fontsize=10, color='gray')
    
    # 保存图像
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[表情] {title} 分布图已保存至: {output_path}")
    plt.close()
    
    # 返回关键统计值
    stats = {
        'mean': results_df[distribution_type].mean(),
        'median': results_df[distribution_type].median(),
        'std_dev': results_df[distribution_type].std(),
        'min': results_df[distribution_type].min(),
        'max': results_df[distribution_type].max(),
        'q1': np.percentile(results_df[distribution_type], 25),
        'q3': np.percentile(results_df[distribution_type], 75)
    }
    
    return stats

# --- 新增功能：联合分布分析 ---
def plot_joint_distribution(results_df, output_dir):
    """绘制噪音点比例与前三个簇占比的联合分布图"""
    if len(results_df) == 0:
        print("[表情] 没有用户数据可用于绘制联合分布图")
        return
    
    plt.figure(figsize=(10, 8))
    sns.jointplot(
        x='noise_ratio', 
        y='top3_ratio', 
        data=results_df,
        kind='hex',  # 六边形箱图
        gridsize=20,  # 网格大小
        cmap='Blues', 
        marginal_kws=dict(bins=20, fill=True),
        height=10
    )
    
    plt.suptitle("噪音比例与前三个簇占比联合分布", y=0.92, fontsize=16)
    plt.text(0.5, -0.1, 
             "红色虚线表示脚本检测阈值 (噪音≤10%且前三簇≥80%)", 
             ha='center', transform=plt.gca().transAxes,
             color='red')
    
    # 添加阈值线
    plt.axvline(x=0.1, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.axhline(y=0.8, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # 添加脚本用户区域说明
    plt.text(0.02, 0.85, "正常用户区域", fontsize=10, color='darkblue')
    plt.text(0.02, 0.95, "脚本用户区域", fontsize=10, color='darkred')
    
    # 保存图像
    output_path = os.path.join(output_dir, "joint_distribution.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[表情] 联合分布图已保存至: {output_path}")
    plt.close()

# --- 优化的主分析函数 ---
def analyze_all_users():
    start_time = time.time()
    
    # 加载数据
    print("[表情] 开始加载数据...")
    df = load_data()
    if df is None:
        return
    
    load_time = time.time() - start_time
    print(f"[表情] 数据加载完成，耗时: {load_time:.2f}秒")
    
    # 获取所有用户ID
    user_ids = df['user_id'].unique()
    print(f"[表情] 找到 {len(user_ids)} 个用户")
    
    # 创建结果目录
    results_dir = os.path.join(OUTPUT_DIR, "Results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 准备用户数据
    user_data_list = [(user_id, df[df['user_id'] == user_id].copy()) for user_id in user_ids]
    
    # 并行或串行处理
    processing_start = time.time()
    
    if ENABLE_PARALLEL and len(user_ids) > 1:
        print(f"[表情] 启用并行处理，使用 {MAX_WORKERS} 个进程")
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有任务
            futures = [executor.submit(process_single_user, user_data) for user_data in user_data_list]
            
            # 收集结果（带进度条）
            all_results = []
            with tqdm(total=len(futures), desc="[表情] 处理用户", unit="用户") as pbar:
                for i, future in enumerate(futures):
                    try:
                        result = future.result(timeout=300)  # 5分钟超时
                        all_results.append(result)
                        
                        if result['status'] == 'processed':
                            pbar.set_postfix({"当前": f"{result['user_id']} ({'脚本' if result['is_script'] else '正常'})"})
                        elif result['status'] == 'skipped':
                            pbar.set_postfix({"当前": f"{result['user_id']} (跳过)"})
                        else:
                            pbar.set_postfix({"当前": f"{result['user_id']} (错误)"})
                            
                        pbar.update(1)
                            
                    except Exception as e:
                        print(f"\n[表情] 处理用户时发生错误: {str(e)}")
                        pbar.update(1)
                        continue
    else:
        print("[表情] 使用串行处理")
        all_results = []
        with tqdm(user_data_list, desc="[表情] 处理用户", unit="用户") as pbar:
            for user_data in pbar:
                result = process_single_user(user_data)
                all_results.append(result)
                
                if result['status'] == 'processed':
                    pbar.set_postfix({"当前": f"{result['user_id']} ({'脚本' if result['is_script'] else '正常'})"})
                elif result['status'] == 'skipped':
                    pbar.set_postfix({"当前": f"{result['user_id']} (跳过)"})
    
    processing_time = time.time() - processing_start
    print(f"[表情] 用户处理完成，耗时: {processing_time:.2f}秒")
    
    # 处理结果和生成输出
    output_start = time.time()
    
    # 分离成功处理的结果
    processed_results = [r for r in all_results if r['status'] == 'processed']
    results = []
    script_users = []
    plot_tasks = []  # 存储需要生成图像的任务
    
    # 第一步：优先生成所有结果文件
    print("[表情] 正在保存结果文件...")
    with tqdm(all_results, desc="[表情] 保存结果", unit="文件") as pbar:
        for result in pbar:
            if result['status'] != 'processed':
                pbar.set_postfix({"状态": "跳过"})
                continue
                
            # 提取基本检测结果
            detection_result = {
                'user_id': result['user_id'],
                'is_script': result['is_script'],
                'total_clicks': result['total_clicks'],
                'noise_clicks': result['noise_clicks'],
                'noise_ratio': result['noise_ratio'],
                'cluster_count': result['cluster_count'],
                'top3_ratio': result['top3_ratio'],
                'top3_clusters': result['top3_clusters']
            }
            results.append(detection_result)
            
            user_id = result['user_id']
            user_df = result['user_df']
            labels = user_df['cluster'].values
            
            # 确定输出目录
            if result['is_script']:
                script_users.append(user_id)
                user_dir = os.path.join(OUTPUT_DIR, "ScriptUsers", user_id)
            else:
                user_dir = os.path.join(OUTPUT_DIR, "NormalUsers", user_id)
            
            os.makedirs(user_dir, exist_ok=True)
            
            # 保存聚类结果CSV
            csv_path = os.path.join(user_dir, f"{user_id}_clusters.csv")
            user_df.to_csv(csv_path, index=False)
            
            # 如果需要生成图像，添加到任务列表
            if GENERATE_PLOTS:
                plot_tasks.append({
                    'user_df': user_df,
                    'labels': labels,
                    'user_id': user_id,
                    'user_dir': user_dir
                })
            
            pbar.set_postfix({"状态": "已保存"})
    
    # 第二步：批量生成图像（如果启用）
    if GENERATE_PLOTS and plot_tasks:
         plot_start = time.time()
         print(f"\n[表情] 开始生成 {len(plot_tasks)} 个用户的可视化图像...")
         print(f"[表情] 图像生成慢的原因分析:")
         print(f"  - matplotlib初始化开销: 每个图形需要创建新的画布和坐标系")
         print(f"  - 数据点渲染: 大量散点图绘制需要逐点计算位置和颜色")
         print(f"  - 图像保存I/O: PNG压缩和磁盘写入操作")
         print(f"  - 内存管理: 图形对象创建和销毁的内存分配")
         if OPTIMIZED_PLOTS:
             print(f"[表情] 已启用优化措施: 降低DPI、减小点大小、栅格化渲染、优化PNG压缩")
         
         with tqdm(plot_tasks, desc="[表情] 生成图像", unit="图像") as pbar:
             for task in pbar:
                 try:
                     pbar.set_postfix({"用户": task['user_id']})
                     
                     # 生成带噪音点的图像
                     plot_clusters(task['user_df'], task['labels'], task['user_id'], 
                                  os.path.join(task['user_dir'], f"{task['user_id']}_with_noise.png"), 
                                  show_noise=True)
                     
                     # 生成不带噪音点的图像
                     plot_clusters(task['user_df'], task['labels'], task['user_id'], 
                                  os.path.join(task['user_dir'], f"{task['user_id']}_without_noise.png"), 
                                  show_noise=False)
                                  
                 except Exception as e:
                     print(f"\n[表情] 生成用户 {task['user_id']} 图像时出错: {str(e)}")
                     continue
         
         plot_time = time.time() - plot_start
         avg_plot_time = plot_time / len(plot_tasks) if plot_tasks else 0
         print(f"\n[表情] 图像生成完成，总耗时: {plot_time:.2f}秒")
         print(f"[表情] 平均每个用户图像生成时间: {avg_plot_time:.3f}秒")
         if avg_plot_time > 0.5:
             print(f"[表情] 提示: 如需更快速度，可设置 GENERATE_PLOTS = False 跳过图像生成")
    
    output_time = time.time() - output_start
    print(f"[表情] 输出生成完成，耗时: {output_time:.2f}秒")
    
    # 保存总体结果
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        # 添加标记用户类型
        results_df['user_type'] = results_df['is_script'].map({True: '脚本用户', False: '正常用户'})
        results_csv = os.path.join(results_dir, "detection_results.csv")
        results_df.to_csv(results_csv, index=False)
        
        # --- 新增分析: 用户分布可视化 ---
        if GENERATE_PLOTS:
            try:
                # 分析噪音点比例分布
                noise_stats = plot_distribution(results_df, 'noise_ratio', results_dir)
                print("\n[表情] 噪音点比例分布统计:")
                print(f"  平均值: {noise_stats['mean']:.4f} | 中位数: {noise_stats['median']:.4f}")
                print(f"  标准差: {noise_stats['std_dev']:.4f} | 范围: [{noise_stats['min']:.4f}, {noise_stats['max']:.4f}]")
                print(f"  四分位数: Q1={noise_stats['q1']:.4f} | Q3={noise_stats['q3']:.4f}")
                
                # 分析前三个簇占比分布
                top3_stats = plot_distribution(results_df, 'top3_ratio', results_dir)
                print("\n[表情] 前三个簇占比分布统计:")
                print(f"  平均值: {top3_stats['mean']:.4f} | 中位数: {top3_stats['median']:.4f}")
                print(f"  标准差: {top3_stats['std_dev']:.4f} | 范围: [{top3_stats['min']:.4f}, {top3_stats['max']:.4f}]")
                print(f"  四分位数: Q1={top3_stats['q1']:.4f} | Q3={top3_stats['q3']:.4f}")
                
                # 分析两者联合分布
                plot_joint_distribution(results_df, results_dir)
                
            except Exception as e:
                print(f"[表情] 分布分析出错: {str(e)}")
    
    # 保存脚本用户列表
    script_csv = os.path.join(results_dir, "script_users.csv")
    pd.DataFrame({'user_id': script_users}).to_csv(script_csv, index=False)
    
    # 计算总耗时
    total_time = time.time() - start_time
    
    # 打印汇总结果
    print("\n" + "="*50)
    print(f"[表情] 分析完成! 共处理 {len(user_ids)} 个用户")
    print(f"[表情] 总耗时: {total_time:.2f}秒 (加载: {load_time:.2f}s, 处理: {processing_time:.2f}s, 输出: {output_time:.2f}s)")
    if not results_df.empty:
        print(f"[表情] 检测到 {len(script_users)} 个脚本用户 (占 {len(script_users)/len(results_df):.1%})")
        print(f"[表情] 检测到 {len(results_df)-len(script_users)} 个正常用户")
        print(f"[表情] 用户噪音比例平均: {results_df['noise_ratio'].mean():.2%}")
        print(f"[表情] 前三个簇占比平均: {results_df['top3_ratio'].mean():.2%}")
    else:
        print("[表情] 没有符合条件的用户数据")
    print(f"[表情] 详细结果保存至: {results_csv if 'results_csv' in locals() else '无'}")
    print(f"[表情] 脚本用户列表保存至: {script_csv}")
    print("="*50)

# --- 性能测试函数 ---
def performance_test():
    """
    简单的性能测试，比较优化前后的效果
    """
    print("[表情] 开始性能测试...")
    
    # 加载少量数据进行测试
    df = load_data()
    if df is None:
        return
    
    # 选择前10个用户进行测试
    test_users = df['user_id'].unique()[:10]
    test_df = df[df['user_id'].isin(test_users)]
    
    print(f"[表情] 使用 {len(test_users)} 个用户进行性能测试")
    
    # 测试串行处理
    print("\n--- 串行处理测试 ---")
    start_time = time.time()
    
    for user_id in test_users:
        user_df = test_df[test_df['user_id'] == user_id].copy()
        if user_df['count'].sum() >= MIN_SAMPLES:
            coords = user_df[['x', 'y']].values
            weights = user_df['count'].values
            _ = optimized_dbscan(coords, weights, EPS, MIN_SAMPLES)
    
    serial_time = time.time() - start_time
    print(f"串行处理耗时: {serial_time:.2f}秒")
    
    # 测试并行处理（如果启用）
    if ENABLE_PARALLEL and len(test_users) > 1:
        print("\n--- 并行处理测试 ---")
        start_time = time.time()
        
        user_data_list = [(user_id, test_df[test_df['user_id'] == user_id].copy()) for user_id in test_users]
        
        with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(test_users))) as executor:
            futures = [executor.submit(process_single_user, user_data) for user_data in user_data_list]
            results = [future.result() for future in futures]
        
        parallel_time = time.time() - start_time
        print(f"并行处理耗时: {parallel_time:.2f}秒")
        
        if serial_time > 0:
            speedup = serial_time / parallel_time
            print(f"加速比: {speedup:.2f}x")
    
    print("[表情] 性能测试完成!\n")

if __name__ == "__main__":
    print("="*60)
    print("[表情] 点击聚类分析系统 - 优化版本")
    print("="*60)
    
    print(f"\n[表情] 当前配置参数:")
    print(f"  - 数据文件: {DATA_PATH}")
    print(f"  - 输出目录: {OUTPUT_DIR}")
    print(f"  - 并行处理: {'启用' if ENABLE_PARALLEL else '禁用'}")
    print(f"  - 最大工作进程: {MAX_WORKERS}")
    print(f"  - 生成可视化: {'启用' if GENERATE_PLOTS else '禁用'}")
    print(f"  - 优化DBSCAN: {'启用' if USE_OPTIMIZED_DBSCAN else '禁用'}")
    print(f"  - 优化图像生成: {'启用' if OPTIMIZED_PLOTS else '禁用'} (启用时图像生成更快但质量略低)")
    print(f"  - EPS参数: {EPS}")
    print(f"  - MIN_SAMPLES参数: {MIN_SAMPLES}")
    
    # 可选的性能测试
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        performance_test()
    
    print(f"\n[表情] 开始完整分析...")
    analyze_all_users()
    print("\n[表情] 所有分析任务完成!")
    print("="*60)
