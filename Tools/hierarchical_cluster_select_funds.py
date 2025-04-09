# -*- encoding: utf-8 -*-
"""
@File: hierarchical_cluster_select_funds.py
@Modify Time: 2025/4/8 20:51       
@Author: Kevin-Chen
@Descriptions: 层次聚类（hierarchical clustering）在基金间的相关性上进行分群，并在每个聚类中保留 x 只“最具有代表性”（和本簇其他成员的平均距离最小）以及若干只历史收益/夏普比最高的基金
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import plotly.express as px

pd.set_option('display.max_columns', 1000)  # 显示字段的数量
pd.set_option('display.width', 1000)  # 表格不分段显示


def compute_sharpe_ratio(returns: pd.Series, freq: int = 252) -> float:
    """
    计算某只基金的夏普比率 (年化).
    returns: 该基金的日度收益率序列 (或其他等频数据)
    freq:    年化频率(一般 252 个交易日/年)
    """
    mean_ret = returns.mean() * freq
    std_ret = returns.std() * np.sqrt(freq)
    if std_ret == 0:
        return 0.0
    return mean_ret / std_ret


# def hierarchical_cluster_select_funds(
#         log_return_df: pd.DataFrame,
#         n_clusters: int = 5,
#         x_rep: int = 1,
#         x_sharpe: int = 1
# ):
#     """
#     对基金的 log_return 矩阵做层次聚类, 并在每个簇中:
#       - 保留 x_rep 只"最具代表性"的基金 (对本簇平均距离最小)
#       - 保留 x_sharpe 只夏普比率最高的基金
#     返回:
#       - cluster_labels: 每只基金所属的簇标签
#       - selected_funds: 最终被保留下来的基金列表
#       - Z: 层次聚类的linkage结果, 用于绘图
#     """
#     # ========== 第一步: 计算相关系数 & 距离矩阵 ==========
#     corr_matrix = log_return_df.corr()  # 基金两两之间的相关系数
#     dist_matrix = 1 - corr_matrix  # 以 (1 - 相关系数) 作为距离度量
#     # 将矩阵压平(condensed)供 hierarchy.linkage 使用
#     dist_condensed = squareform(dist_matrix.values[np.triu_indices(len(dist_matrix), k=1)])
#
#     # ========== 第二步: 执行层次聚类 ==========
#     # method 可以是 "complete", "average", "ward" 等, 示例用 "complete"
#     Z = linkage(dist_condensed, method='complete')
#
#     # ========== 第三步: 划分为 n_clusters 个簇 ==========
#     cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')
#
#     # 基金列表
#     fund_list = corr_matrix.columns.tolist()
#
#     # ========== 第四步: 计算夏普比率, 并按簇筛选基金 ==========
#     # 先把每只基金的夏普比率算好
#     sharpe_dict = {}
#     for fund in fund_list:
#         # 取该基金对应的时间序列
#         series = log_return_df[fund].dropna()
#         sharpe_dict[fund] = compute_sharpe_ratio(series)
#
#     # 分簇处理 -> 对每个簇内进行筛选
#     df_cluster = pd.DataFrame({'fund': fund_list, 'cluster': cluster_labels})
#     selected_funds = []
#
#     # 为了计算“最具代表性”，我们需要在簇内比较距离(或相关性)
#     # 这里用平均距离(对同簇内所有基金)的加总或平均来衡量
#     for c_id, sub_df in df_cluster.groupby('cluster'):
#         funds_in_cluster = sub_df['fund'].tolist()
#
#         if len(funds_in_cluster) == 1:
#             # 簇里只有 1 只基金, 直接保留
#             selected_funds.append(funds_in_cluster[0])
#             continue
#
#         # 取出子矩阵
#         sub_corr = corr_matrix.loc[funds_in_cluster, funds_in_cluster]
#         sub_dist = 1 - sub_corr  # 距离
#
#         # 计算“平均距离”
#         avg_dist = sub_dist.mean(axis=1)  # 每只基金相对于本簇其他成员的平均距离
#         # 按平均距离从小到大排序(距离小 => 越“有代表性”)
#         avg_dist_sorted = avg_dist.sort_values(ascending=True)
#
#         # 先选 x_rep 只代表性最强的
#         rep_candidates = list(avg_dist_sorted.index[:x_rep])
#
#         # 再选 x_sharpe 只夏普比率最高的
#         # 但我们只在本簇内排名, 这样兼顾相似度和收益情况
#         cluster_sharpes = {f: sharpe_dict[f] for f in funds_in_cluster}
#         sorted_by_sharpe = sorted(cluster_sharpes.items(), key=lambda x: x[1], reverse=True)
#         sharpe_candidates = [item[0] for item in sorted_by_sharpe[:x_sharpe]]
#
#         # 汇总(去重)
#         final_select = list(set(rep_candidates + sharpe_candidates))
#         selected_funds.extend(final_select)
#
#     return cluster_labels, selected_funds, Z
#
#
# def plot_dendrogram(Z, fund_labels, selected_funds, title="Hierarchical Clustering Dendrogram"):
#     """
#     绘制层次聚类的树状图(dendrogram), 并在树叶上对被选中的fund进行高亮(此处用名字后缀表示).
#     注: 为了简单, 直接给选中的基金加个星标(*), 也可用更多样的可视化方式.
#     """
#     # 对被保留的基金在标签上做个标记
#     labeled_funds = []
#     for f in fund_labels:
#         if f in selected_funds:
#             labeled_funds.append(f + "*")
#         else:
#             labeled_funds.append(f)
#
#     plt.figure(figsize=(10, 6))
#     dendrogram(
#         Z,
#         labels=labeled_funds,
#         leaf_rotation=90,  # 旋转标签
#     )
#     plt.title(title)
#     plt.tight_layout()
#     plt.show()


def hierarchical_cluster_select_funds(
        log_return_df: pd.DataFrame,
        n_clusters: int = 5,
        x_rep: int = 1,
        x_sharpe: int = 1
):
    """
    对基金的 log_return 矩阵做层次聚类, 并在每个簇中:
      - 保留 x_rep 只"最具代表性"的基金 (对本簇平均距离最小)
      - 保留 x_sharpe 只夏普比率最高的基金
    返回:
      - cluster_labels: 每只基金所属的簇标签(pd.Series, index为基金code, 值为cluster_id)
      - selected_funds: 最终被保留下来的基金列表
      - Z: 层次聚类的 linkage 结果
      - dist_matrix: (1 - corr)的距离矩阵, 用于后面做 MDS
    """
    # 1. 计算相关系数 & 距离矩阵
    corr_matrix = log_return_df.corr()
    dist_matrix = 1 - corr_matrix
    # 压平供层次聚类
    dist_condensed = squareform(dist_matrix.values[np.triu_indices(len(dist_matrix), k=1)])

    # 2. 层次聚类 (method可"complete"/"average"/"ward"等)
    Z = linkage(dist_condensed, method='complete')

    # 3. 分簇
    cluster_ids = fcluster(Z, t=n_clusters, criterion='maxclust')
    fund_list = corr_matrix.columns.tolist()
    cluster_labels = pd.Series(cluster_ids, index=fund_list)

    # 4. 计算夏普比率
    sharpe_dict = {}
    for fund in fund_list:
        series = log_return_df[fund].dropna()
        mean_ret = series.mean() * 252  # 年化
        std_ret = series.std() * np.sqrt(252)
        sharpe_dict[fund] = (mean_ret / std_ret) if std_ret != 0 else 0.0

    # 5. 在每个簇内保留“代表性基金”和“夏普高”的基金
    selected_funds = []
    for c_id in np.unique(cluster_ids):
        funds_in_cluster = cluster_labels[cluster_labels == c_id].index.tolist()
        if len(funds_in_cluster) == 1:
            selected_funds.append(funds_in_cluster[0])
            continue
        # 子距离矩阵
        sub_dist = dist_matrix.loc[funds_in_cluster, funds_in_cluster]
        avg_dist = sub_dist.mean(axis=1)  # 平均距离
        avg_dist_sorted = avg_dist.sort_values(ascending=True)

        # x_rep只最具代表性(对本簇距离最小)
        rep_candidates = list(avg_dist_sorted.index[:x_rep])

        # x_sharpe只高夏普
        cluster_sharpes = {f: sharpe_dict[f] for f in funds_in_cluster}
        sorted_by_sharpe = sorted(cluster_sharpes.items(), key=lambda x: x[1], reverse=True)
        sharpe_candidates = [item[0] for item in sorted_by_sharpe[:x_sharpe]]

        final_select = list(set(rep_candidates + sharpe_candidates))
        selected_funds.extend(final_select)

    return cluster_labels, selected_funds, Z, dist_matrix


def mds_plot_interactive(
        dist_matrix: pd.DataFrame,
        cluster_labels: pd.Series,
        selected_funds: list,
        title="MDS Scatter Plot"
):
    """
    使用 MDS 将距离矩阵映射到 2D, 并用 Plotly 画可交互散点图.
      - dist_matrix:   (1 - corr)的DataFrame, 行列是基金
      - cluster_labels: pd.Series, index是基金, 值是簇编号
      - selected_funds: 被保留基金列表
    """
    from sklearn.manifold import MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    # 注意: MDS需要传入numpy array类型
    embedding = mds.fit_transform(dist_matrix.values)

    # 构造一个DataFrame来存放2D坐标和其他信息
    funds = dist_matrix.index.tolist()
    df_plot = pd.DataFrame({
        'fund': funds,
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'cluster': cluster_labels.loc[funds].values
    })
    # 加一个boolean字段"selected"表示是否被保留
    df_plot['selected'] = df_plot['fund'].isin(selected_funds)

    # Plotly可视化
    # color不同簇, symbol/marker区分 selected 与否, hover_data显示基金名
    fig = px.scatter(
        df_plot,
        x='x', y='y',
        color='cluster',  # 按簇上色
        symbol='selected',  # selected与否, 用不同形状
        hover_data=['fund'],  # 鼠标悬浮显示
        title=title
    )
    fig.update_layout(legend_title_text='Cluster ID')
    fig.show()


if __name__ == "__main__":
    # 假设你已有一个 log_return_df: 行是日期, 列是基金代码, 值是该日对数收益
    from Tools.factor_factory import data_prepare

    open_df, high_df, low_df, close_df, change_df, _, vol_df, amount_df, log_return_df, etf_info = data_prepare()

    # (2) 层次聚类并选出代表性基金 (每簇2只最具代表性 + 1只高夏普)
    cluster_labels, selected_funds, Z, dist_matrix = hierarchical_cluster_select_funds(
        log_return_df,
        n_clusters=10,
        x_rep=2,
        x_sharpe=2
    )

    print("各基金所属的簇:")
    print(cluster_labels)
    print("\n最终保留的基金:")
    print(selected_funds)

    etf_info = etf_info[etf_info['ts_code'].isin(selected_funds)]
    print(etf_info)

    # # (3) 使用 MDS + Plotly 可交互散点图
    # mds_plot_interactive(
    #     dist_matrix=dist_matrix,
    #     cluster_labels=cluster_labels,
    #     selected_funds=selected_funds,
    #     title="MDS Visualization of Fund Clusters"
    # )
