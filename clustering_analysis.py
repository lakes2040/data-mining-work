"""
聚类分析模块
功能：K-Means聚类、最佳k值选择、聚类结果可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class LoanClusteringAnalyzer:
    """贷款数据聚类分析器"""
    
    def __init__(self, X, y=None):
        """
        初始化聚类分析器
        
        Args:
            X (pd.DataFrame): 特征数据
            y (pd.Series): 目标变量（可选）
        """
        self.X = X
        self.y = y
        self.kmeans_model = None
        self.optimal_k = None
        self.cluster_labels = None
        self.pca_model = None
        self.X_pca = None
        
    def find_optimal_k(self, max_k=10):
        """
        使用多种方法确定最佳k值
        
        Args:
            max_k (int): 最大k值
            
        Returns:
            dict: 包含各种指标的结果
        """
        print("正在寻找最佳k值...")
        
        k_range = range(2, max_k + 1)
        inertias = []
        silhouette_scores = []
        davies_bouldin_scores = []
        
        for k in k_range:
            # K-Means聚类
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.X)
            
            # 计算各种指标
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.X, cluster_labels))
            davies_bouldin_scores.append(davies_bouldin_score(self.X, cluster_labels))
        
        # 找到最佳k值
        # Elbow方法：选择inertia下降幅度最大的点
        elbow_k = self._find_elbow_point(k_range, inertias)
        
        # Silhouette方法：选择轮廓系数最高的k
        silhouette_k = k_range[np.argmax(silhouette_scores)]
        
        # Davies-Bouldin方法：选择DB指数最低的k
        db_k = k_range[np.argmin(davies_bouldin_scores)]
        
        results = {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'davies_bouldin_scores': davies_bouldin_scores,
            'elbow_k': elbow_k,
            'silhouette_k': silhouette_k,
            'db_k': db_k
        }
        
        # 综合选择最佳k（优先考虑轮廓系数）
        self.optimal_k = silhouette_k
        
        print(f"Elbow方法推荐k值: {elbow_k}")
        print(f"Silhouette方法推荐k值: {silhouette_k}")
        print(f"Davies-Bouldin方法推荐k值: {db_k}")
        print(f"最终选择k值: {self.optimal_k}")
        
        return results
    
    def _find_elbow_point(self, k_range, inertias):
        """找到肘部点"""
        # 计算二阶导数来找到肘部点
        second_derivatives = []
        for i in range(1, len(inertias) - 1):
            second_deriv = inertias[i-1] - 2*inertias[i] + inertias[i+1]
            second_derivatives.append(second_deriv)
        
        # 找到二阶导数最大的点
        elbow_idx = np.argmax(second_derivatives) + 1
        return k_range[elbow_idx]
    
    def plot_k_selection(self, results):
        """绘制k值选择图表"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('K值选择分析', fontsize=16)
        
        k_range = results['k_range']
        
        # 1. Elbow方法
        axes[0].plot(k_range, results['inertias'], 'bo-')
        axes[0].axvline(x=results['elbow_k'], color='red', linestyle='--', 
                       label=f'Elbow k={results["elbow_k"]}')
        axes[0].set_title('Elbow方法')
        axes[0].set_xlabel('聚类数量 (k)')
        axes[0].set_ylabel('Inertia')
        axes[0].legend()
        axes[0].grid(True)
        
        # 2. Silhouette方法
        axes[1].plot(k_range, results['silhouette_scores'], 'go-')
        axes[1].axvline(x=results['silhouette_k'], color='red', linestyle='--',
                       label=f'Silhouette k={results["silhouette_k"]}')
        axes[1].set_title('Silhouette方法')
        axes[1].set_xlabel('聚类数量 (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].legend()
        axes[1].grid(True)
        
        # 3. Davies-Bouldin方法
        axes[2].plot(k_range, results['davies_bouldin_scores'], 'ro-')
        axes[2].axvline(x=results['db_k'], color='red', linestyle='--',
                       label=f'DB k={results["db_k"]}')
        axes[2].set_title('Davies-Bouldin方法')
        axes[2].set_xlabel('聚类数量 (k)')
        axes[2].set_ylabel('Davies-Bouldin Score')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def perform_clustering(self, k=None):
        """
        执行K-Means聚类
        
        Args:
            k (int): 聚类数量，如果为None则使用optimal_k
        """
        if k is None:
            k = self.optimal_k
        
        print(f"执行K-Means聚类，k={k}")
        
        # 执行聚类
        self.kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans_model.fit_predict(self.X)
        
        # 计算聚类质量指标
        silhouette_avg = silhouette_score(self.X, self.cluster_labels)
        db_score = davies_bouldin_score(self.X, self.cluster_labels)
        
        print(f"聚类完成！")
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        print(f"Davies-Bouldin Score: {db_score:.3f}")
        
        # 打印每个聚类的样本数量
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        print(f"各聚类样本数量: {dict(zip(unique, counts))}")
        
        return self.cluster_labels
    
    def analyze_clusters(self):
        """分析聚类结果"""
        if self.cluster_labels is None:
            print("请先执行聚类！")
            return
        
        print("\n=== 聚类分析 ===")
        
        # 创建包含聚类标签的数据框
        cluster_data = self.X.copy()
        cluster_data['cluster'] = self.cluster_labels
        
        if self.y is not None:
            cluster_data['loan_approved'] = self.y
        
        # 分析每个聚类的特征
        cluster_summary = cluster_data.groupby('cluster').agg({
            'income': ['mean', 'std'],
            'credit_score': ['mean', 'std'],
            'loan_amount': ['mean', 'std'],
            'years_employed': ['mean', 'std'],
            'points': ['mean', 'std']
        }).round(2)
        
        print("各聚类特征统计:")
        print(cluster_summary)
        
        # 分析聚类与贷款审批的关系
        if self.y is not None:
            print("\n各聚类的贷款审批率:")
            approval_rate = cluster_data.groupby('cluster')['loan_approved'].agg(['count', 'sum', 'mean'])
            approval_rate.columns = ['总数', '批准数', '批准率']
            approval_rate['批准率'] = approval_rate['批准率'].round(3)
            print(approval_rate)
        
        return cluster_data
    
    def visualize_clusters(self):
        """可视化聚类结果"""
        if self.cluster_labels is None:
            print("请先执行聚类！")
            return
        
        # 使用PCA降维到2D
        self.pca_model = PCA(n_components=2, random_state=42)
        self.X_pca = self.pca_model.fit_transform(self.X)
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('聚类结果可视化', fontsize=16)
        
        # 1. PCA 2D散点图
        scatter = axes[0,0].scatter(self.X_pca[:, 0], self.X_pca[:, 1], 
                                  c=self.cluster_labels, cmap='viridis', alpha=0.7)
        axes[0,0].set_title('PCA 2D聚类结果')
        axes[0,0].set_xlabel(f'PC1 (解释方差: {self.pca_model.explained_variance_ratio_[0]:.2%})')
        axes[0,0].set_ylabel(f'PC2 (解释方差: {self.pca_model.explained_variance_ratio_[1]:.2%})')
        plt.colorbar(scatter, ax=axes[0,0])
        
        # 2. 聚类中心
        cluster_centers_pca = self.pca_model.transform(self.kmeans_model.cluster_centers_)
        axes[0,1].scatter(self.X_pca[:, 0], self.X_pca[:, 1], 
                         c=self.cluster_labels, cmap='viridis', alpha=0.3)
        axes[0,1].scatter(cluster_centers_pca[:, 0], cluster_centers_pca[:, 1], 
                         c='red', marker='x', s=200, linewidths=3, label='聚类中心')
        axes[0,1].set_title('聚类中心')
        axes[0,1].set_xlabel('PC1')
        axes[0,1].set_ylabel('PC2')
        axes[0,1].legend()
        
        # 3. 聚类与贷款审批关系
        if self.y is not None:
            colors = ['red' if not approved else 'green' for approved in self.y]
            axes[1,0].scatter(self.X_pca[:, 0], self.X_pca[:, 1], 
                             c=colors, alpha=0.7)
            axes[1,0].set_title('贷款审批分布 (红色=拒绝, 绿色=批准)')
            axes[1,0].set_xlabel('PC1')
            axes[1,0].set_ylabel('PC2')
        
        # 4. 聚类大小分布
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        axes[1,1].bar(unique, counts, color='skyblue', alpha=0.7)
        axes[1,1].set_title('各聚类样本数量')
        axes[1,1].set_xlabel('聚类编号')
        axes[1,1].set_ylabel('样本数量')
        
        plt.tight_layout()
        plt.show()
    
    def get_cluster_characteristics(self):
        """获取聚类特征描述"""
        if self.cluster_labels is None:
            print("请先执行聚类！")
            return
        
        print("\n=== 聚类特征描述 ===")
        
        cluster_data = self.X.copy()
        cluster_data['cluster'] = self.cluster_labels
        
        for cluster_id in range(self.optimal_k):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_samples = cluster_data[cluster_mask]
            
            print(f"\n聚类 {cluster_id} (样本数: {len(cluster_samples)}):")
            print(f"  平均收入: {cluster_samples['income'].mean():.0f}")
            print(f"  平均信用评分: {cluster_samples['credit_score'].mean():.0f}")
            print(f"  平均贷款金额: {cluster_samples['loan_amount'].mean():.0f}")
            print(f"  平均工作年限: {cluster_samples['years_employed'].mean():.1f}")
            print(f"  平均积分: {cluster_samples['points'].mean():.1f}")
            
            if self.y is not None:
                approval_rate = self.y[cluster_mask].mean()
                print(f"  贷款批准率: {approval_rate:.1%}")
    
    def run_clustering_analysis(self, max_k=8):
        """运行完整的聚类分析流程"""
        print("开始聚类分析...")
        
        # 1. 寻找最佳k值
        k_results = self.find_optimal_k(max_k)
        
        # 2. 绘制k值选择图表
        self.plot_k_selection(k_results)
        
        # 3. 执行聚类
        self.perform_clustering()
        
        # 4. 分析聚类结果
        cluster_data = self.analyze_clusters()
        
        # 5. 可视化聚类结果
        self.visualize_clusters()
        
        # 6. 获取聚类特征描述
        self.get_cluster_characteristics()
        
        print("\n聚类分析完成！")
        return cluster_data, k_results

def main():
    """主函数示例"""
    # 假设已经完成了数据预处理
    from data_preprocessing import LoanDataPreprocessor
    
    # 加载和预处理数据
    preprocessor = LoanDataPreprocessor('loan_approval.csv')
    X, y, raw_data = preprocessor.run_preprocessing()
    
    # 创建聚类分析器
    clustering_analyzer = LoanClusteringAnalyzer(X, y)
    
    # 运行聚类分析
    cluster_data, k_results = clustering_analyzer.run_clustering_analysis()
    
    return clustering_analyzer, cluster_data, k_results

if __name__ == "__main__":
    clustering_analyzer, cluster_data, k_results = main()
