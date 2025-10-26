"""
主程序模块
功能：串联整个数据挖掘流程，输出主要结果与可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from data_preprocessing import LoanDataPreprocessor
from clustering_analysis import LoanClusteringAnalyzer
from association_rules_simple import LoanAssociationAnalyzer

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class LoanDataMiningPipeline:
    """贷款数据挖掘完整流程"""
    
    def __init__(self, data_path):
        """
        初始化数据挖掘流程
        
        Args:
            data_path (str): 数据文件路径
        """
        self.data_path = data_path
        self.preprocessor = None
        self.clustering_analyzer = None
        self.association_analyzer = None
        self.results = {}
        
    def run_complete_analysis(self):
        """运行完整的数据挖掘分析流程"""
        print("=" * 60)
        print("贷款审批数据挖掘分析系统")
        print("=" * 60)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 第一阶段：数据预处理
        print("第一阶段：数据预处理")
        print("-" * 30)
        self._run_preprocessing()
        
        # 第二阶段：聚类分析
        print("\n第二阶段：聚类分析")
        print("-" * 30)
        self._run_clustering()
        
        # 第三阶段：关联规则分析
        print("\n第三阶段：关联规则分析")
        print("-" * 30)
        self._run_association_rules()
        
        # 第四阶段：结果汇总
        print("\n第四阶段：结果汇总")
        print("-" * 30)
        self._summarize_results()
        
        print(f"\n分析完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        return self.results
    
    def _run_preprocessing(self):
        """运行数据预处理"""
        print("正在执行数据预处理...")
        
        # 创建预处理器
        self.preprocessor = LoanDataPreprocessor(self.data_path)
        
        # 运行预处理
        X, y, raw_data = self.preprocessor.run_preprocessing()
        
        # 保存结果
        self.results['preprocessing'] = {
            'X': X,
            'y': y,
            'raw_data': raw_data,
            'feature_names': X.columns.tolist(),
            'data_shape': X.shape,
            'target_distribution': y.value_counts().to_dict()
        }
        
        print("数据预处理完成！")
    
    def _run_clustering(self):
        """运行聚类分析"""
        print("正在执行聚类分析...")
        
        if self.preprocessor is None:
            raise ValueError("请先运行数据预处理！")
        
        X = self.results['preprocessing']['X']
        y = self.results['preprocessing']['y']
        
        # 创建聚类分析器
        self.clustering_analyzer = LoanClusteringAnalyzer(X, y)
        
        # 运行聚类分析
        cluster_data, k_results = self.clustering_analyzer.run_clustering_analysis()
        
        # 保存结果
        self.results['clustering'] = {
            'cluster_data': cluster_data,
            'k_results': k_results,
            'optimal_k': self.clustering_analyzer.optimal_k,
            'cluster_labels': self.clustering_analyzer.cluster_labels,
            'silhouette_score': silhouette_score(X, self.clustering_analyzer.cluster_labels) if self.clustering_analyzer.cluster_labels is not None else None
        }
        
        print("聚类分析完成！")
    
    def _run_association_rules(self):
        """运行关联规则分析"""
        print("正在执行关联规则分析...")
        
        if self.preprocessor is None:
            raise ValueError("请先运行数据预处理！")
        
        raw_data = self.results['preprocessing']['raw_data']
        
        # 创建关联规则分析器
        self.association_analyzer = LoanAssociationAnalyzer(raw_data)
        
        # 运行关联规则分析
        rules, approval_rules = self.association_analyzer.run_association_analysis()
        
        # 保存结果
        self.results['association_rules'] = {
            'all_rules': rules,
            'approval_rules': approval_rules,
            'rules_count': len(rules) if rules is not None else 0,
            'approval_rules_count': len(approval_rules) if approval_rules is not None else 0
        }
        
        print("关联规则分析完成！")
    
    def _summarize_results(self):
        """汇总分析结果"""
        print("正在汇总分析结果...")
        
        # 创建结果汇总报告
        summary = {
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_info': {},
            'clustering_summary': {},
            'association_rules_summary': {},
            'key_insights': []
        }
        
        # 数据信息汇总
        if 'preprocessing' in self.results:
            prep_results = self.results['preprocessing']
            summary['data_info'] = {
                'total_samples': prep_results['data_shape'][0],
                'total_features': prep_results['data_shape'][1],
                'approval_rate': prep_results['target_distribution'].get(True, 0) / 
                               sum(prep_results['target_distribution'].values()),
                'feature_names': prep_results['feature_names']
            }
        
        # 聚类分析汇总
        if 'clustering' in self.results:
            cluster_results = self.results['clustering']
            summary['clustering_summary'] = {
                'optimal_clusters': cluster_results['optimal_k'],
                'silhouette_score': cluster_results['silhouette_score'],
                'cluster_distribution': np.bincount(cluster_results['cluster_labels']).tolist()
            }
        
        # 关联规则汇总
        if 'association_rules' in self.results:
            ar_results = self.results['association_rules']
            summary['association_rules_summary'] = {
                'total_rules': ar_results['rules_count'],
                'approval_related_rules': ar_results['approval_rules_count']
            }
        
        # 关键洞察
        self._generate_key_insights(summary)
        
        # 保存汇总结果
        self.results['summary'] = summary
        
        # 打印汇总报告
        self._print_summary_report(summary)
        
        # 创建综合可视化
        self._create_comprehensive_visualization()
    
    def _generate_key_insights(self, summary):
        """生成关键洞察"""
        insights = []
        
        # 基于数据信息的洞察
        if 'data_info' in summary:
            data_info = summary['data_info']
            approval_rate = data_info['approval_rate']
            
            if approval_rate < 0.3:
                insights.append(f"贷款批准率较低 ({approval_rate:.1%})，需要重点关注审批标准")
            elif approval_rate > 0.7:
                insights.append(f"贷款批准率较高 ({approval_rate:.1%})，审批标准可能较为宽松")
            else:
                insights.append(f"贷款批准率适中 ({approval_rate:.1%})，审批标准相对平衡")
        
        # 基于聚类分析的洞察
        if 'clustering_summary' in summary:
            cluster_summary = summary['clustering_summary']
            optimal_k = cluster_summary['optimal_clusters']
            silhouette_score = cluster_summary['silhouette_score']
            
            if silhouette_score > 0.5:
                insights.append(f"聚类效果良好 (Silhouette Score: {silhouette_score:.3f})，客户群体特征明显")
            else:
                insights.append(f"聚类效果一般 (Silhouette Score: {silhouette_score:.3f})，客户群体特征不够明显")
            
            insights.append(f"建议将客户分为 {optimal_k} 个群体进行差异化服务")
        
        # 基于关联规则的洞察
        if 'association_rules_summary' in summary:
            ar_summary = summary['association_rules_summary']
            total_rules = ar_summary['total_rules']
            approval_rules = ar_summary['approval_related_rules']
            
            if approval_rules > 0:
                insights.append(f"发现 {approval_rules} 条与贷款审批相关的关联规则，可用于优化审批流程")
            else:
                insights.append("未发现明显的贷款审批关联规则，建议调整参数重新分析")
        
        summary['key_insights'] = insights
    
    def _print_summary_report(self, summary):
        """打印汇总报告"""
        print("\n" + "=" * 60)
        print("分析结果汇总报告")
        print("=" * 60)
        
        # 数据概览
        if 'data_info' in summary:
            data_info = summary['data_info']
            print(f"\n📊 数据概览:")
            print(f"   • 总样本数: {data_info['total_samples']:,}")
            print(f"   • 特征数量: {data_info['total_features']}")
            print(f"   • 贷款批准率: {data_info['approval_rate']:.1%}")
        
        # 聚类分析结果
        if 'clustering_summary' in summary:
            cluster_summary = summary['clustering_summary']
            print(f"\n🔍 聚类分析结果:")
            print(f"   • 最佳聚类数: {cluster_summary['optimal_clusters']}")
            print(f"   • 聚类质量 (Silhouette Score): {cluster_summary['silhouette_score']:.3f}")
            print(f"   • 各聚类样本分布: {cluster_summary['cluster_distribution']}")
        
        # 关联规则结果
        if 'association_rules_summary' in summary:
            ar_summary = summary['association_rules_summary']
            print(f"\n🔗 关联规则分析结果:")
            print(f"   • 总规则数: {ar_summary['total_rules']}")
            print(f"   • 审批相关规则数: {ar_summary['approval_related_rules']}")
        
        # 关键洞察
        if 'key_insights' in summary:
            print(f"\n💡 关键洞察:")
            for i, insight in enumerate(summary['key_insights'], 1):
                print(f"   {i}. {insight}")
        
        print("\n" + "=" * 60)
    
    def _create_comprehensive_visualization(self):
        """创建综合可视化"""
        print("正在创建综合可视化...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('贷款审批数据挖掘分析综合报告', fontsize=16, fontweight='bold')
        
        # 1. 贷款审批分布
        if 'preprocessing' in self.results:
            y = self.results['preprocessing']['y']
            approval_counts = y.value_counts()
            axes[0,0].pie(approval_counts.values, labels=['拒绝', '批准'], autopct='%1.1f%%', 
                         colors=['lightcoral', 'lightgreen'])
            axes[0,0].set_title('贷款审批分布')
        
        # 2. 聚类结果
        if 'clustering' in self.results:
            cluster_labels = self.results['clustering']['cluster_labels']
            if cluster_labels is not None:
                unique, counts = np.unique(cluster_labels, return_counts=True)
                axes[0,1].bar(unique, counts, color='skyblue', alpha=0.7)
                axes[0,1].set_title('聚类分布')
                axes[0,1].set_xlabel('聚类编号')
                axes[0,1].set_ylabel('样本数量')
        
        # 3. 特征重要性（基于方差）
        if 'preprocessing' in self.results:
            X = self.results['preprocessing']['X']
            feature_vars = X.var().sort_values(ascending=True)
            axes[0,2].barh(range(len(feature_vars)), feature_vars.values)
            axes[0,2].set_yticks(range(len(feature_vars)))
            axes[0,2].set_yticklabels(feature_vars.index)
            axes[0,2].set_title('特征方差分布')
            axes[0,2].set_xlabel('方差')
        
        # 4. 收入与信用评分关系
        if 'preprocessing' in self.results:
            raw_data = self.results['preprocessing']['raw_data']
            scatter = axes[1,0].scatter(raw_data['income'], raw_data['credit_score'], 
                                      c=raw_data['loan_approved'].astype(int), 
                                      alpha=0.6, cmap='RdYlGn')
            axes[1,0].set_title('收入 vs 信用评分')
            axes[1,0].set_xlabel('收入')
            axes[1,0].set_ylabel('信用评分')
        
        # 5. 关联规则质量分布
        if 'association_rules' in self.results:
            rules = self.results['association_rules']['all_rules']
            if rules is not None and len(rules) > 0:
                axes[1,1].scatter(rules['support'], rules['confidence'], 
                                c=rules['lift'], cmap='viridis', alpha=0.6)
                axes[1,1].set_title('关联规则质量分布')
                axes[1,1].set_xlabel('支持度')
                axes[1,1].set_ylabel('置信度')
                cbar = plt.colorbar(axes[1,1].collections[0], ax=axes[1,1])
                cbar.set_label('提升度')
        
        # 6. 分析时间轴
        analysis_steps = ['数据预处理', '聚类分析', '关联规则分析', '结果汇总']
        step_times = [1, 2, 3, 4]  # 模拟时间
        axes[1,2].plot(step_times, analysis_steps, 'o-', linewidth=2, markersize=8)
        axes[1,2].set_title('分析流程时间轴')
        axes[1,2].set_xlabel('步骤')
        axes[1,2].set_ylabel('分析阶段')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, output_path='loan_analysis_results.xlsx'):
        """保存分析结果到Excel文件"""
        print(f"正在保存结果到 {output_path}...")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 保存原始数据
            if 'preprocessing' in self.results:
                self.results['preprocessing']['raw_data'].to_excel(
                    writer, sheet_name='原始数据', index=False)
                
                # 保存预处理后的特征数据
                feature_data = self.results['preprocessing']['X'].copy()
                feature_data['loan_approved'] = self.results['preprocessing']['y']
                feature_data.to_excel(writer, sheet_name='预处理数据', index=False)
            
            # 保存聚类结果
            if 'clustering' in self.results:
                self.results['clustering']['cluster_data'].to_excel(
                    writer, sheet_name='聚类结果', index=False)
            
            # 保存关联规则
            if 'association_rules' in self.results:
                rules = self.results['association_rules']['all_rules']
                if rules is not None and len(rules) > 0:
                    rules.to_excel(writer, sheet_name='关联规则', index=False)
        
        print(f"结果已保存到 {output_path}")

def main():
    """主函数"""
    # 创建数据挖掘流程
    pipeline = LoanDataMiningPipeline('loan_approval.csv')
    
    # 运行完整分析
    results = pipeline.run_complete_analysis()
    
    # 保存结果
    pipeline.save_results()
    
    return pipeline, results

if __name__ == "__main__":
    pipeline, results = main()
