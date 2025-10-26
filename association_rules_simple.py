"""
关联规则分析模块（简化版，不依赖networkx）
功能：FP-Growth频繁项集挖掘、关联规则生成、规则可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class LoanAssociationAnalyzer:
    """贷款数据关联规则分析器"""
    
    def __init__(self, data):
        """
        初始化关联规则分析器
        
        Args:
            data (pd.DataFrame): 包含贷款数据的DataFrame
        """
        self.data = data
        self.transaction_data = None
        self.frequent_itemsets = None
        self.rules = None
        
    def prepare_transaction_data(self):
        """准备事务数据"""
        print("正在准备事务数据...")
        
        # 创建事务数据
        transactions = []
        
        for idx, row in self.data.iterrows():
            transaction = []
            
            # 收入水平
            if row['income'] < 50000:
                transaction.append('低收入')
            elif row['income'] < 100000:
                transaction.append('中等收入')
            else:
                transaction.append('高收入')
            
            # 信用评分水平
            if row['credit_score'] < 500:
                transaction.append('低信用')
            elif row['credit_score'] < 700:
                transaction.append('中等信用')
            else:
                transaction.append('高信用')
            
            # 贷款金额水平
            if row['loan_amount'] < 20000:
                transaction.append('小额贷款')
            elif row['loan_amount'] < 50000:
                transaction.append('中等贷款')
            else:
                transaction.append('大额贷款')
            
            # 工作年限
            if row['years_employed'] < 5:
                transaction.append('工作年限短')
            elif row['years_employed'] < 15:
                transaction.append('工作年限中等')
            else:
                transaction.append('工作年限长')
            
            # 积分水平
            if row['points'] < 30:
                transaction.append('低积分')
            elif row['points'] < 60:
                transaction.append('中等积分')
            else:
                transaction.append('高积分')
            
            # 贷款审批结果
            if row['loan_approved']:
                transaction.append('贷款批准')
            else:
                transaction.append('贷款拒绝')
            
            transactions.append(transaction)
        
        # 转换为事务编码器格式
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        self.transaction_data = pd.DataFrame(te_ary, columns=te.columns_)
        
        print(f"事务数据准备完成，形状: {self.transaction_data.shape}")
        print(f"项目数量: {len(te.columns_)}")
        
        return self.transaction_data
    
    def mine_frequent_itemsets(self, min_support=0.1):
        """
        挖掘频繁项集
        
        Args:
            min_support (float): 最小支持度阈值
        """
        print(f"正在挖掘频繁项集 (最小支持度: {min_support})...")
        
        if self.transaction_data is None:
            self.prepare_transaction_data()
        
        # 使用FP-Growth算法挖掘频繁项集
        self.frequent_itemsets = fpgrowth(self.transaction_data, 
                                        min_support=min_support, 
                                        use_colnames=True)
        
        print(f"找到 {len(self.frequent_itemsets)} 个频繁项集")
        
        # 显示前10个频繁项集
        print("\n前10个频繁项集:")
        print(self.frequent_itemsets.head(10))
        
        return self.frequent_itemsets
    
    def generate_association_rules(self, min_confidence=0.5, min_lift=1.0):
        """
        生成关联规则
        
        Args:
            min_confidence (float): 最小置信度阈值
            min_lift (float): 最小提升度阈值
        """
        print(f"正在生成关联规则 (最小置信度: {min_confidence}, 最小提升度: {min_lift})...")
        
        if self.frequent_itemsets is None:
            print("请先挖掘频繁项集！")
            return
        
        # 生成关联规则
        self.rules = association_rules(self.frequent_itemsets, 
                                     metric="confidence", 
                                     min_threshold=min_confidence)
        
        # 过滤提升度
        self.rules = self.rules[self.rules['lift'] >= min_lift]
        
        # 按置信度排序
        self.rules = self.rules.sort_values('confidence', ascending=False)
        
        print(f"生成 {len(self.rules)} 条关联规则")
        
        return self.rules
    
    def analyze_rules(self):
        """分析关联规则"""
        if self.rules is None or len(self.rules) == 0:
            print("没有找到关联规则！")
            return
        
        print("\n=== 关联规则分析 ===")
        
        # 显示规则统计信息
        print(f"规则总数: {len(self.rules)}")
        print(f"平均支持度: {self.rules['support'].mean():.3f}")
        print(f"平均置信度: {self.rules['confidence'].mean():.3f}")
        print(f"平均提升度: {self.rules['lift'].mean():.3f}")
        
        # 显示前10条规则
        print("\n前10条关联规则:")
        display_rules = self.rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10)
        for idx, rule in display_rules.iterrows():
            print(f"规则 {idx+1}: {set(rule['antecedents'])} -> {set(rule['consequents'])}")
            print(f"  支持度: {rule['support']:.3f}, 置信度: {rule['confidence']:.3f}, 提升度: {rule['lift']:.3f}")
        
        # 分析贷款批准相关的规则
        approval_rules = self.rules[
            self.rules['consequents'].apply(lambda x: '贷款批准' in x) |
            self.rules['antecedents'].apply(lambda x: '贷款批准' in x)
        ]
        
        if len(approval_rules) > 0:
            print(f"\n与贷款批准相关的规则数量: {len(approval_rules)}")
            print("前5条贷款批准相关规则:")
            for idx, rule in approval_rules.head(5).iterrows():
                print(f"  {set(rule['antecedents'])} -> {set(rule['consequents'])}")
                print(f"  支持度: {rule['support']:.3f}, 置信度: {rule['confidence']:.3f}, 提升度: {rule['lift']:.3f}")
        
        return approval_rules
    
    def visualize_rules(self, top_n=20):
        """可视化关联规则"""
        if self.rules is None or len(self.rules) == 0:
            print("没有规则可以可视化！")
            return
        
        # 选择前N条规则
        top_rules = self.rules.head(top_n)
        
        # 创建可视化（不依赖networkx）
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('关联规则分析可视化', fontsize=16)
        
        # 1. 规则散点图（替代网络图）
        scatter = axes[0,0].scatter(self.rules['support'], self.rules['confidence'], 
                                  c=self.rules['lift'], cmap='viridis', alpha=0.7, s=50)
        axes[0,0].set_title('关联规则质量分布')
        axes[0,0].set_xlabel('支持度')
        axes[0,0].set_ylabel('置信度')
        cbar = plt.colorbar(scatter, ax=axes[0,0])
        cbar.set_label('提升度')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 支持度分布
        axes[0,1].hist(self.rules['support'], bins=20, alpha=0.7, color='skyblue')
        axes[0,1].set_title('支持度分布')
        axes[0,1].set_xlabel('支持度')
        axes[0,1].set_ylabel('频次')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. 置信度分布
        axes[1,0].hist(self.rules['confidence'], bins=20, alpha=0.7, color='lightgreen')
        axes[1,0].set_title('置信度分布')
        axes[1,0].set_xlabel('置信度')
        axes[1,0].set_ylabel('频次')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. 提升度分布
        axes[1,1].hist(self.rules['lift'], bins=20, alpha=0.7, color='lightcoral')
        axes[1,1].set_title('提升度分布')
        axes[1,1].set_xlabel('提升度')
        axes[1,1].set_ylabel('频次')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 创建规则热力图
        self._create_rules_heatmap(top_rules)
    
    def _create_rules_heatmap(self, rules):
        """创建规则热力图"""
        if len(rules) == 0:
            return
        
        # 创建规则矩阵
        rule_matrix = []
        rule_labels = []
        
        for idx, rule in rules.iterrows():
            rule_str = f"{set(rule['antecedents'])} -> {set(rule['consequents'])}"
            rule_labels.append(rule_str[:50] + "..." if len(rule_str) > 50 else rule_str)
            rule_matrix.append([rule['support'], rule['confidence'], rule['lift']])
        
        rule_matrix = np.array(rule_matrix)
        
        # 创建热力图
        plt.figure(figsize=(12, max(8, len(rules) * 0.3)))
        sns.heatmap(rule_matrix, 
                   xticklabels=['支持度', '置信度', '提升度'],
                   yticklabels=rule_labels,
                   annot=True, 
                   fmt='.3f',
                   cmap='YlOrRd')
        plt.title('关联规则质量指标热力图')
        plt.tight_layout()
        plt.show()
    
    def get_top_rules(self, metric='confidence', top_n=10):
        """获取顶级规则"""
        if self.rules is None or len(self.rules) == 0:
            print("没有找到规则！")
            return
        
        top_rules = self.rules.nlargest(top_n, metric)
        
        print(f"\n=== 按{metric}排序的前{top_n}条规则 ===")
        for idx, rule in top_rules.iterrows():
            print(f"规则 {idx+1}: {set(rule['antecedents'])} -> {set(rule['consequents'])}")
            print(f"  支持度: {rule['support']:.3f}")
            print(f"  置信度: {rule['confidence']:.3f}")
            print(f"  提升度: {rule['lift']:.3f}")
            print()
        
        return top_rules
    
    def run_association_analysis(self, min_support=0.1, min_confidence=0.5, min_lift=1.0):
        """运行完整的关联规则分析流程"""
        print("开始关联规则分析...")
        
        # 1. 准备事务数据
        self.prepare_transaction_data()
        
        # 2. 挖掘频繁项集
        self.mine_frequent_itemsets(min_support)
        
        # 3. 生成关联规则
        self.generate_association_rules(min_confidence, min_lift)
        
        # 4. 分析规则
        approval_rules = self.analyze_rules()
        
        # 5. 可视化规则
        self.visualize_rules()
        
        # 6. 显示顶级规则
        self.get_top_rules('confidence', 10)
        self.get_top_rules('lift', 10)
        
        print("\n关联规则分析完成！")
        return self.rules, approval_rules

def main():
    """主函数示例"""
    # 假设已经完成了数据预处理
    from data_preprocessing import LoanDataPreprocessor
    
    # 加载和预处理数据
    preprocessor = LoanDataPreprocessor('loan_approval.csv')
    X, y, raw_data = preprocessor.run_preprocessing()
    
    # 创建关联规则分析器
    association_analyzer = LoanAssociationAnalyzer(raw_data)
    
    # 运行关联规则分析
    rules, approval_rules = association_analyzer.run_association_analysis(
        min_support=0.1, 
        min_confidence=0.5, 
        min_lift=1.0
    )
    
    return association_analyzer, rules, approval_rules

if __name__ == "__main__":
    association_analyzer, rules, approval_rules = main()
