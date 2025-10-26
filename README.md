# 贷款审批数据挖掘分析系统

## 项目简介

本项目是一个完整的贷款审批数据挖掘分析系统，采用多种数据挖掘技术来研究贷款审批的关键因素和预测模型。系统包含数据预处理、聚类分析、关联规则挖掘等核心功能模块。

## 功能特性

### 🔍 数据预处理
- 数据读取与探索性分析（EDA）
- 缺失值处理与异常值检测
- 类别特征编码（LabelEncoder/OneHotEncoder）
- 数值特征标准化
- 数据可视化分析

### 🎯 聚类分析
- K-Means聚类算法
- 多种k值选择方法（Elbow、Silhouette、Davies-Bouldin）
- PCA降维可视化
- 聚类结果分析与解释

### 🔗 关联规则分析
- FP-Growth频繁项集挖掘
- 关联规则生成与评估
- 支持度、置信度、提升度分析
- 规则网络图可视化

### 📊 综合分析
- 完整流程自动化
- 综合报告生成
- 关键洞察提取
- 结果导出（Excel格式）

## 项目结构

```
loan_analysis/
│── loan_approval.csv          # 原始数据集
│── data_preprocessing.py      # 数据预处理模块
│── clustering_analysis.py     # 聚类分析模块
│── association_rules.py       # 关联规则分析模块
│── main.py                   # 主程序入口
│── requirements.txt          # 依赖包列表
│── README.md                 # 项目说明文档
│── loan_analysis_notebook.ipynb  # 分析概述笔记本
```

## 安装与使用

### 环境要求
- Python 3.8+ (注意：Python 3.14可能存在networkx兼容性问题)
- 推荐使用虚拟环境
- 如果遇到networkx导入问题，请使用 `association_rules_simple.py` 模块

### 安装依赖
```bash
pip install -r requirements.txt
```

### 快速开始

#### 方法1：运行完整流程
```python
from main import LoanDataMiningPipeline

# 创建分析流程
pipeline = LoanDataMiningPipeline('loan_approval.csv')

# 运行完整分析
results = pipeline.run_complete_analysis()

# 保存结果
pipeline.save_results('results.xlsx')
```

#### 方法2：分步执行
```python
# 1. 数据预处理
from data_preprocessing import LoanDataPreprocessor
preprocessor = LoanDataPreprocessor('loan_approval.csv')
X, y, raw_data = preprocessor.run_preprocessing()

# 2. 聚类分析
from clustering_analysis import LoanClusteringAnalyzer
clustering_analyzer = LoanClusteringAnalyzer(X, y)
cluster_data, k_results = clustering_analyzer.run_clustering_analysis()

# 3. 关联规则分析
from association_rules import LoanAssociationAnalyzer
association_analyzer = LoanAssociationAnalyzer(raw_data)
rules, approval_rules = association_analyzer.run_association_analysis()
```

#### 方法3：使用Jupyter笔记本
打开 `loan_analysis_notebook.ipynb` 文件，按顺序执行各个分析步骤。

## 数据格式

### 输入数据要求
CSV文件，包含以下字段：
- `name`: 申请人姓名
- `city`: 城市
- `income`: 收入
- `credit_score`: 信用评分
- `loan_amount`: 贷款金额
- `years_employed`: 工作年限
- `points`: 积分
- `loan_approved`: 贷款审批结果（True/False）

### 输出结果
- 预处理后的特征数据
- 聚类分析结果
- 关联规则列表
- 综合分析报告（Excel格式）

## 核心算法

### 聚类分析
- **K-Means**: 基于距离的无监督聚类算法
- **PCA**: 主成分分析降维
- **评估指标**: Silhouette Score, Davies-Bouldin Index

### 关联规则
- **FP-Growth**: 频繁模式增长算法
- **评估指标**: 支持度(Support), 置信度(Confidence), 提升度(Lift)

## 参数配置

### 聚类分析参数
```python
# 在 clustering_analysis.py 中调整
max_k = 10  # 最大聚类数
```

### 关联规则参数
```python
# 在 association_rules.py 中调整
min_support = 0.1      # 最小支持度
min_confidence = 0.5   # 最小置信度
min_lift = 1.0         # 最小提升度
```

## 结果解读

### 聚类分析结果
- **Silhouette Score**: 值越接近1表示聚类效果越好
- **Davies-Bouldin Index**: 值越小表示聚类效果越好
- **聚类中心**: 描述各聚类的典型特征

### 关联规则结果
- **支持度**: 规则在数据集中出现的频率
- **置信度**: 规则的可信程度
- **提升度**: 规则的有效性，大于1表示正相关

## 扩展功能

### 自定义分析
可以继承基础类来实现自定义分析：
```python
class CustomAnalyzer(LoanDataPreprocessor):
    def custom_analysis(self):
        # 自定义分析逻辑
        pass
```

### 添加新的聚类算法
```python
from sklearn.cluster import DBSCAN
# 在 clustering_analysis.py 中添加新的聚类方法
```

## 常见问题

### Q: 如何处理中文显示问题？
A: 确保系统安装了中文字体，代码中已设置 `plt.rcParams['font.sans-serif'] = ['SimHei']`

### Q: 内存不足怎么办？
A: 可以调整数据采样或使用更高效的算法参数

### Q: 如何调整可视化效果？
A: 修改各模块中的 `figsize` 和 `dpi` 参数

## 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 许可证

MIT License

## 联系方式

如有问题，请通过Issue联系。
