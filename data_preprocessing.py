"""
数据预处理模块
功能：数据读取、探索性数据分析、缺失值处理、特征编码、标准化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class LoanDataPreprocessor:
    """贷款数据预处理器"""
    
    def __init__(self, data_path):
        """
        初始化预处理器
        
        Args:
            data_path (str): 数据文件路径
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """加载原始数据"""
        print("正在加载数据...")
        self.raw_data = pd.read_csv(self.data_path)
        print(f"数据加载完成，形状: {self.raw_data.shape}")
        return self.raw_data
    
    def explore_data(self):
        """探索性数据分析"""
        print("\n=== 数据基本信息 ===")
        print(f"数据形状: {self.raw_data.shape}")
        print(f"列名: {list(self.raw_data.columns)}")
        
        print("\n=== 数据类型 ===")
        print(self.raw_data.dtypes)
        
        print("\n=== 缺失值统计 ===")
        missing_data = self.raw_data.isnull().sum()
        print(missing_data[missing_data > 0])
        
        print("\n=== 数值特征统计 ===")
        numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        print(self.raw_data[numeric_cols].describe())
        
        print("\n=== 类别特征统计 ===")
        categorical_cols = self.raw_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            print(f"\n{col} 的唯一值数量: {self.raw_data[col].nunique()}")
            print(f"{col} 的值分布:")
            print(self.raw_data[col].value_counts().head())
    
    def visualize_data(self):
        """数据可视化"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('贷款数据探索性分析', fontsize=16)
        
        # 1. 贷款审批分布
        self.raw_data['loan_approved'].value_counts().plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('贷款审批分布')
        axes[0,0].set_xlabel('贷款审批')
        axes[0,0].set_ylabel('数量')
        
        # 2. 收入分布
        axes[0,1].hist(self.raw_data['income'], bins=30, alpha=0.7)
        axes[0,1].set_title('收入分布')
        axes[0,1].set_xlabel('收入')
        axes[0,1].set_ylabel('频次')
        
        # 3. 信用评分分布
        axes[0,2].hist(self.raw_data['credit_score'], bins=30, alpha=0.7)
        axes[0,2].set_title('信用评分分布')
        axes[0,2].set_xlabel('信用评分')
        axes[0,2].set_ylabel('频次')
        
        # 4. 贷款金额分布
        axes[1,0].hist(self.raw_data['loan_amount'], bins=30, alpha=0.7)
        axes[1,0].set_title('贷款金额分布')
        axes[1,0].set_xlabel('贷款金额')
        axes[1,0].set_ylabel('频次')
        
        # 5. 工作年限分布
        axes[1,1].hist(self.raw_data['years_employed'], bins=20, alpha=0.7)
        axes[1,1].set_title('工作年限分布')
        axes[1,1].set_xlabel('工作年限')
        axes[1,1].set_ylabel('频次')
        
        # 6. 收入与信用评分散点图
        scatter = axes[1,2].scatter(self.raw_data['income'], self.raw_data['credit_score'], 
                                   c=self.raw_data['loan_approved'].astype(int), alpha=0.6)
        axes[1,2].set_title('收入 vs 信用评分')
        axes[1,2].set_xlabel('收入')
        axes[1,2].set_ylabel('信用评分')
        
        plt.tight_layout()
        plt.show()
    
    def handle_missing_values(self):
        """处理缺失值"""
        print("\n=== 处理缺失值 ===")
        missing_before = self.raw_data.isnull().sum().sum()
        print(f"处理前缺失值总数: {missing_before}")
        
        # 对于数值型特征，使用中位数填充
        numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='median')
        self.raw_data[numeric_cols] = imputer.fit_transform(self.raw_data[numeric_cols])
        
        # 对于类别型特征，使用众数填充
        categorical_cols = self.raw_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.raw_data[col].isnull().sum() > 0:
                mode_value = self.raw_data[col].mode()[0]
                self.raw_data[col].fillna(mode_value, inplace=True)
        
        missing_after = self.raw_data.isnull().sum().sum()
        print(f"处理后缺失值总数: {missing_after}")
    
    def encode_categorical_features(self):
        """编码类别特征"""
        print("\n=== 编码类别特征 ===")
        categorical_cols = self.raw_data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != 'name':  # 姓名列不进行编码
                le = LabelEncoder()
                self.raw_data[f'{col}_encoded'] = le.fit_transform(self.raw_data[col])
                self.encoders[col] = le
                print(f"{col} 编码完成，唯一值数量: {len(le.classes_)}")
    
    def detect_outliers(self):
        """检测异常值"""
        print("\n=== 异常值检测 ===")
        numeric_cols = ['income', 'credit_score', 'loan_amount', 'years_employed', 'points']
        
        for col in numeric_cols:
            Q1 = self.raw_data[col].quantile(0.25)
            Q3 = self.raw_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.raw_data[(self.raw_data[col] < lower_bound) | 
                                   (self.raw_data[col] > upper_bound)]
            print(f"{col}: 异常值数量 {len(outliers)} ({len(outliers)/len(self.raw_data)*100:.2f}%)")
    
    def prepare_features(self):
        """准备特征数据"""
        print("\n=== 准备特征数据 ===")
        
        # 选择用于分析的特征
        feature_cols = ['income', 'credit_score', 'loan_amount', 'years_employed', 'points']
        
        # 添加编码后的类别特征
        if 'city_encoded' in self.raw_data.columns:
            feature_cols.append('city_encoded')
        
        # 创建特征矩阵
        X = self.raw_data[feature_cols].copy()
        y = self.raw_data['loan_approved'].copy()
        
        # 标准化特征
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        self.processed_data = {
            'X': X,
            'X_scaled': X_scaled,
            'y': y,
            'feature_names': feature_cols
        }
        
        print(f"特征矩阵形状: {X_scaled.shape}")
        print(f"目标变量分布: {y.value_counts().to_dict()}")
        
        return X_scaled, y
    
    def run_preprocessing(self):
        """运行完整的预处理流程"""
        print("开始数据预处理流程...")
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 探索数据
        self.explore_data()
        
        # 3. 可视化
        self.visualize_data()
        
        # 4. 处理缺失值
        self.handle_missing_values()
        
        # 5. 编码类别特征
        self.encode_categorical_features()
        
        # 6. 检测异常值
        self.detect_outliers()
        
        # 7. 准备特征
        X, y = self.prepare_features()
        
        print("\n数据预处理完成！")
        return X, y, self.raw_data

def main():
    """主函数示例"""
    # 创建预处理器
    preprocessor = LoanDataPreprocessor('loan_approval.csv')
    
    # 运行预处理
    X, y, raw_data = preprocessor.run_preprocessing()
    
    return preprocessor, X, y, raw_data

if __name__ == "__main__":
    preprocessor, X, y, raw_data = main()
