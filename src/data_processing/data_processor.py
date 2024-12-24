import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

class DataProcessor:
    """
    数据处理类，负责数据的加载、分析和预处理
    """
    
    @staticmethod
    def load_and_analyze_data(file_path: str) -> tuple:
        """
        加载并分析数据
        
        Args:
            file_path (str): 数据文件路径
            
        Returns:
            tuple: (特征数据X, 目标变量y)
        """
        logging.info("开始加载数据...")
        df = pd.read_csv(file_path)
        
        logging.info(f"数据形状: {df.shape}")
        logging.info(f"\n数据类型信息:\n{df.dtypes}")
        logging.info(f"\n数据统计信息:\n{df.describe()}")
        
        X = df.drop(['price'], axis=1)
        y = df['price']
        
        return X, y

    @staticmethod
    def prepare_data(X, y, test_size=0.2, random_state=0) -> tuple:
        """
        准备训练集和测试集，包括数据标准化
        
        Args:
            X: 特征数据
            y: 目标变量
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            tuple: (X_train_normalized, X_test_normalized, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # X标签进行标准化
        X_train_np = np.array(X_train)
        X_test_np = np.array(X_test)
        
        mean = X_train_np.mean(axis=0)
        std = X_train_np.std(axis=0) + 1e-9
        
        X_train_normalized = (X_train_np - mean) / std
        X_test_normalized = (X_test_np - mean) / std
        

        # y标签进行标准化
        y_train_np = np.array(y_train)  
        y_test_np = np.array(y_test)

        mean = y_train_np.mean()
        std = y_train_np.std() + 1e-9
        
        y_train_normalized = (y_train_np - mean) / std
        y_test_normalized = (y_test_np - mean) / std

        return X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized 
