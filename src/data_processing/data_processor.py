import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

class DataProcessor:
    """
    数据处理类，负责数据的加载、分析和预处理
    """
    
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """
        从文件加载数据
        
        Args:
            file_path (str): 数据文件路径
            
        Returns:
            pd.DataFrame: 加载的数据
        """
        logging.info("开始加载数据...")
        return pd.read_csv(file_path)

    @staticmethod
    def analyze_data(df: pd.DataFrame) -> tuple:
        """
        分析数据并准备特征
        
        Args:
            df (pd.DataFrame): 输入的DataFrame数据
            
        Returns:
            tuple: (特征数据X, 目标变量y)
        """
        logging.info("开始分析数据...")
        logging.info(f"数据形状: {df.shape}")
        logging.info(f"\n数据类型信息:\n{df.dtypes}")
        logging.info(f"\n数据统计信息:\n{df.describe()}")
        
        if 'price' not in df.columns:
            raise ValueError("数据中缺少'price'列")
        
        X = df.drop(['price'], axis=1)
        y = df['price']
        
        return X, y

    @staticmethod
    def load_and_analyze_data(input_data) -> tuple:
        """
        加载并分析数据（支持文件路径或DataFrame输入）
        
        Args:
            input_data (str or pd.DataFrame): 数据文件路径或DataFrame
            
        Returns:
            tuple: (特征数据X, 目标变量y)
        """
        if isinstance(input_data, str):
            df = DataProcessor.load_data(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data
        else:
            raise TypeError("input_data必须是文件路径(str)或DataFrame")
        
        return DataProcessor.analyze_data(df)

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
        # 确保数据为数值类型
        X = X.select_dtypes(include=[np.number]).copy()
        
        # 处理缺失值
        X = X.fillna(X.mean())
        y = pd.to_numeric(y, errors='coerce')
        y = y.fillna(y.mean())
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 转换为numpy数组
        X_train_np = X_train.values
        X_test_np = X_test.values
        y_train_np = y_train.values
        y_test_np = y_test.values
        
        # X标签标准化
        mean_X = X_train_np.mean(axis=0)
        std_X = X_train_np.std(axis=0)
        std_X[std_X == 0] = 1e-9  # 避免除以零
        
        X_train_normalized = (X_train_np - mean_X) / std_X
        X_test_normalized = (X_test_np - mean_X) / std_X
        
        # y标签标准化
        mean_y = y_train_np.mean()
        std_y = y_train_np.std()
        if std_y == 0:
            std_y = 1e-9
        
        y_train_normalized = (y_train_np - mean_y) / std_y
        y_test_normalized = (y_test_np - mean_y) / std_y
        
        return X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized 
