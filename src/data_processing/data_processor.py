#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
import logging

class DataProcessor:
    """
    数据处理类，负责数据的加载、分析和预处理
    """
    
    def __init__(self):
        # 添加属性来存储训练集的统计量
        self.mean_X = None
        self.std_X = None
        self.mean_y = None
        self.std_y = None

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
        logging.info("数据形状: %s", df.shape)
        logging.info("\n数据类型信息:\n%s", df.dtypes)
        logging.info("\n数据统计信息:\n%s", df.describe())
        
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
    def _preprocess_features(X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        预处理特征数据，包括数据类型转换和处理缺失值
        
        Args:
            X (pd.DataFrame): 特征数据
            y (pd.Series): 目标变量
            
        Returns:
            tuple: (处理后的特征X, 处理后的目标变量y)
        """
        # 数据类型转换
        for column in X.columns:
            X[column] = pd.to_numeric(X[column], errors='coerce')

        # 剔除包含缺失值的行
        combined_df = pd.concat([X, pd.Series(y, name='target')], axis=1)
        combined_df = combined_df.dropna()
        X = combined_df.drop('target', axis=1)
        y = combined_df['target']
        
        return X, y

    @staticmethod
    def _remove_missing_values(X: pd.DataFrame) -> pd.DataFrame:
        """
        仅剔除特征数据中的缺失值
        
        Args:
            X (pd.DataFrame): 特征数据
            
        Returns:
            pd.DataFrame: 处理后的特征数据
        """
        # 剔除包含缺失值的行
        X_cleaned = X.dropna()
        
        return X_cleaned


    def prepare_data(self,X, y, test_size=0.2, random_state=0) -> tuple:
        """
        准备训练集和测试集，包括数据标准化
        
        Args:
            X: 特征数据
            y: 目标变量
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            tuple: (X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized)
        """
        # 预处理数据
        X, y = DataProcessor._preprocess_features(X, y)

        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 转换为numpy数组
        X_train_np = X_train.values
        X_test_np = X_test.values
        y_train_np = y_train.values
        y_test_np = y_test.values
        
        # 计算并保存统计量
        mean_X = X_train_np.mean(axis=0)
        std_X = X_train_np.std(axis=0)
        std_X[std_X == 0] = 1e-9
        
        mean_y = y_train_np.mean()
        std_y = y_train_np.std()
        if std_y == 0:
            std_y = 1e-9
            
        # 将统计量保存为类属性
        self.mean_X = mean_X
        self.std_X = std_X
        self.mean_y = mean_y
        self.std_y = std_y
        
        X_train_normalized = (X_train_np - mean_X) / std_X
        X_test_normalized = (X_test_np - mean_X) / std_X
        
        # y标签标准化
        y_train_normalized = (y_train_np - mean_y) / std_y
        y_test_normalized = (y_test_np - mean_y) / std_y
        

        
        return X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized 

    def preprocess_sample(self, sample_data: pd.DataFrame) -> pd.DataFrame:
        """
        使用训练集的统计量预处理数据用于预测
        
        Args:
            sample_data (pd.Series): 单条数据
            
        Returns:
            np.ndarray: 预处理后的特征数据
        """

        
        # 剔除缺失值
        sample_df = self._remove_missing_values(sample_data)
        
        # 数据类型转换
        for column in sample_df.columns:
            sample_df.loc[:, column] = pd.to_numeric(sample_df[column], errors='coerce')
        
        return sample_df


