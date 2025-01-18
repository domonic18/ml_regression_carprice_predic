#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
二手车价格预测系统的 Streamlit 前端应用程序。
本模块实现了streamlit的前端页面，功能包括：
1. 数据分析
2. 数据清理
3. 模型训练
4. 效果验证

作者: 诸葛东明
日期: 2025-01-04
版本: 1.0.0
"""

import streamlit as st
import pandas as pd
import torch
import os
from ydata_profiling import ProfileReport
from data_processing.data_processor import DataProcessor
from models.car_price_model import CarPriceModel
from train import train_car_price_model
import numpy as np
import logging

def update_progress(epoch, train_loss, test_loss):
    """
    更新训练进度的回调函数
    
    Args:
        epoch (int): 当前训练轮次
        train_loss (float): 训练损失
        test_loss (float): 测试损失
    """
    if not hasattr(update_progress, 'train_losses'):
        update_progress.train_losses = []
        update_progress.test_losses = []
    
    # 更新进度条
    progress = (epoch + 1) / st.session_state.get('total_epochs', 100)
    st.session_state.progress_bar.progress(progress)
    
    # 更新状态文本
    st.session_state.status_text.text(
        f'训练轮次 {epoch+1}/{st.session_state.get("total_epochs", 100)} - '
        f'训练损失: {train_loss:.5f}, 测试损失: {test_loss:.5f}'
    )
    
    # 更新损失图表
    update_progress.train_losses.append(train_loss)
    update_progress.test_losses.append(test_loss)
    loss_df = pd.DataFrame({
        "训练损失": update_progress.train_losses,
        "测试损失": update_progress.test_losses
    })
    st.session_state.loss_chart.line_chart(loss_df)

def train_model(epochs, learning_rate):
    """
    在Streamlit界面中训练模型并显示进度
    
    Args:
        epochs (int): 训练轮数
        learning_rate (float): 学习率
    """
    # 创建进度条和状态显示
    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_chart = st.empty()
    
    # 创建用于实时显示损失的列表
    train_losses = []
    test_losses = []
    
    try:
        # 调用训练函数
        model, train_losses, test_losses, processor = train_car_price_model(
            st.session_state.uploaded_file.name,
            epochs=epochs,
            learning_rate=learning_rate,
            progress_callback=update_progress
        )
        
        # 保存模型
        model_save_path = "trained_model.pth"
        torch.save(model.state_dict(), model_save_path)
        
        # 保存缩放参数 - 现在processor是训练函数返回的
        if processor is not None:  # 添加检查
            scaling_params_path = "scaling_params.npz"
            np.savez(
                scaling_params_path,
                mean_X=processor.mean_X,
                std_X=processor.std_X,
                mean_y=processor.mean_y,
                std_y=processor.std_y
            )
        else:
            st.warning("未能获取数据处理器实例，缩放参数未保存")
        
        return model, train_losses, test_losses
        
    except Exception as e:
        st.error(f"训练过程中发生错误: {str(e)}")
        return None, None, None

def load_model(model_path, input_features):
    """
    加载模型并准备预测
    
    Args:
        model_path: 模型文件路径或文件对象
        input_features: 输入特征数量
        
    Returns:
        model: 加载好的模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = CarPriceModel(in_features=input_features, out_features=1)
    
    try:
        # 如果输入是文件对象，需要先读取内容
        if hasattr(model_path, 'read'):
            model_data = model_path.read()
            import io
            buffer = io.BytesIO(model_data)
            # 尝试加载模型
            loaded = torch.load(buffer, map_location=device)
        else:
            loaded = torch.load(model_path, map_location=device)
        
        # 判断加载的是整个模型还是状态字典
        if isinstance(loaded, CarPriceModel):
            # 如果加载的是整个模型，直接使用
            model = loaded
        else:
            # 如果加载的是状态字典，加载参数
            model.load_state_dict(loaded)
            
        model.to(device)
        model.eval()  # 设置为评估模式
        return model
        
    except Exception as e:
        raise Exception(f"加载模型时出错: {str(e)}")

def predict_price(model, features):
    """
    使用模型进行预测
    
    Args:
        model: 加载好的模型
        features: 预处理后的特征数据
        
    Returns:
        float: 预测价格
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # 确保输入是正确的形状和类型
        if isinstance(features, pd.Series):
            features = features.values
            
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
            
        # 转换为tensor并移动到正确的设备
        features = torch.FloatTensor(features).to(device)
        
        # 进行预测
        with torch.no_grad():
            prediction = model(features)
            return prediction.item()
            
    except Exception as e:
        raise Exception(f"预测过程中发生错误: {str(e)}")

def show_file_uploader():
    """显示文件上传组件"""
    uploaded_file = st.sidebar.file_uploader("选择CSV文件", type="csv", key="shared_uploader")
    if uploaded_file and uploaded_file != st.session_state.get('uploaded_file'):
        st.session_state.uploaded_file = uploaded_file
        # 读取CSV时指定数据类型
        try:
            df = pd.read_csv(uploaded_file, delimiter='\s+')
            # 尝试将可能是数值的列转换为float类型
            for col in df.columns:
                try:
                    if df[col].dtype == 'object':  # 如果列是对象类型
                        df[col] = pd.to_numeric(df[col], errors='coerce')  # 尝试转换为数值
                except Exception as e:
                    logging.warning(f"列 {col} 转换类型时出错: {str(e)}")
            
            st.session_state.df = df
        except Exception as e:
            st.error(f"读取文件时出错: {str(e)}")
            return None
    return uploaded_file

def data_analysis():
    """数据分析功能"""
    st.sidebar.header("数据上传")
    uploaded_file = show_file_uploader()
    
    if uploaded_file:
        st.dataframe(st.session_state.df)

        if st.sidebar.button("分析数据"):
            with st.spinner("正在生成详细分析报告，请稍候..."):
                profile = ProfileReport(st.session_state.df, title="数据分析报告", minimal=True)
                report_html = profile.to_html()
                st.subheader("Pandas Profiling Report")
                st.components.v1.html(report_html, width=1000, height=550, scrolling=True)

def data_cleaning():
    """数据清洗功能"""
    st.sidebar.header("数据上传")
    uploaded_file = show_file_uploader()
    
    if uploaded_file:
        if st.sidebar.button("开始数据清洗", type="primary"):
            processor = DataProcessor()
            with st.spinner("正在清洗数据..."):
                X, y = processor.load_and_analyze_data(st.session_state.df)
                cleaned_df = pd.DataFrame(X, columns=X.columns)
                cleaned_df['price'] = y  # 添加price列
                st.session_state.cleaned_data = (X, y)
                st.session_state.cleaned_df = cleaned_df
            
            st.success("数据清洗完成！")
            
        # 在侧边栏添加保存相关控件
        if 'cleaned_df' in st.session_state:
            st.sidebar.markdown("---")  # 添加分隔线
            st.sidebar.subheader("保存清洗后的数据")
            
            # 添加目录选择输入框到侧边栏,默认保存到上级的data目录
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            default_save_dir = os.path.join(current_dir, "data")
            
            save_dir = st.sidebar.text_input(
                "保存目录",
                value=default_save_dir,
                help="输入要保存洗后数据的目录路径",
                key="save_dir_input"
            )
            
            if st.sidebar.button("保存数据"):
                original_filename = os.path.splitext(st.session_state.uploaded_file.name)[0]
                cleaned_filename = f"{original_filename}_cleaned.csv"
                
                try:
                    save_path = os.path.join(save_dir, cleaned_filename)
                    os.makedirs(save_dir, exist_ok=True)
                    st.session_state.cleaned_df.to_csv(save_path, index=False)
                    st.sidebar.success(f"数据已保存至: {save_path}")
                except Exception as e:
                    st.sidebar.error(f"保存文件时发生错误: {str(e)}")
            
            # 在页面显示数据预览
            if 'cleaned_df' in st.session_state:
                st.subheader("清洗后的数据预览")
                st.dataframe(st.session_state.cleaned_df.head(10))
        else:
            st.warning("请先上传数据文件")

def model_training():
    """模型训练功能"""
    st.sidebar.header("训练参数设置")
    training_file = st.sidebar.file_uploader(
        "选择训练数据文件", 
        type="csv", 
        key="training_file_uploader"
    )
    if training_file:
        # 显示文件信息
        st.subheader("训练数据信息")
        st.write(f"文件名: {training_file.name}")
        
        # 读取并显示数据预览
        df = pd.read_csv(training_file)
        st.write(f"数据总行数: {len(df)}")
        st.write("数据预览 (前10行):")
        st.dataframe(df.head(10))
        
        # 显示基本统计信息
        st.write("数据统计信息:")
        st.dataframe(df.describe())
        
        # 训练参数设置
        epochs = st.sidebar.number_input("训练轮数", value=100, min_value=1)
        learning_rate = st.sidebar.number_input("学习率", value=1e-4, format="%.4f")
        
        if st.sidebar.button("开始训练"):
            # 创建进度显示组件
            st.session_state.progress_bar = st.progress(0)
            st.session_state.status_text = st.empty()
            st.session_state.loss_chart = st.empty()
            st.session_state.total_epochs = epochs
            
            with st.spinner("正在训练模型..."):
                model, train_losses, test_losses, processor = train_car_price_model(
                    df,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    progress_callback=update_progress
                )
                # 保存模型和缩放参数
                try:
                    # 保存模型
                    model_save_path = "model.pth"
                    torch.save(model.state_dict(), model_save_path)
                    
                    # 保存缩放参数
                    if processor is not None:
                        scaling_params_path = "scaling_params.npz"
                        np.savez(
                            scaling_params_path,
                            mean_X=processor.mean_X,
                            std_X=processor.std_X,
                            mean_y=processor.mean_y,
                            std_y=processor.std_y
                        )
                        st.write(f"mean_X: {processor.mean_X}")
                        st.write(f"std_X: {processor.std_X}")
                        st.write(f"mean_y: {processor.mean_y}")
                        st.write(f"std_y: {processor.std_y}")
                        st.success(f"模型已保存至: {model_save_path}")
                        st.success(f"缩放参数已保存至: {scaling_params_path}")
                    else:
                        st.warning("未能获取数据处理器实例，缩放参数未保存")
                except Exception as e:
                    st.error(f"保存模型或缩放参数时发生错误: {str(e)}")
                    
            st.success("模型训练完成！")
            st.line_chart(pd.DataFrame({
                "训练损失": train_losses,
                "测试损失": test_losses
            }))

def model_validation():
    """效果验证功能"""
    st.sidebar.header("🔍 模型载入")
    
    # 获取当前目录下的所有.pth和.npz文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_files = [f for f in os.listdir(current_dir) if f.endswith('.pth')]
    scaling_param_files = [f for f in os.listdir(current_dir) if f.endswith('.npz')]
    
    # 创建下拉框选择模型文件和缩放参数文件
    model_file = st.sidebar.selectbox("📁 选择模型文件", model_files)
    scaling_params_file = st.sidebar.selectbox("📁 选择缩放参数文件", scaling_param_files)
    
    # 添加随机读取数据的按钮
    if st.sidebar.button("🎲 随机读取新数据"):
        # 清除已保存的样本数据，触发重新随机选择
        st.session_state.pop('sample_data', None)
    
    if model_file and scaling_params_file:
        # 构建测试数据文件的绝对路径
        test_file = os.path.join(os.path.dirname(current_dir), "data", "used_car_testB_20200421.csv")
        
        if os.path.exists(test_file):
            # 只在首次加载或点击随机按钮时处理数据
            if 'sample_data' not in st.session_state:
                test_df = pd.read_csv(test_file, delimiter='\s+')
                processor = DataProcessor()
                sample_df = processor.preprocess_sample(test_df)
                # 随机选择一条预处理后的数据并存储在session state中
                st.session_state.sample_data = pd.DataFrame(sample_df).sample(n=1).iloc[0]
            
            # 使用存储的样本数据
            sample_data = st.session_state.sample_data

            # 显示输入数据
            st.subheader("📝 样本数据")
            
            # 创建两列布局
            col1, col2 = st.columns(2)
            
            # 为每个特征创建编辑框
            edited_values = {}
            for i, (feature, value) in enumerate(sample_data.items()):
                # 在左列或右列显示编辑框
                with col1 if i % 2 == 0 else col2:
                    edited_values[feature] = st.number_input(
                        f"📊 {feature}",
                        value=float(value),
                        format="%.2f",
                        key=f"feature_{feature}"
                    )
            
            # 使用编辑后的值创建新的样本数据
            edited_sample_data = pd.Series(edited_values)
            
            if st.button("🎯 开始预测"):
                try:
                    # 加载训练时保存的缩放参数
                    scaling_params_path = os.path.join(current_dir, scaling_params_file)
                    scaling_params = np.load(scaling_params_path, allow_pickle=True)
                    processor = DataProcessor()
                    processor.mean_y = scaling_params['mean_y']
                    processor.std_y = scaling_params['std_y']
                    processor.mean_X = scaling_params['mean_X']
                    processor.std_X = scaling_params['std_X']
                    
                    if processor.mean_X is None or processor.std_X is None:
                        st.error("❌ 缩放参数未正确加载，请检查模型和缩放参数文件。")
                    else:
                        # 使用缩放参数进行标准化
                        X = (edited_sample_data - processor.mean_X) / processor.std_X
                        X = X.astype(np.float32)
                        
                        # 加载模型并预测
                        model_path = os.path.join(current_dir, model_file)
                        model = load_model(model_path, len(X))
                        predicted_price = predict_price(model, X)
                        predicted_price = predicted_price * processor.std_y + processor.mean_y
                        
                        st.success(f"💰 预测价格：¥{predicted_price:.2f}")
                        
                except Exception as e:
                    st.error(f"❌ 预测过程中发生错误: {str(e)}")
        else:
            st.error(f"❌ 找不到测试数据文件：{test_file}")
    else:
        st.warning("⚠️ 请上传模型文件和缩放参数文件")

def main():
    st.title("🚗 二手车价格预测系统")
    
    # 添加项目说明
    if not st.session_state.get('uploaded_file'):
        st.markdown("""
        ### 👋 欢迎使用二手车价格预测系统！
        
        本系统是一个基于深度学习的二手车价格预测工具，提供以下核心功能：
        
        - 📊 **数据分析**：对二手车数据进行可视化分析和统计
        - 🧹 **数据清洗**：自动处理和清理原始数据
        - 🔄 **模型训练**：使用深度学习模型训练价格预测器
        - ✨ **效果验证**：验证模型预测效果并进行实时预测
        
        #### 使用说明
        1. 在左侧边栏选择需要使用的功能
        2. 上传数据文件（CSV格式）
        3. 根据界面提示进行操作
        
        #### 关于我们
        本项目由[17AI技术社区](http://17aitech.com)开发和维护，致力于为用户提供专业的人工智能解决方案。
        
        欢迎访问[17AI技术社区](http://17aitech.com)获取更多AI相关资源和教程！
        """)
        
        st.markdown("---")
    
    # 侧边栏导航
    st.sidebar.header("🎯 功能导航")
    menu = st.sidebar.selectbox(
        "请选择功能",
        ["📊 数据分析", "🧹 数据清洗", "🔄 模型训练", "✨ 效果验证"]
    )
    
    if menu == "📊 数据分析":
        data_analysis()
    elif menu == "🧹 数据清洗":
        data_cleaning()
    elif menu == "🔄 模型训练":
        model_training()
    elif menu == "✨ 效果验证":
        model_validation()

if __name__ == "__main__":
    main() 