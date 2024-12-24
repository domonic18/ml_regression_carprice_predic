"""
二手车价格预测系统的 Streamlit 前端应用程序。
提供数据分析、数据清理、模型训练和效果验证等功能。
"""

import streamlit as st
import pandas as pd
import torch
import os
from ydata_profiling import ProfileReport
import pandas as pd
from data_processing.data_processor import DataProcessor
from models.car_price_model import CarPriceModel
from train import train_car_price_model
# import logging
# import random


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
        # 调用训练函数，但需要修改train_car_price_model来支持回调
        def update_progress(epoch, train_loss, test_loss):
            # 更新进度条
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            
            # 更新状态文本
            status_text.text(f'训练轮次 {epoch+1}/{epochs} - 训练损失: {train_loss:.5f}, 测试损失: {test_loss:.5f}')
            
            # 更新损失图表
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            loss_df = pd.DataFrame({
                "训练损失": train_losses,
                "测试损失": test_losses
            })
            loss_chart.line_chart(loss_df)
        
        # 调用训练函数
        model, train_losses, test_losses = train_car_price_model(
            st.session_state.uploaded_file.name,
            epochs=epochs,
            learning_rate=learning_rate,
            progress_callback=update_progress
        )
        
        # 保存模型
        model_save_path = "trained_model.pth"
        torch.save(model.state_dict(), model_save_path)
        
        return model, train_losses, test_losses
        
    except Exception as e:
        st.error(f"训练过程中发生错误: {str(e)}")
        return None, None, None

def load_model(model_path, input_features):
    """加载模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CarPriceModel(in_features=input_features, out_features=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def show_file_uploader():
    """显示文件上传组件"""
    uploaded_file = st.sidebar.file_uploader("选择CSV文件", type="csv", key="shared_uploader")
    if uploaded_file and uploaded_file != st.session_state.get('uploaded_file'):
        st.session_state.uploaded_file = uploaded_file
        st.session_state.df = pd.read_csv(uploaded_file, delimiter='\s+')
    return uploaded_file

def main():
    st.title("二手车价格预测系统")
    
    # 侧边栏导航
    menu = st.sidebar.selectbox(
        "功能选择",
        ["数据分析", "数据清理", "模型训练", "效果验证"]
    )
    
    # 存储会话状态
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    
    if menu == "数据分析":
        st.sidebar.header("数据上传")
        uploaded_file = show_file_uploader()
        
        if uploaded_file:
            st.dataframe(st.session_state.df)

            if st.button("分析数据"):
                with st.spinner("正在生成详细分析报告，请稍候..."):
                    profile = ProfileReport(st.session_state.df, title="数据分析报告", minimal=True)
                    report_html = profile.to_html()
                    st.subheader("Pandas Profiling Report")
                    st.components.v1.html(report_html, width=1000, height=550, scrolling=True)
    
    elif menu == "数据清理":
        st.sidebar.header("数据上传")
        uploaded_file = show_file_uploader()
        
        if uploaded_file:
            # 将数据清理按钮移到侧边栏
            if st.sidebar.button("开始数据清理", type="primary"):
                processor = DataProcessor()
                
                # 在主页面显示进度
                with st.spinner("正在清理数据..."):
                    X, y = processor.load_and_analyze_data(st.session_state.df)
                    X_test, X_train, y_test, y_train = processor.prepare_data(X, y)
                    st.session_state.cleaned_data = (X_train, X_test, y_train, y_test)
                    st.session_state.cleaned_df = pd.DataFrame(X_train)
                
                st.success("数据清理完成！")
                
            # 在侧边栏添加保存相关控件
            if 'cleaned_df' in st.session_state:
                st.sidebar.markdown("---")  # 添加分隔线
                st.sidebar.subheader("保存清理后的数据")
                
                # 添加目录选择输入框到侧边栏,默认保存到上级的data目录
                current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                default_save_dir = os.path.join(current_dir, "data")
                
                save_dir = st.sidebar.text_input(
                    "保存目录",
                    value=default_save_dir,
                    help="输入要保存清理后数据的目录路径",
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
            
            # 在主页面显示数据预览
            if 'cleaned_df' in st.session_state:
                st.subheader("清理后的数据预览")
                st.dataframe(st.session_state.cleaned_df.head(10))
        else:
            st.warning("请先上传数据文件")
    
    elif menu == "模型训练":
        st.sidebar.header("训练参数设置")
        epochs = st.sidebar.number_input("训练轮数", value=10000, min_value=1)
        learning_rate = st.sidebar.number_input("学习率", value=1e-4, format="%.4f")
        
        if st.sidebar.button("开始训练"):
            if st.session_state.uploaded_file is None:
                st.error("请先上传并清理数据！")
            else:
                with st.spinner("正在训练模型..."):
                    model, train_losses, test_losses = train_model(epochs, learning_rate)
                st.success("模型训练完成！")
                st.line_chart(pd.DataFrame({
                    "训练损失": train_losses,
                    "测试损失": test_losses
                }))
    
    elif menu == "效果验证":
        st.sidebar.header("模型载")
        model_file = st.sidebar.file_uploader("选择模型文件", type="pth")
        
        if model_file:
            # 从测试数据中随机选择一条记录
            test_file = "data/used_car_testB_20200421.csv"
            if os.path.exists(test_file):
                test_df = pd.read_csv(test_file)
                sample_data = test_df.sample(n=1).iloc[0]
                
                # 显示输入数据
                input_data = st.text_area("输入数据", value=str(sample_data))
                
                if st.button("预测"):
                    # 加载模型并预测
                    processor = DataProcessor()
                    X = processor.preprocess_single_sample(sample_data)
                    model = load_model(model_file, X.shape[1])
                    
                    with torch.no_grad():
                        prediction = model(torch.FloatTensor(X))
                        st.success(f"预测价格：{prediction.item():.2f}")

if __name__ == "__main__":
    main() 