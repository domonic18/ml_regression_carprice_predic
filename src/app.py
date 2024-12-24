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
        # 调用训练函数，但需要修改train_car_price_model来支持回调
        def update_progress(epoch, train_loss, test_loss):
            # 更新进度条
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            
            # 更新状态文本
            status_text.text(f'训练轮次 {epoch+1}/{epochs} - 训练损失: {train_loss:.5f}, 测试损失: {test_loss:.5f}')
            
            # 更新损失图���
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
    
    # 如果输入是文件对象，需要先读取内容
    if hasattr(model_path, 'read'):
        model_data = model_path.read()
        # 创建临时的BytesIO对象
        import io
        buffer = io.BytesIO(model_data)
        # 加载模型参数
        state_dict = torch.load(buffer, map_location=device)
    else:
        # 如果是文件路径，直接加载
        state_dict = torch.load(model_path, map_location=device)
    
    # 加载模型参数
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # 设置为评估模式
    return model

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
    
    # 确保输入是正确的形状
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    
    # 转换为tensor
    features_tensor = torch.FloatTensor(features).to(device)
    
    # 进行预测
    with torch.no_grad():
        prediction = model(features_tensor)
    
    return prediction.item()

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
        ["数据分析", "数据清洗", "模型训练", "效果验证"]
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
    
    elif menu == "数据清洗":
        st.sidebar.header("数据上传")
        uploaded_file = show_file_uploader()
        
        if uploaded_file:
            # 将数据清理按钮移到侧边栏
            if st.sidebar.button("开始数据清洗", type="primary"):
                processor = DataProcessor()
                
                # 在主页面显示进度
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
                
                # 添���目录选择输入框到侧边栏,默认保存到上级的data目录
                current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                default_save_dir = os.path.join(current_dir, "data")
                
                save_dir = st.sidebar.text_input(
                    "保存目录",
                    value=default_save_dir,
                    help="输入要保存清洗后数据的目录路径",
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
    
    elif menu == "模型训练":
        st.sidebar.header("训练参数设置")
        
        # 添加独立的文件上传器
        training_file = st.sidebar.file_uploader(
            "选择训练数据文件", 
            type="csv", 
            key="training_file_uploader"  # 使用不同的key避免与其他上传器冲突
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
                    model, train_losses, test_losses = train_car_price_model(
                        df,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        progress_callback=update_progress
                    )
                st.success("模型训练完成！")
                st.line_chart(pd.DataFrame({
                    "训练损失": train_losses,
                    "测试损失": test_losses
                }))
    
    elif menu == "效果验证":
        st.sidebar.header("模型载入")
        model_file = st.sidebar.file_uploader("选择模型文件", type="pth")
        
        if model_file:
            # 构建测试数据文件的绝对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            test_file = os.path.join(os.path.dirname(current_dir), "data", "used_car_testB_20200421.csv")
            
            if os.path.exists(test_file):
                test_df = pd.read_csv(test_file, delimiter='\s+')
                sample_data = test_df.sample(n=1).iloc[0]
                
                # 显示输入数据
                st.subheader("样本数据")
                st.write(sample_data)
                
                if st.button("预测"):
                    try:
                        # 数据预处理
                        processor = DataProcessor()
                        X = processor.preprocess_single_sample(sample_data)
                        
                        # 加载模型并预测
                        model = load_model(model_file, X.shape[1])
                        predicted_price = predict_price(model, X)
                        
                        # 如果有缩放参数，进行反向转换
                        scaling_params_path = os.path.join(os.path.dirname(current_dir), "models", "scaling_params.npy")
                        if os.path.exists(scaling_params_path):
                            scaling_params = processor.load_scaling_params(scaling_params_path)
                            predicted_price = predicted_price * scaling_params['y_std'] + scaling_params['y_mean']
                        
                        st.success(f"预测价格：{predicted_price:.2f}")
                    except Exception as e:
                        st.error(f"预测过程中发生错误: {str(e)}")
            else:
                st.error(f"找不到测试数据文件：{test_file}")

if __name__ == "__main__":
    main() 