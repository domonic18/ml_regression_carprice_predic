from ydata_profiling import ProfileReport
import pandas as pd
import streamlit as st


st.sidebar.header("数据上传")
uploaded_file = st.sidebar.file_uploader("选择CSV文件", type="csv")

if uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    df = pd.read_csv(uploaded_file, delimiter='\s+')
    st.dataframe(df)
    profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)

    # Generate the report HTML
    report_html = profile.to_html()

    st.subheader("Pandas Profiling Report")
    # Adjust the width and height to best fit your screen
    st.components.v1.html(report_html, width=1000, height=550, scrolling=True)