import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from streamlit_option_menu import option_menu

# 페이지 기본 설정
st.set_page_config(page_title="Noise & Label Error Dashboard", layout="wide", page_icon="📊")

# 사이드바 메뉴 설정
with st.sidebar:
    st.image(
        "pikachu-boxing.gif",
        width=128,
    )  # 로고
    st.title("Data Recovery Analysis")
    selected = option_menu(
        "Main Menu", ["Home", "Compare"], icons=["house", "arrows-expand"], menu_icon="menu", default_index=0
    )

# HOME 탭
if selected == "Home":
    st.title("📊 Data Restoration and Noise Analysis - Home")

    # 단일 CSV 파일 업로드
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file for analysis", type="csv")

    if uploaded_file:
        # 데이터 읽기
        df = pd.read_csv(uploaded_file)

        # 주요 통계 카드
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Average Noise Ratio", f"{df['noise_ratio'].mean():.2f}")
        with col2:
            st.metric("Average Similarity Score", f"{df['sim_score'].mean():.2f}")
        with col3:
            st.metric("Total Data Points", f"{len(df)}")
        with col4:
            st.metric("Total Noise Labels", f"{df['noise_label'].nunique()}")
        with col5:
            st.metric("Target Classes", f"{df['target'].nunique()}")

        # 노이즈 레이블 상태별 target 분포
        st.subheader("Target Distribution by Noise Label")
        fig1 = px.histogram(df, x="target", color="noise_label", title="Target Distribution by Noise Label")
        st.plotly_chart(fig1, use_container_width=True)

        # 유사도 점수 분포 히스토그램
        st.subheader("Similarity Score Distribution")
        fig2, ax = plt.subplots()
        sns.histplot(df["sim_score"], bins=20, kde=True, ax=ax)
        ax.set_title("Similarity Score Distribution")
        st.pyplot(fig2)

        # 노이즈 비율에 따른 유사도 산점도
        st.subheader("Similarity Score by Noise Ratio")
        fig3 = px.scatter(df, x="noise_ratio", y="sim_score", color="target", title="Similarity Score by Noise Ratio")
        st.plotly_chart(fig3, use_container_width=True)

    else:
        st.write("Please upload a CSV file to view the analytics.")

# COMPARE 탭
elif selected == "Compare":
    st.title("📈 Data Restoration Comparison - Compare")

    # 이전 버전과 현재 버전의 CSV 파일 업로드
    previous_file = st.sidebar.file_uploader("Upload a previous CSV file", type="csv", key="previous")
    current_file = st.sidebar.file_uploader("Upload a current CSV file", type="csv", key="current")

    if previous_file and current_file:
        # 데이터 읽기
        prev_df = pd.read_csv(previous_file)
        curr_df = pd.read_csv(current_file)

        # 두 데이터셋 비교: 노이즈 비율과 유사도 평균 변화
        prev_noise_ratio = prev_df["noise_ratio"].mean()
        curr_noise_ratio = curr_df["noise_ratio"].mean()
        prev_sim_score = prev_df["sim_score"].mean()
        curr_sim_score = curr_df["sim_score"].mean()

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Noise Ratio Change", f"{curr_noise_ratio:.2f}", delta=f"{curr_noise_ratio - prev_noise_ratio:.2f}"
            )
        with col2:
            st.metric(
                "Similarity Score Change", f"{curr_sim_score:.2f}", delta=f"{curr_sim_score - prev_sim_score:.2f}"
            )

        # 노이즈 비율 및 유사도 점수 비교 그래프
        st.subheader("Noise Ratio and Similarity Score Comparison")
        comparison_df = pd.DataFrame(
            {
                "Metric": ["Noise Ratio", "Similarity Score"],
                "Previous": [prev_noise_ratio, prev_sim_score],
                "Current": [curr_noise_ratio, curr_sim_score],
            }
        )
        fig4 = px.bar(
            comparison_df,
            x="Metric",
            y=["Previous", "Current"],
            barmode="group",
            title="Metric Comparison: Previous vs Current",
        )
        st.plotly_chart(fig4, use_container_width=True)

        # Target 분포 비교
        st.subheader("Target Distribution Comparison by Noise Label")
        prev_fig = px.histogram(
            prev_df, x="target", color="noise_label", title="Previous Data - Target Distribution by Noise Label"
        )
        curr_fig = px.histogram(
            curr_df, x="target", color="noise_label", title="Current Data - Target Distribution by Noise Label"
        )

        st.write("### Previous Data")
        st.plotly_chart(prev_fig, use_container_width=True)

        st.write("### Current Data")
        st.plotly_chart(curr_fig, use_container_width=True)

    else:
        st.write("Please upload both previous and current CSV files to compare.")
