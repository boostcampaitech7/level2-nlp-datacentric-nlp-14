import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from streamlit_option_menu import option_menu

# 페이지 기본 설정
st.set_page_config(page_title="Data Analysis Dashboard", layout="wide", page_icon="📊")

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

    # 세션 유지 위한 세부 탭 설정
    tab1, tab2, tab3 = st.tabs(["전체적인 진단", "Noise 위주 검사", "Label Error 검사"])

    if uploaded_file:
        # 데이터 읽기
        st.session_state["data"] = pd.read_csv(uploaded_file)

        # 전체적인 진단 탭
        with tab1:
            st.title("📊 전체적인 진단")
            if st.session_state["data"] is not None:
                df = st.session_state["data"]

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

            else:
                st.write("Please upload a CSV file to view the analytics.")

        # Noise 위주 검사 탭
        with tab2:
            st.title("🔍 Noise 위주 검사")
            if st.session_state["data"] is not None:
                df = st.session_state["data"]

                # 노이즈 비율에 따른 유사도 산점도
                st.subheader("Similarity Score by Noise Ratio")
                fig3 = px.scatter(
                    df, x="noise_ratio", y="sim_score", color="target", title="Similarity Score by Noise Ratio"
                )
                st.plotly_chart(fig3, use_container_width=True)

                # 노이즈 복구 성능 확인
                noise_df = df[df["noise_label"] == True]
                restored_noise = noise_df[noise_df["sim_score"] >= 0.9]
                restored_percentage = (len(restored_noise) / len(noise_df)) * 100 if len(noise_df) > 0 else 0
                st.metric("Noise Restoration Success Rate", f"{restored_percentage:.2f}%")

            else:
                st.write("Please upload a CSV file to view the noise analysis.")

        # Label Error 검사 탭
        with tab3:
            st.title("🔍 Label Error 검사")
            if st.session_state["data"] is not None:
                df = st.session_state["data"]

                # 예를 들어, 라벨 에러 분석을 위해 클린 데이터 기반의 라벨링을 적용하는 부분
                clean_df = df[df["sim_score"] >= 0.9]  # 유사도가 높은 클린 데이터 필터링

                # K-means 클러스터링을 통해 라벨 에러를 자체 탐지 (예시로 작성, 실제 클러스터링 코드 필요)
                # from sklearn.cluster import KMeans
                # 클린 데이터에서 클러스터링 예시 (KMeans 클러스터링 코드를 적용할 경우)
                # embedding_matrix = ... # Sentence embedding 생성 필요
                # kmeans = KMeans(n_clusters=df['target'].nunique())
                # clusters = kmeans.fit_predict(embedding_matrix)

                # 임의로 target과 clustering 결과 비교하는 코드 부분
                # st.write("라벨 에러 탐지 결과 (예시)")
                # 라벨 에러가 있는 데이터 필터링 후 시각화

                st.write("라벨 에러 탐지를 위한 클린 데이터 클러스터링을 통해 얻은 결과를 표시할 수 있습니다.")

            else:
                st.write("Please upload a CSV file to view the label error analysis.")

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
