import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.caption("영화 데이터를 바탕으로 비슷한 특징을 가진 영화끼리 세 유형으로 묶어 봅니다.")

FEATURES = {
    "스크린 수 (로그)": "log_first_scrn",
    "누적 관객 (로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_index",
}
ORIGINAL_FEATURE_NAMES = {
    "log_first_scrn": "스크린 수 (원래 단위)",
    "log_total_audi": "누적 관객 (원래 단위)",
    "days_in_top10": "10위권 일수",
    "long_run_index": "롱런 지수",
}
CLUSTER_NAMES = ["㉮", "㉯", "㉰"]

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    required = [
        "movieCd", "movieNm", "openDt", "genre", "nation", "first_scrn",
        "first_show", "first_date", "peak", "first_week_audi", "total_audi",
        "days_in_top10"
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("데이터에 필요한 열이 없습니다: " + ", ".join(missing))

    numeric_cols = ["first_scrn", "first_week_audi", "total_audi", "days_in_top10"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 네 가지 분류 속성 중 값이 없거나 첫 주 관객이 0 이하인 영화는 제외
    df["long_run_index"] = df["total_audi"] / df["first_week_audi"].where(df["first_week_audi"] > 0)
    df["long_run_index"] = df["long_run_index"].clip(upper=20)
    valid = (
        df["first_scrn"].notna()
        & df["total_audi"].notna()
        & df["days_in_top10"].notna()
        & df["long_run_index"].notna()
        & (df["first_scrn"] > 0)
        & (df["total_audi"] > 0)
    )
    df = df.loc[valid].copy()
    df["log_first_scrn"] = df["first_scrn"].apply(lambda x: __import__("math").log10(x))
    df["log_total_audi"] = df["total_audi"].apply(lambda x: __import__("math").log10(x))
    df["movieNm"] = df["movieNm"].fillna("(제목 없음)").astype(str)
    return df, len(pd.read_csv(DATA_URL, encoding="utf-8"))

try:
    df, total_count = load_data()
except Exception as e:
    st.error("영화 데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()

if df.empty:
    st.warning("조건에 맞는 영화가 없습니다.")
    st.stop()

st.write(f"**전체 영화 {total_count:,}편 · 분류에 사용한 영화 {len(df):,}편**")

st.subheader("1. 유형을 나눌 속성 선택")
selected_labels = st.multiselect(
    "K-평균 분류에 사용할 속성 (두 개 이상 선택)",
    options=list(FEATURES.keys()),
    default=list(FEATURES.keys()),
    help="스크린 수와 누적 관객은 상용로그로 바꿔 사용합니다. 롱런 지수는 최대 20으로 제한합니다."
)
selected_features = [FEATURES[label] for label in selected_labels]

if len(selected_features) < 2:
    st.info("K-평균으로 유형을 나누려면 속성을 두 개 이상 선택해 주세요.")
    st.stop()

scaler = StandardScaler()
scaled = scaler.fit_transform(df[selected_features])
model = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster_raw"] = model.fit_predict(scaled)

# 누적 관객 평균이 큰 그룹부터 ㉮, ㉯, ㉰ 부여
order = (
    df.groupby("cluster_raw")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index
    .tolist()
)
cluster_label_map = {raw: CLUSTER_NAMES[i] for i, raw in enumerate(order)}
df["유형"] = df["cluster_raw"].map(cluster_label_map)

st.caption("스크린 수와 누적 관객은 로그 변환한 값으로 분류하지만, 아래 통계표에는 원래 단위의 평균을 표시합니다.")

st.subheader("2. 2차원 산점도")
plot_col1, plot_col2 = st.columns(2)
with plot_col1:
    x_label = st.selectbox("가로축 속성", options=list(FEATURES.keys()), index=0, key="scatter_x")
with plot_col2:
    y_default = 1 if len(FEATURES) > 1 else 0
    y_label = st.selectbox("세로축 속성", options=list(FEATURES.keys()), index=y_default, key="scatter_y")

fig2 = px.scatter(
    df,
    x=FEATURES[x_label],
    y=FEATURES[y_label],
    color="유형",
    hover_name="movieNm",
    hover_data={"유형": True, FEATURES[x_label]: ":.3f", FEATURES[y_label]: ":.3f"},
    category_orders={"유형": CLUSTER_NAMES},
    labels={FEATURES[x_label]: x_label, FEATURES[y_label]: y_label},
    title="영화 유형 분포"
)
fig2.update_traces(marker={"size": 7, "opacity": 0.78})
fig2.update_layout(legend_title_text="영화 유형", margin=dict(l=10, r=10, t=55, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.subheader("3. 3차원 산점도")
if len(selected_features) < 3:
    st.info("3차원 산점도를 보려면 위에서 분류 속성을 세 개 이상 선택해 주세요.")
else:
    axis_options = list(FEATURES.keys())
    c1, c2, c3 = st.columns(3)
    with c1:
        x3_label = st.selectbox("X축", axis_options, index=0, key="x3")
    with c2:
        y3_label = st.selectbox("Y축", axis_options, index=1, key="y3")
    with c3:
        z3_label = st.selectbox("Z축", axis_options, index=2, key="z3")

    fig3 = px.scatter_3d(
        df,
        x=FEATURES[x3_label],
        y=FEATURES[y3_label],
        z=FEATURES[z3_label],
        color="유형",
        hover_name="movieNm",
        hover_data={"유형": True},
        category_orders={"유형": CLUSTER_NAMES},
        labels={
            FEATURES[x3_label]: x3_label,
            FEATURES[y3_label]: y3_label,
            FEATURES[z3_label]: z3_label,
        },
        title="영화 유형의 3차원 분포"
    )
    fig3.update_traces(marker={"size": 3, "opacity": 0.8})
    fig3.update_layout(legend_title_text="영화 유형", margin=dict(l=0, r=0, t=55, b=0))
    st.plotly_chart(fig3, use_container_width=True)

st.subheader("4. 유형별 편수와 평균")
summary = (
    df.groupby("유형")
    .agg(
        편수=("movieNm", "size"),
        스크린_수=("first_scrn", "mean"),
        누적_관객=("total_audi", "mean"),
        십위권_일수=("days_in_top10", "mean"),
        롱런_지수=("long_run_index", "mean"),
    )
    .reindex(CLUSTER_NAMES)
    .reset_index()
    .rename(columns={
        "유형": "영화 유형",
        "스크린_수": "스크린 수 평균",
        "누적_관객": "누적 관객 평균",
        "십위권_일수": "10위권 일수 평균",
        "롱런_지수": "롱런 지수 평균",
    })
)
summary["편수"] = summary["편수"].astype(int)
summary["스크린 수 평균"] = summary["스크린 수 평균"].round(1)
summary["누적 관객 평균"] = summary["누적 관객 평균"].round(0).astype("Int64")
summary["10위권 일수 평균"] = summary["10위권 일수 평균"].round(1)
summary["롱런 지수 평균"] = summary["롱런 지수 평균"].round(2)
st.dataframe(summary, use_container_width=True, hide_index=True)

st.subheader("5. 유형별 누적 관객 상위 영화 5편")
for label in CLUSTER_NAMES:
    group = df[df["유형"] == label].sort_values("total_audi", ascending=False).head(5)
    with st.expander(f"{label} 유형 · 누적 관객 상위 영화", expanded=True):
        if group.empty:
            st.write("해당 유형에 영화가 없습니다.")
        else:
            display = group[["movieNm", "total_audi", "first_week_audi", "days_in_top10"]].copy()
            display = display.rename(columns={
                "movieNm": "영화 제목",
                "total_audi": "누적 관객",
                "first_week_audi": "첫 주 관객",
                "days_in_top10": "10위권 일수",
            })
            display["누적 관객"] = display["누적 관객"].map(lambda x: f"{x:,.0f}")
            display["첫 주 관객"] = display["첫 주 관객"].map(lambda x: f"{x:,.0f}")
            st.dataframe(display, use_container_width=True, hide_index=True)

st.caption("참고: K-평균은 선택한 속성과 표준화 방식에 따라 유형이 달라질 수 있습니다. ㉮·㉯·㉰는 매 실행 시 누적 관객 평균 순서로 정해집니다.")
