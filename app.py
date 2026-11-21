
import pandas as pd
import streamlit as st
import plotly.express as px


st.set_page_config(
    page_title="VeriFact - News Insights",
    layout="wide"
)

# Styling
st.markdown("""
<style>
.block-container {
    max-width: 1400px;
    padding-top: 1.5rem;
}
[data-testid="stMetric"] {
    background-color: #172338;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #263954;
}
</style>
""", unsafe_allow_html=True)


# Load datasets
real = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

real["News Category"] = "Real"
fake["News Category"] = "Fake"

data = pd.concat([real, fake], ignore_index=True)

# Prepare text
data["news_text"] = data["title"].fillna("") + " " + data["text"].fillna("")

# Text features
data["word_count"] = data["news_text"].str.split().str.len()
data["uppercase_count"] = data["news_text"].str.count(r"[A-Z]")
data["exclamation_count"] = data["news_text"].str.count("!")
data["question_count"] = data["news_text"].str.count(r"\?")

# Identify length anomalies
Q1 = data["word_count"].quantile(0.25)
Q3 = data["word_count"].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

data["Length Anomaly"] = (
    (data["word_count"] < lower_limit) |
    (data["word_count"] > upper_limit)
)

# Sensational words
sensational_words = [
    "shocking", "breaking", "urgent", "exposed",
    "unbelievable", "secret", "bombshell", "amazing",
    "must see", "what they don't want you to know"
]

data["Sensational Words"] = data["news_text"].str.lower().apply(
    lambda text: sum(text.count(word) for word in sensational_words)
)

data["Text Anomaly"] = (
    data["Length Anomaly"] |
    (data["exclamation_count"] >= 5) |
    (data["Sensational Words"] >= 2)
)


# Dashboard title
st.title("📰 VeriFact – News Insights Report")
st.caption("Explore news trends and unusual textual patterns in the dataset.")

st.divider()


# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Trends",
    "Text Analysis",
    "Anomalies"
])


# ---------------- OVERVIEW TAB ----------------
with tab1:

    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Articles", f"{len(data):,}")
    col2.metric("Real Articles", f"{len(real):,}")
    col3.metric("Fake Articles", f"{len(fake):,}")

    st.divider()

    st.subheader("News Distribution")

    category_counts = data["News Category"].value_counts().reset_index()
    category_counts.columns = ["Category", "Article Count"]

    fig1 = px.pie(
        category_counts,
        names="Category",
        values="Article Count",
        hole=0.5,
        title="Real and Fake News Distribution"
    )

    fig1.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(fig1, use_container_width=True)


# ---------------- TRENDS TAB ----------------
with tab2:

    st.subheader("News Subject Trends")

    if "subject" in data.columns:

        subject_counts = data.groupby(
            ["subject", "News Category"]
        ).size().reset_index(name="Article Count")

        fig2 = px.bar(
            subject_counts,
            x="subject",
            y="Article Count",
            color="News Category",
            barmode="group",
            title="Article Trends by Subject"
        )

        fig2.update_layout(
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.subheader("Article Length Insights")

    col1, col2 = st.columns(2)

    with col1:

        length_comparison = data.groupby(
            "News Category"
        )["word_count"].mean().reset_index()

        fig3 = px.bar(
            length_comparison,
            x="News Category",
            y="word_count",
            color="News Category",
            title="Average Article Word Count"
        )

        fig3.update_layout(
            template="plotly_dark",
            height=420
        )

        st.plotly_chart(fig3, use_container_width=True)

    with col2:

        fig4 = px.box(
            data,
            x="News Category",
            y="word_count",
            color="News Category",
            title="Article Length Distribution"
        )

        fig4.update_layout(
            template="plotly_dark",
            height=420
        )

        st.plotly_chart(fig4, use_container_width=True)


# ---------------- TEXT ANALYSIS TAB ----------------
with tab3:

    st.subheader("Textual Pattern Comparison")

    pattern_data = data.groupby("News Category")[
        ["uppercase_count", "exclamation_count",
         "question_count", "Sensational Words"]
    ].mean().reset_index()

    pattern_melted = pattern_data.melt(
        id_vars="News Category",
        var_name="Text Pattern",
        value_name="Average Count"
    )

    fig5 = px.bar(
        pattern_melted,
        x="Text Pattern",
        y="Average Count",
        color="News Category",
        barmode="group",
        title="Average Text Patterns in Real vs Fake News"
    )

    fig5.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(fig5, use_container_width=True)

    st.divider()

    st.subheader("Text Pattern Details")

    st.write("Compare average text characteristics between Real and Fake news.")

    st.dataframe(
        pattern_data.rename(columns={
            "uppercase_count": "Uppercase Letters",
            "exclamation_count": "Exclamation Marks",
            "question_count": "Question Marks",
            "Sensational Words": "Sensational Word Count"
        }),
        use_container_width=True
    )


# ---------------- ANOMALIES TAB ----------------
with tab4:

    st.subheader("Textual Anomaly Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Length Anomalies", int(data["Length Anomaly"].sum()))
    col2.metric(
        "Articles with Sensational Words",
        int((data["Sensational Words"] > 0).sum())
    )
    col3.metric("Articles with Text Anomalies", int(data["Text Anomaly"].sum()))

    st.divider()

    st.subheader("Explore Unusual Articles")

    col1, col2 = st.columns(2)

    with col1:
        category_filter = st.selectbox(
            "News Category",
            ["All", "Real", "Fake"]
        )

    with col2:
        anomaly_filter = st.selectbox(
            "Select Pattern",
            [
                "All Articles",
                "Text Anomalies",
                "Length Anomalies",
                "Sensational Words"
            ]
        )

    filtered_data = data.copy()

    if category_filter != "All":
        filtered_data = filtered_data[
            filtered_data["News Category"] == category_filter
        ]

    if anomaly_filter == "Text Anomalies":
        filtered_data = filtered_data[filtered_data["Text Anomaly"]]

    elif anomaly_filter == "Length Anomalies":
        filtered_data = filtered_data[filtered_data["Length Anomaly"]]

    elif anomaly_filter == "Sensational Words":
        filtered_data = filtered_data[
            filtered_data["Sensational Words"] > 0
        ]

    st.metric("Matching Articles", f"{len(filtered_data):,}")

    st.dataframe(
        filtered_data[[
            "title",
            "News Category",
            "word_count",
            "uppercase_count",
            "exclamation_count",
            "Sensational Words",
            "Text Anomaly"
        ]],
        use_container_width=True,
        height=450
    )

    st.download_button(
        "⬇ Download Insights Report",
        filtered_data.to_csv(index=False),
        file_name="verifact_insights_report.csv",
        mime="text/csv"
    )
