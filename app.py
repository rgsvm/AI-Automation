import os
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="AI Workforce Dashboard", page_icon="", layout="wide")

DATA_URL = "global_ai_workforce_automation_2015_2025.csv"

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def safe_unique_list(df: pd.DataFrame, col: str):
    if col in df.columns:
        return sorted(df[col].dropna().astype(str).unique().tolist())
    return []

def main():
    st.title("AI Workforce Dashboard")

    try:
        df = load_data(DATA_URL)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    required_cols = [
        "Year",
        "Country",
        "AI_Investment_BillionUSD",
        "Automation_Rate_Percent",
        "Employment_Rate_Percent",
        "Average_Salary_USD",
        "Productivity_Index",
        "Reskilling_Investment_MillionUSD",
        "AI_Policy_Index",
        "Job_Displacement_Million",
        "Job_Creation_Million",
        "AI_Readiness_Score",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        st.stop()

    st.sidebar.header("Filters")
    country = st.sidebar.selectbox("Country", ["All"] + safe_unique_list(df, "Country"))
    year = st.sidebar.selectbox("Year", ["All"] + safe_unique_list(df, "Year"))

    filtered = df.copy()
    if country != "All" and "Country" in filtered.columns:
        filtered = filtered[filtered["Country"].astype(str) == country]
    if year != "All" and "Year" in filtered.columns:
        filtered = filtered[filtered["Year"].astype(str) == year]

    tab1, tab2, tab3 = st.tabs(["Dashboard", "Data Table", "Summary"])

    with tab1:
        left, right = st.columns(2)

        with left:
            plot_df = filtered.copy()
            plot_df["AI_Readiness_Score"] = pd.to_numeric(plot_df["AI_Readiness_Score"], errors="coerce")
            plot_df = plot_df.dropna(subset=["AI_Readiness_Score"]).sort_values("AI_Readiness_Score", ascending=False).head(10)

            if len(plot_df) > 0:
                fig1 = px.bar(
                    plot_df,
                    x="Country",
                    y="AI_Readiness_Score",
                    color="Country",
                    title="Top AI Readiness Scores",
                )
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("No valid AI readiness scores available for charting.")

        with right:
            plot_df = filtered.copy()
            plot_df["Automation_Rate_Percent"] = pd.to_numeric(plot_df["Automation_Rate_Percent"], errors="coerce")
            plot_df = plot_df.dropna(subset=["Automation_Rate_Percent"])

            if len(plot_df) > 0:
                fig2 = px.line(
                    plot_df,
                    x="Year",
                    y="Automation_Rate_Percent",
                    color="Country",
                    title="Automation Rate Over Time",
                    markers=True,
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No valid automation rate values available for charting.")

        bottom_left, bottom_right = st.columns(2)

        with bottom_left:
            plot_df = filtered.copy()
            plot_df["Average_Salary_USD"] = pd.to_numeric(plot_df["Average_Salary_USD"], errors="coerce")
            plot_df = plot_df.dropna(subset=["Average_Salary_USD"])

            if len(plot_df) > 0:
                fig3 = px.box(
                    plot_df,
                    x="Country",
                    y="Average_Salary_USD",
                    title="Average Salary by Country",
                )
                st.plotly_chart(fig3, use_container_width=True)

        with bottom_right:
            plot_df = filtered.copy()
            plot_df["Reskilling_Investment_MillionUSD"] = pd.to_numeric(plot_df["Reskilling_Investment_MillionUSD"], errors="coerce")
            plot_df = plot_df.dropna(subset=["Reskilling_Investment_MillionUSD"])

            if len(plot_df) > 0:
                fig4 = px.pie(
                    plot_df,
                    names="Country",
                    values="Reskilling_Investment_MillionUSD",
                    title="Reskilling Investment Share",
                )
                st.plotly_chart(fig4, use_container_width=True)

        st.subheader("AI Insights")
        st.write("This dashboard shows how automation, readiness, salaries, and reskilling trends compare across countries and years.")

    with tab2:
        st.subheader("Filtered Data")
        st.dataframe(filtered, use_container_width=True)

        st.download_button(
            label="Download filtered data as CSV",
            data=convert_df_to_csv(filtered),
            file_name="filtered_ai_workforce_data.csv",
            mime="text/csv",
        )

    with tab3:
        st.subheader("Dataset Summary")
        st.write(f"Rows: {len(df)}")
        st.write(f"Columns: {len(df.columns)}")
        st.dataframe(df.describe(include="all"), use_container_width=True)

if __name__ == "__main__":
    main()
