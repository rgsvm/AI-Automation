import os
import pandas as pd
import streamlit as st
import plotly.express as px
from openai import OpenAI
from openai import RateLimitError, AuthenticationError, APIConnectionError, BadRequestError

st.set_page_config(page_title="AI Job Risk Report Generator", page_icon="", layout="wide")

API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY) if API_KEY else None

DATA_URL = "https://raw.githubusercontent.com/rgsvm/AI-Automation/main/your_data_file.csv"

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

def build_job_prompt(row):
    return f"""You are an AI workforce analyst.

Analyze this job role for AI impact by 2030:

Job title: {row.get('job_title', 'N/A')}
Industry: {row.get('industry', 'N/A')}
Category: {row.get('job_category', 'N/A')}
Risk score: {row.get('risk_score', 'N/A')}
Risk level: {row.get('risk_level', 'N/A')}
Main skills: {row.get('skills', 'N/A')}

Return the response in this exact format:

Summary:
Why this role is at risk:
Skills that reduce risk:
Recommended action plan:
Executive note:
Tags:""".strip()

def make_fallback_report(row, message):
    return {
        "job_title": row.get("job_title", "N/A"),
        "industry": row.get("industry", "N/A"),
        "risk_score": row.get("risk_score", "N/A"),
        "risk_level": row.get("risk_level", "N/A"),
        "generated_report": message,
    }

def generate_job_report(row, model="gpt-4o-mini"):
    if client is None:
        return make_fallback_report(row, "OPENAI_API_KEY is missing.")

    try:
        response = client.responses.create(model=model, input=build_job_prompt(row))
        text = getattr(response, "output_text", "").strip()

        if not text:
            return make_fallback_report(row, "No output returned from the model.")

        return {
            "job_title": row.get("job_title", "N/A"),
            "industry": row.get("industry", "N/A"),
            "risk_score": row.get("risk_score", "N/A"),
            "risk_level": row.get("risk_level", "N/A"),
            "generated_report": text,
        }

    except RateLimitError:
        return make_fallback_report(row, "Rate limit reached. Please try again later.")
    except AuthenticationError:
        return make_fallback_report(row, "Authentication error. Check your API key.")
    except APIConnectionError:
        return make_fallback_report(row, "Connection error. Please check your internet or API status.")
    except BadRequestError as e:
        return make_fallback_report(row, f"Bad request error: {e}")
    except Exception as e:
        return make_fallback_report(row, f"Unexpected error: {e}")

def generate_reports_for_dataframe(df_input):
    reports = []
    for _, row in df_input.iterrows():
        reports.append(generate_job_report(row))
    return pd.DataFrame(reports)

def main():
    st.title("AI Job Risk Report Generator")

    try:
        df = load_data(DATA_URL)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    st.sidebar.header("Filters")
    job_category = st.sidebar.selectbox("Job Category", ["All"] + safe_unique_list(df, "job_category"))
    industry = st.sidebar.selectbox("Industry", ["All"] + safe_unique_list(df, "industry"))
    risk_level = st.sidebar.selectbox("Risk Level", ["All"] + safe_unique_list(df, "risk_level"))
    selected_job = st.sidebar.selectbox("Select Job Title", ["All"] + safe_unique_list(df, "job_title"))
    show_high_risk_only = st.sidebar.checkbox("Show only high-risk jobs", value=False)

    filtered = df.copy()
    if job_category != "All" and "job_category" in filtered.columns:
        filtered = filtered[filtered["job_category"].astype(str) == job_category]
    if industry != "All" and "industry" in filtered.columns:
        filtered = filtered[filtered["industry"].astype(str) == industry]
    if risk_level != "All" and "risk_level" in filtered.columns:
        filtered = filtered[filtered["risk_level"].astype(str) == risk_level]
    if selected_job != "All" and "job_title" in filtered.columns:
        filtered = filtered[filtered["job_title"].astype(str) == selected_job]
    if show_high_risk_only and "risk_level" in filtered.columns:
        filtered = filtered[filtered["risk_level"].astype(str) == "High"]

    tab1, tab2, tab3 = st.tabs(["Dashboard", "Job Report", "Data Table"])

    with tab1:
        left, right = st.columns(2)

        with left:
            if len(filtered) > 0 and "risk_score" in filtered.columns and "job_title" in filtered.columns:
                plot_df = filtered.copy()
                plot_df["risk_score"] = pd.to_numeric(plot_df["risk_score"], errors="coerce")
                plot_df = plot_df.dropna(subset=["risk_score"]).sort_values("risk_score", ascending=False).head(10)

                if len(plot_df) > 0:
                    fig1 = px.bar(
                        plot_df,
                        x="job_title",
                        y="risk_score",
                        color="risk_level" if "risk_level" in plot_df.columns else None,
                        title="Top 10 Highest Risk Jobs",
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                else:
                    st.info("No valid risk scores available for charting.")
            else:
                st.info("No valid risk scores available for charting.")

        with right:
            if len(filtered) > 0 and "risk_level" in filtered.columns:
                fig2 = px.histogram(
                    filtered,
                    x="risk_level",
                    color="risk_level",
                    title="Risk Level Distribution",
                )
                st.plotly_chart(fig2, use_container_width=True)

        bottom_left, bottom_right = st.columns(2)

        with bottom_left:
            if len(filtered) > 0 and "industry" in filtered.columns and "risk_score" in filtered.columns:
                plot_df = filtered.copy()
                plot_df["risk_score"] = pd.to_numeric(plot_df["risk_score"], errors="coerce")
                plot_df = plot_df.dropna(subset=["risk_score"])

                if len(plot_df) > 0:
                    fig3 = px.box(plot_df, x="industry", y="risk_score", title="Risk Score by Industry")
                    st.plotly_chart(fig3, use_container_width=True)

        with bottom_right:
            if len(filtered) > 0 and "skill_category" in filtered.columns:
                fig4 = px.pie(filtered, names="skill_category", title="Skill Category Share")
                st.plotly_chart(fig4, use_container_width=True)

        st.subheader("AI Insights")
        st.write("This dashboard helps identify which roles are more exposed to AI automation and what skills may improve resilience.")

    with tab2:
        st.subheader("AI-Generated Job Report")

        if len(filtered) > 0:
            job_row = filtered.iloc[0]
            report = generate_job_report(job_row)

            st.write(f"**Job Title:** {report['job_title']}")
            st.write(f"**Industry:** {report['industry']}")
            st.write(f"**Risk Score:** {report['risk_score']}")
            st.write(f"**Risk Level:** {report['risk_level']}")

            st.markdown("### Generated Report")
            st.write(report["generated_report"])

            st.markdown("### Export This Report")
            single_report_df = pd.DataFrame([report])

            st.download_button(
                label="Download selected job report as CSV",
                data=convert_df_to_csv(single_report_df),
                file_name="selected_job_report.csv",
                mime="text/csv",
            )
        else:
            st.info("No job selected or no records available.")

    with tab3:
        st.subheader("Filtered Job Data")
        st.dataframe(filtered, use_container_width=True)

        st.download_button(
            label="Download filtered data as CSV",
            data=convert_df_to_csv(filtered),
            file_name="filtered_ai_jobs.csv",
            mime="text/csv",
        )

        if st.button("Generate Reports for All Filtered Jobs"):
            with st.spinner("Generating reports..."):
                report_df = generate_reports_for_dataframe(filtered)

            st.success("Reports generated successfully.")
            st.dataframe(report_df, use_container_width=True)

            st.download_button(
                label="Download all generated reports as CSV",
                data=convert_df_to_csv(report_df),
                file_name="ai_job_reports.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()
