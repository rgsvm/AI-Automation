import streamlit as st
import pandas as pd
import plotly.express as px

# Helper function to get unique values safely
def safe_unique_list(df, column):
    if column in df.columns:
        return df[column].dropna().unique().tolist()
    return []

# Convert DataFrame to CSV
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Generate job report for a single job
def generate_job_report(job_row):
    report = {
        'job_title': job_row.get('job_title', 'N/A'),
        'industry': job_row.get('industry', 'N/A'),
        'risk_score': job_row.get('risk_score', 'N/A'),
        'risk_level': job_row.get('risk_level', 'N/A'),
        'generated_report': f"This job in {job_row.get('industry', 'N/A')} has a risk score of {job_row.get('risk_score', 'N/A')}."
    }
    return report

# Generate reports for all jobs in DataFrame
def generate_reports_for_dataframe(df):
    reports = []
    for _, row in df.iterrows():
        reports.append(generate_job_report(row))
    return pd.DataFrame(reports)
