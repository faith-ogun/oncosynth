import streamlit as st
import pandas as pd
import os
import json
from oncosynth.common import run_research
import threading
import time

# Page config
st.set_page_config(
    page_title="OncoSynth",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 OncoSynth: Synthetic Lethality Reporter")
st.write("Generate structured synthetic lethality reports from gene pairs")

# Sidebar for API keys
st.sidebar.header("⚙️ Configuration")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
entrez_email = st.sidebar.text_input("NCBI Entrez Email")

if openai_key and entrez_email:
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["ENTREZ_EMAIL"] = entrez_email

# Main interface
tab1, tab2 = st.tabs(["🔬 Single Pair", "📊 Batch Analysis"])

with tab1:
    st.header("Single Gene Pair Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        biomarker = st.text_input("Biomarker Gene", placeholder="e.g., BRCA1").upper()
    with col2:
        target = st.text_input("Target Gene", placeholder="e.g., PARP1").upper()
    
    if st.button("🚀 Generate Report", type="primary"):
        if biomarker and target and openai_key and entrez_email:
            with st.spinner(f"Analyzing {biomarker} - {target}..."):
                try:
                    run_research(biomarker, target)
                    st.success("✅ Report generated successfully!")
                    
                    # Display the report
                    report_path = f"oncosynth/reports/{biomarker}_{target}_report.md"
                    if os.path.exists(report_path):
                        with open(report_path, 'r') as f:
                            report_content = f.read()
                        
                        # Add download button
                        st.download_button(
                            label="📥 Download Report",
                            data=report_content,
                            file_name=f"{biomarker}_{target}_report.md",
                            mime="text/markdown"
                        )
                        
                        st.markdown(report_content)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please fill in all fields")

with tab2:
    st.header("Batch Analysis")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())
        
        if st.button("🚀 Run Batch Analysis", type="primary"):
            if openai_key and entrez_email:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, row in df.iterrows():
                    biomarker = str(row.iloc[0]).strip().upper()
                    target = str(row.iloc[1]).strip().upper()
                    
                    status_text.text(f"Processing {biomarker} - {target} ({idx+1}/{len(df)})")
                    
                    try:
                        run_research(biomarker, target)
                    except Exception as e:
                        st.error(f"Failed for {biomarker}-{target}: {e}")
                    
                    progress_bar.progress((idx + 1) / len(df))
                
                st.success("✅ Batch analysis complete!")
                # Add bulk download option
                reports_dir = "oncosynth/reports"
                if os.path.exists(reports_dir):
                    reports = [f for f in os.listdir(reports_dir) if f.endswith('.md')]

                    if reports:
                        import zipfile
                        import io

                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            for report_file in reports:
                                report_path = os.path.join(reports_dir, report_file)
                                with open(report_path, 'r') as f:
                                    zip_file.writestr(report_file, f.read())

                        st.download_button(
                            label="📦 Download All Reports (ZIP)",
                            data=zip_buffer.getvalue(),
                            file_name="oncosynth_batch_reports.zip",
                            mime="application/zip"
                        )
            else:
                st.warning("⚠️ Please configure API credentials")

# Results browser
st.header("📄 Generated Reports")
reports_dir = "oncosynth/reports"
if os.path.exists(reports_dir):
    reports = [f for f in os.listdir(reports_dir) if f.endswith('.md')]
    
    if reports:
        selected_report = st.selectbox("Select a report to view:", reports)
        
        if selected_report:
            report_path = os.path.join(reports_dir, selected_report)
            with open(report_path, 'r') as f:
                content = f.read()

            # Add download button
            col1, col2 = st.columns([1, 4])
            with col1:
                st.download_button(
                    label="📥 Download",
                    data=content,
                    file_name=selected_report,
                    mime="text/markdown"
                )

            st.markdown(content)
    else:
        st.info("No reports generated yet")