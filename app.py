import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import time

# URL of your deployed AWS Lambda batch endpoint
API_URL = "https://jeraevxv6d4mjcsb22ovqlzg3y0lrmtm.lambda-url.us-east-1.on.aws/predict_csv"

# 1. Load reference data (cached for performance)
@st.cache_data
def load_reference_data():
    meal_info = pd.read_csv('data/meal_info.csv')
    center_info = pd.read_csv('data/fulfilment_center_info.csv')
    return meal_info, center_info

st.set_page_config(page_title="Food Demand Predictor", layout="wide")

st.title("🍱 Food Demand Forecasting Dashboard")
st.markdown("""
This dashboard merges raw transaction data with metadata and calls a containerized ML model on AWS Lambda.
**Batch processing is enabled** to handle large files without exceeding AWS payload limits.
""")

# Load references
try:
    meal_info, center_info = load_reference_data()
    st.sidebar.success("✅ Reference Data Loaded")
except Exception as e:
    st.sidebar.error("⚠️ Ensure 'meal_info.csv' and 'fulfilment_center_info.csv' are in the folder.")
    st.stop()

# 2. Input Selection
st.subheader("Step 1: Select Data Source")
data_source = st.radio("Choose how to provide data:", ("Upload CSV File", "Use Sample Data (50 rows)"))

test_df = None

if data_source == "Upload CSV File":
    uploaded_file = st.file_uploader("Upload 'test_QoiMO9B.csv'", type=["csv"])
    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)
elif data_source == "Use Sample Data (50 rows)":
    if st.button("Load 50 Sample Rows"):
        try:
            # Load the full test set and sample 50 rows
            full_test_df = pd.read_csv('data/test_QoiMO9B.csv')
            test_df = full_test_df.sample(n=50, random_state=42)  # Random_state for reproducibility
            st.success("✅ Loaded 50 random rows from 'data/test_QoiMO9B.csv'")
        except FileNotFoundError:
            st.error("❌ Test data file 'data/test_QoiMO9B.csv' not found locally.")
        except Exception as e:
            st.error(f"❌ Error loading sample data: {e}")

# 3. Processing & Prediction
if test_df is not None:
    # Data Processing: Merge and Drop
    enriched_df = test_df.merge(meal_info, on='meal_id', how='left')
    enriched_df = enriched_df.merge(center_info, on='center_id', how='left')
    
    if 'id' in enriched_df.columns:
        enriched_df = enriched_df.drop(columns=['id'])
    
    st.write(f"### Data Ready for Inference ({len(enriched_df)} rows)")
    st.dataframe(enriched_df.head(5))

    # 4. Trigger Batch Predictions
    if st.button("🚀 Generate Demand Forecast"):
        payload_list = enriched_df.to_dict(orient='records')
        
        # --- BATCHING LOGIC ---
        chunk_size = 50  # Sending 50 rows at a time to stay under 6MB limit
        all_predictions = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        
        try:
            for i in range(0, len(payload_list), chunk_size):
                chunk = payload_list[i : i + chunk_size]
                
                # Update UI progress
                percent = min(100, int((i / len(payload_list)) * 100))
                progress_bar.progress(percent)
                status_text.text(f"Processing batch {int(i/chunk_size) + 1}... (Rows {i} to {i+len(chunk)})")
                
                # Call AWS API
                response = requests.post(API_URL, json=chunk)
                
                if response.status_code == 200:
                    batch_results = response.json().get('predictions', [])
                    all_predictions.extend([res['predicted_orders'] for res in batch_results])
                else:
                    st.error(f"Failed at batch starting row {i}. Error: {response.status_code}")
                    st.stop()
            
            progress_bar.progress(100)
            status_text.text("✅ All batches processed successfully!")
            
            # Attach results back to original dataframe
            enriched_df['Predicted_Orders'] = all_predictions
            
            # --- DASHBOARD VISUALS ---
            st.divider()
            total_time = time.time() - start_time
            st.info(f"Total processing time: {total_time:.2f} seconds")

            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Forecasted Results")
                st.dataframe(enriched_df[['week', 'center_id', 'meal_id', 'category', 'Predicted_Orders']])
            
            with col2:
                st.subheader("🍕 Demand by Category")
                # Grouping for visual clarity
                cat_chart = enriched_df.groupby('category')['Predicted_Orders'].sum().reset_index()
                fig = px.bar(cat_chart, x='category', y='Predicted_Orders', color='category')
                st.plotly_chart(fig, use_container_width=True)

            # Optional: Download button for results
            csv_results = enriched_df.to_csv(index=False)
            st.download_button("📥 Download Full Forecast CSV", data=csv_results, file_name="forecast_results.csv")

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

# Sidebar logic for clear-down
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.rerun()
