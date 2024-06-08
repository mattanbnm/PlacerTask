import pandas as pd
import streamlit as st

# Load data
chain_models_df = pd.read_csv('ChainModels_anom.csv')
selected_features_df = pd.read_csv('SelectedFeatures_anom.csv')
venues_df = pd.read_csv('Venues_anom.csv')

# Merge datasets to align Venue data with Chain data
merged_df = venues_df.merge(chain_models_df, on=['Model ID', 'Experiment instance'])

# Calculate performance metrics for each chain
chain_performance = merged_df.groupby('Chain ID').agg(
    mean_absolute_error=('True Value', lambda x: (x - merged_df.loc[x.index, 'Model Prediction']).abs().mean()),
    mean_squared_error=('True Value', lambda x: ((x - merged_df.loc[x.index, 'Model Prediction']) ** 2).mean())
).reset_index()

# Define pass/fail criteria
mae_90th_percentile = chain_performance['mean_absolute_error'].quantile(0.90)
mse_90th_percentile = chain_performance['mean_squared_error'].quantile(0.90)

chain_performance['pass_fail'] = (chain_performance['mean_absolute_error'] < mae_90th_percentile) & \
                                 (chain_performance['mean_squared_error'] < mse_90th_percentile)

# Streamlit dashboard
st.title("Chain Performance Dashboard")

# Select chain ID
selected_chain_id = st.selectbox("Select Chain ID", chain_performance['Chain ID'])


# Beeswarm plot for the selected chain
st.write("### Beeswarm Plot of True Values vs Model Predictions")
model_ids = model_info['Model ID'].unique()
if len(model_ids) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.swarmplot(data=merged_df[merged_df['Model ID'].isin(model_ids)], x='Model Prediction', y='True Value', ax=ax)
    plt.title(f'Beeswarm Plot for Chain ID: {selected_chain_id}')
    st.pyplot(fig)
else:
    st.write("No model predictions available for the selected chain.")
    
# Display chain-level performance
chain_info = chain_performance[chain_performance['Chain ID'] == selected_chain_id]
st.write("### Chain Performance Metrics")
st.write(chain_info)

# Display model and feature details
model_info = chain_models_df[chain_models_df['Chain ID'] == selected_chain_id]
feature_info = selected_features_df[selected_features_df['Chain ID'] == selected_chain_id]
st.write("### Model Information")
st.write(model_info)
st.write("### Feature Information")
st.write(feature_info)

# Ensure the necessary columns are available for plotting
if not chain_info.empty and 'mean_absolute_error' in chain_info.columns and 'mean_squared_error' in chain_info.columns:
    st.write("### Performance Metrics Visualization")
    st.bar_chart(chain_info[['mean_absolute_error', 'mean_squared_error']])
    st.write(f"Pass/Fail Status: {'Pass' if chain_info['pass_fail'].values[0] else 'Fail'}")
else:
    st.error("Required columns for visualization are missing or data is not available.")
