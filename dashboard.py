import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Load data
chain_models_df = pd.read_csv('ChainModels_anom.csv')
selected_features_df = pd.read_csv('SelectedFeatures_anom.csv')
venues_df = pd.read_csv('Venues_anom.csv')

# Merge datasets to align Venue data with Chain data
merged_df = venues_df.merge(chain_models_df, on=['Model ID', 'Experiment instance'])

# Calculate performance metrics for each chain
chain_performance = merged_df.groupby('Chain ID').agg(
    mean_absolute_error=('True Value', lambda x: (x - merged_df.loc[x.index, 'Model Prediction']).abs().mean()),
    mean_squared_error=('True Value', lambda x: ((x - merged_df.loc[x.index, 'Model Prediction']) ** 2).mean()),
    mean_absolute_percentage_error=('True Value', lambda x: ((x - merged_df.loc[x.index, 'Model Prediction']).abs() / x).mean() * 100),
    avg_true_value=('True Value', 'mean'),
    total_venues=('Venue ID', 'count')
).reset_index()

# Default MAPE threshold
default_mape_threshold = 20

# Streamlit dashboard
st.title("Chain Performance Dashboard")

# Full list of chain ids
combined_chain_ids = pd.concat([chain_performance['Chain ID'], selected_features_df['Chain ID']], ignore_index=True)

# Convert to a Series
combined_chain_ids_series = pd.Series(combined_chain_ids)

# Select chain ID
selected_chain_id = st.selectbox("Select Chain ID", combined_chain_ids_series)

# Set MAPE threshold
mape_threshold = st.slider("Set MAPE Threshold (%)", min_value=0, max_value=100, value=default_mape_threshold, step=1)

try:
    # Determine pass/fail criteria based on MAPE threshold
    chain_performance['pass_fail'] = chain_performance['mean_absolute_percentage_error'] <= mape_threshold

    # Display chain-level performance
    chain_info = chain_performance[chain_performance['Chain ID'] == selected_chain_id]

    # Determine the color based on Pass/Fail status
    color = 'green' if chain_info['pass_fail'].values[0] else 'red'

    # Visualization using Matplotlib
    fig, ax1 = plt.subplots(figsize=(12, 6))  # Make the graph wider

    # Plot MAPE
    ax1.boxplot(chain_performance['mean_absolute_percentage_error'], positions=[1], widths=0.6)
    ax1.scatter(1, chain_info['mean_absolute_percentage_error'], color=color, label='Selected Chain MAPE')
    ax1.axhline(y=mape_threshold, color='orange', linestyle='--', label='MAPE Threshold')
    ax1.set_xlabel('Chains')
    ax1.set_ylabel('Mean Absolute Percentage Error')
    ax1.set_title('Comparison of MAPE Across Chains')

    # Show pass/fail status
    st.subheader(selected_chain_id)
    st.subheader(f"Pass/Fail Status: {'***Pass***' if chain_info['pass_fail'].values[0] else '***Fail***'}")

    # Display experiment instance
    experiment_instance = chain_models_df[chain_models_df['Chain ID'] == 'f22f409d47117348d9351c1163571026']['Experiment instance'].iloc[0]
    st.write(f"Experiment instance: ***{experiment_instance}***")
    
    # Plot average true value on secondary y-axis
    ax2 = ax1.twinx()
    ax2.scatter(1, chain_info['avg_true_value'], color='blue', label='Selected Chain Avg True Value')
    ax2.set_ylabel('Average True Value')

    # Add legends
    fig.legend(loc='upper right')

    # Add annotations
    for i, row in chain_info.iterrows():
        ax1.annotate(f'{row["mean_absolute_percentage_error"]:.2f}%', (1, row['mean_absolute_percentage_error']),
                     xytext=(5, 5), textcoords='offset points', color=color)
        ax1.annotate(f'MAE: {row["mean_absolute_error"]:.2f}', (1.1, row['mean_absolute_percentage_error']),
                     xytext=(5, 5), textcoords='offset points', color=color)
        ax2.annotate(f'{row["avg_true_value"]:.2f}', (2, row['avg_true_value']),
                     xytext=(5, 5), textcoords='offset points', color=color)
        ax2.annotate(f'Avg. true value: {row["avg_true_value"]:1.2f}', (1.1, row["avg_true_value"]),
                     xytext=(5, 5), textcoords='offset points', color='blue')

    # Display the plot
    st.pyplot(fig)

    # Pass / fail criteria explanation
    st.write(f"***Here's why it {'Passed' if chain_info['pass_fail'].values[0] else 'Failed'}:***")
    st.write(f"The selected chain has a Mean Absolute Percentage Error (MAPE) of **{chain_info['mean_absolute_percentage_error'].values[0]:.2f}%**, a Mean Absolute Error of **{chain_info['mean_absolute_error'].values[0]:.2f}**, and an Average True Value of **{chain_info['avg_true_value'].values[0]:.2f}**.")
    st.write(f"Note: The MAPE threshold is **{mape_threshold}%**. Chains below this threshold are considered to have a good performance.")

    # Display chain performance metrics
    st.write("### Chain Performance Metrics")
    st.write(chain_info)

except:
    # Show selected_chain_id
    st.subheader(selected_chain_id)
    
    # Display experiment instance
    experiment_instance = chain_models_df[chain_models_df['Chain ID'] == 'f22f409d47117348d9351c1163571026']['Experiment instance'].iloc[0]

    # Display model and feature details
    # Filter the dataframe for the selected Chain ID
    model_info = chain_models_df[chain_models_df['Chain ID'] == selected_chain_id]
    feature_info = selected_features_df[selected_features_df['Chain ID'] == selected_chain_id]

    st.write(f"Experiment instance: ***{experiment_instance}***")
    st.write("### Model Information")
    st.write(model_info)

    st.write("### Feature Information")
    st.write(feature_info)

    if feature_info.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f'No data for Chain ID: {selected_chain_id}', horizontalalignment='center', verticalalignment='center', fontsize=12, color='red')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title('Correlation Heatmap')
        st.pyplot(fig)
    else:
        # Plotting the data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(feature_info['Feature Used'], feature_info['Correlation to Ground Truth'], color='skyblue')

        # Adding title and labels
        ax.set_title('Feature Correlation to Ground Truth', fontsize=16)
        ax.set_xlabel('Feature Used', fontsize=14)
        ax.set_ylabel('Correlation to Ground Truth', fontsize=14)

        # Displaying the correlation value on top of the bar
        for index, value in enumerate(feature_info['Correlation to Ground Truth']):
            ax.text(index, value + 0.01, f"{value:.2f}", ha='center', fontsize=12)

        st.pyplot(fig)
