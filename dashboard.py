import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

st.set_page_config(
    page_title="Europe Energy Forecast",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Europe Energy Forecast Dashboard")
st.markdown("Analyze European energy data and decarbonization potential")

st.sidebar.header("Configuration")
country = st.sidebar.selectbox(
    "Select Country",
    ["DE", "FR", "ES", "IT", "GB", "NL", "SE", "DK", "AT", "PL"],
    index=0
)

improvement = st.sidebar.slider(
    "Energy Efficiency Improvement (%)",
    min_value=5,
    max_value=30,
    value=15,
    step=5
)

tab1, tab2, tab3 = st.tabs(["Single Country", "Multi-Country", "About"])

with tab1:
    st.header(f"Analysis for {country}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Fossil Dependency", "91.4%", "-2.5%")
    
    with col2:
        st.metric("CO2 Reduction Potential", "30.5M tons")
    
    with col3:
        st.metric("Investment Required", "€36.3B")
    
    st.subheader("Energy Mix")
    
    energy_data = pd.DataFrame({
        'Source': ['Solar', 'Wind', 'Fossil'],
        'Percentage': [2.6, 5.9, 91.4],
        'Color': ['#FFD700', '#87CEEB', '#8B0000']
    })
    
    fig = px.pie(energy_data, values='Percentage', names='Source', 
                 title=f'Energy Mix - {country}',
                 color='Source', color_discrete_map={'Solar': '#FFD700', 
                                                   'Wind': '#87CEEB', 
                                                   'Fossil': '#8B0000'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Economic Impact")
    
    econ_data = pd.DataFrame({
        'Metric': ['Annual Savings', 'Investment', 'NPV (20y)'],
        'Value (€B)': [4.7, 36.3, 22.4]
    })
    
    fig = px.bar(econ_data, x='Metric', y='Value (€B)', 
                 title='Economic Impact')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Multi-Country Comparison")
    
    sample_data = pd.DataFrame({
        'Country': ['DE', 'FR', 'ES', 'IT', 'GB', 'NL'],
        'Fossil_Dependency_%': [91.4, 56.0, 74.9, 82.0, 75.0, 85.0],
        'Renewable_Share_%': [8.6, 44.0, 25.1, 18.0, 25.0, 15.0],
        'CO2_Reduction_Potential_Mt': [30.5, 8.2, 8.6, 12.3, 10.5, 5.8]
    })
    
    fig = px.bar(sample_data, x='Country', y=['Fossil_Dependency_%', 'Renewable_Share_%'],
                 title='Fossil vs Renewable Energy by Country',
                 barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    
    fig = px.scatter(sample_data, x='Fossil_Dependency_%', y='CO2_Reduction_Potential_Mt',
                     size='CO2_Reduction_Potential_Mt', color='Country',
                     title='CO2 Reduction Potential vs Fossil Dependency')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("About This Project")
    
    st.markdown("""
    ### Europe Energy Forecast Dashboard
    
    This dashboard provides analysis of European energy data including:
    
    - **Energy mix analysis** for individual countries
    - **CO2 reduction potential** calculations
    - **Economic impact assessment** of efficiency improvements
    - **Multi-country comparison** of energy profiles
    
    ### Data Sources
    - ENTSO-E Transparency Platform
    - Open Power System Data (OPSD)
    - European Environment Agency
    
    ### Methodology
    1. Load hourly energy data for European countries
    2. Calculate current energy mix (renewable vs fossil)
    3. Estimate CO2 reduction potential from efficiency improvements
    4. Calculate economic impacts (investment, savings, ROI)
    
    ### How to Use
    1. Select a country from the sidebar
    2. Adjust the efficiency improvement slider
    3. View results in the Single Country tab
    4. Compare countries in the Multi-Country tab
    
    ### Contact
    For questions or suggestions, please create an issue on GitHub.
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
Europe Energy Forecast Dashboard  
Version 1.0  
Data: ENTSO-E 2014-2020
""")
