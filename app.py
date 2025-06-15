# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
import os
import json
from typing import List, Tuple, Dict, Any
import redis
import yaml
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera
import warnings

warnings.filterwarnings('ignore')

# Redis connection
REDIS_URL = os.getenv('UPSTASH_REDIS_REST_URL')
REDIS_TOKEN = os.getenv('UPSTASH_REDIS_REST_TOKEN')

if REDIS_URL and REDIS_TOKEN:
    r = redis.Redis.from_url(f"redis://default:{REDIS_TOKEN}@{REDIS_URL.replace('https://', '')}")
else:
    r = None  # Fallback for local dev

# Page config
st.set_page_config(page_title="Inverse Edge Lab", layout="wide", initial_sidebar_state="expanded")

# Simple authentication for demo
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔄 Inverse Edge Lab - Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            # Simple hardcoded check for demo
            if username == "admin" and password == "admin123":
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Simulation Controls")
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()
    
    # Bracket pairs
    st.subheader("Bracket Pairs (TP, SL)")
    default_brackets = [(10, 10), (15, 10), (20, 10), (30, 15)]
    
    if 'brackets' not in st.session_state:
        st.session_state.brackets = default_brackets
    
    # Manual entry
    col1, col2 = st.columns(2)
    with col1:
        new_tp = st.number_input("TP", min_value=1, value=20)
    with col2:
        new_sl = st.number_input("SL", min_value=1, value=10)
    
    if st.button("➕ Add Bracket"):
        st.session_state.brackets.append((new_tp, new_sl))
    
    # Display current brackets
    st.write("Current brackets:")
    for i, (tp, sl) in enumerate(st.session_state.brackets):
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.write(f"TP: {tp}")
        with col2:
            st.write(f"SL: {sl}")
        with col3:
            if st.button("❌", key=f"del_{i}"):
                st.session_state.brackets.pop(i)
                st.rerun()
    
    # Risk-reward lock
    rr_lock = st.checkbox("Lock Risk-Reward Ratio")
    if rr_lock:
        rr_ratio = st.slider("R:R Ratio", 0.5, 5.0, 2.0, 0.1)
    
    # Edge metric
    edge_metric = st.selectbox(
        "Optimize for metric:",
        ["Expectancy", "Sharpe Ratio", "Profit Factor", "Hit Rate", "Total P/L"]
    )
    
    # Advanced stats
    advanced_stats = st.checkbox("Run advanced significance tests")
    
    # Run button
    run_sim = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

# Main content
st.title("🔄 Inverse Edge Lab")
st.markdown("*Discover the hidden edge in your trading inversions*")

# File upload
uploaded_files = st.file_uploader(
    "Upload trade logs", 
    type=['csv', 'xlsx', 'xls'], 
    accept_multiple_files=True
)

def load_data(files):
    dfs = []
    rome_tz = pytz.timezone('Europe/Rome')
    
    for file in files:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Show original columns for debugging
        st.write(f"📁 File: {file.name}")
        st.write(f"Original columns: {list(df.columns)}")
        
        # Normalize columns - make lowercase and strip spaces, handle numeric columns
        df.columns = [str(col).strip().lower() if pd.notna(col) else f'column_{i}' 
                      for i, col in enumerate(df.columns)]
        st.write(f"Normalized columns: {list(df.columns)}")
        
        # Map to required schema
        required_cols = ['symbol', 'entry time', 'p/l (points)', 'drawdown (points)', 'run-up (points)', 'd/r flag']
        col_mapping = {}
        missing_cols = []
        
        # Try different matching strategies
        for req_col in required_cols:
            found = False
            req_normalized = req_col.lower()
            
            # First try exact match
            if req_normalized in df.columns:
                col_mapping[req_normalized] = req_col
                found = True
            else:
                # Try partial match - remove spaces and special chars
                req_clean = req_normalized.replace(' ', '').replace('(', '').replace(')', '').replace('/', '')
                
                for col in df.columns:
                    # Ensure col is a string
                    col_str = str(col)
                    col_clean = col_str.replace(' ', '').replace('(', '').replace(')', '').replace('/', '')
                    
                    # Special handling for the Excel file format
                    if req_col == 'p/l (points)' and ('p/l (p)' in col_str.lower() or 'pl(p)' in col_clean):
                        col_mapping[col] = req_col
                        found = True
                        break
                    elif req_col == 'drawdown (points)' and ('dd (p)' in col_str.lower() or 'dd(p)' in col_clean):
                        col_mapping[col] = req_col
                        found = True
                        break
                    elif req_col == 'run-up (points)' and ('rp (p)' in col_str.lower() or 'rp(p)' in col_clean):
                        col_mapping[col] = req_col
                        found = True
                        break
                    elif req_col == 'd/r flag' and col_str.lower() == 'd/r':
                        col_mapping[col] = req_col
                        found = True
                        break
                    
                    # Check if required column is contained in actual column or vice versa
                    if req_clean in col_clean or col_clean in req_clean:
                        col_mapping[col] = req_col
                        found = True
                        break
                    
                    # Also try matching key parts
                    if 'p/l' in req_normalized and 'pl' in col_str:
                        col_mapping[col] = req_col
                        found = True
                        break
                    elif 'drawdown' in req_normalized and ('drawdown' in col_str or 'dd' in col_str):
                        col_mapping[col] = req_col
                        found = True
                        break
                    elif 'run-up' in req_normalized and ('runup' in col_str or 'run up' in col_str or 'run-up' in col_str or 'rp' in col_str):
                        col_mapping[col] = req_col
                        found = True
                        break
                    elif 'd/r flag' in req_normalized and ('dr' in col_str or 'd/r' in col_str or 'flag' in col_str):
                        col_mapping[col] = req_col
                        found = True
                        break
                    elif 'symbol' in req_normalized and 'symbol' in col_str:
                        col_mapping[col] = req_col
                        found = True
                        break
                    elif 'entry time' in req_normalized and ('time' in col_str or 'entry' in col_str or 'date' in col_str):
                        col_mapping[col] = req_col
                        found = True
                        break
            
            if not found:
                # For optional columns, provide defaults
                if req_col == 'symbol':
                    # Create a default symbol column
                    df['symbol'] = 'DEFAULT'
                    col_mapping['symbol'] = req_col
                    found = True
                elif req_col == 'entry time':
                    # Create a default entry time column
                    df['entry time'] = pd.Timestamp.now()
                    col_mapping['entry time'] = req_col
                    found = True
                    
            if not found:
                missing_cols.append(req_col)
        
        # Show mapping results
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.info("Please ensure your file has these columns (exact or similar names):")
            for col in required_cols:
                st.write(f"- {col}")
            return None
        
        st.success(f"✅ Found all required columns!")
        st.write("Column mapping:", col_mapping)
        
        # Rename columns
        df = df.rename(columns=col_mapping)
        
        # Select only required columns
        try:
            df = df[required_cols]
        except KeyError as e:
            st.error(f"Error selecting columns: {e}")
            st.write(f"Available columns after rename: {list(df.columns)}")
            return None
        
        # Parse datetime
        try:
            df['entry time'] = pd.to_datetime(df['entry time']).dt.tz_localize(rome_tz)
        except:
            try:
                df['entry time'] = pd.to_datetime(df['entry time'])
            except Exception as e:
                st.warning(f"Could not parse datetime with timezone: {e}")
                df['entry time'] = pd.to_datetime(df['entry time'])
        
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True) if dfs else None

# Load data
if uploaded_files:
    df = load_data(uploaded_files)
    
    if df is not None:
        st.success(f"Loaded {len(df)} trades from {len(uploaded_files)} files")
        
        # Data preview
        with st.expander("📊 Data Preview"):
            st.dataframe(df.head(20))
        
        # Simulation
        if run_sim and st.session_state.brackets:
            with st.spinner("Running simulations..."):
                
                def simulate_inversions(data, brackets):
                    results = []
                    
                    for tp, sl in brackets:
                        # Copy data
                        sim_df = data.copy()
                        
                        # Invert P/L
                        sim_df['original_pl'] = sim_df['p/l (points)']
                        sim_df['p/l (points)'] = -sim_df['p/l (points)']
                        
                        # Calculate outcomes
                        outcomes = []
                        for _, row in sim_df.iterrows():
                            runup = row['run-up (points)']
                            drawdown = row['drawdown (points)']
                            dr_flag = row['d/r flag']
                            
                            if dr_flag == 'RP':  # Run-up first
                                if runup >= tp:
                                    outcomes.append(tp)
                                elif drawdown >= sl:
                                    outcomes.append(-sl)
                                else:
                                    outcomes.append(0)
                            else:  # Drawdown first
                                if drawdown >= sl:
                                    outcomes.append(-sl)
                                elif runup >= tp:
                                    outcomes.append(tp)
                                else:
                                    outcomes.append(0)
                        
                        sim_df['sim_pl'] = outcomes
                        
                        # Calculate metrics
                        total_pl = sim_df['sim_pl'].sum()
                        avg_pl = sim_df['sim_pl'].mean()
                        wins = (sim_df['sim_pl'] > 0).sum()
                        losses = (sim_df['sim_pl'] < 0).sum()
                        hit_rate = wins / len(sim_df) if len(sim_df) > 0 else 0
                        
                        avg_win = sim_df[sim_df['sim_pl'] > 0]['sim_pl'].mean() if wins > 0 else 0
                        avg_loss = abs(sim_df[sim_df['sim_pl'] < 0]['sim_pl'].mean()) if losses > 0 else 1
                        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0
                        
                        expectancy = (hit_rate * avg_win) - ((1 - hit_rate) * avg_loss)
                        
                        # Sharpe ratio
                        returns = sim_df['sim_pl']
                        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                        
                        # Max drawdown
                        cumsum = returns.cumsum()
                        running_max = cumsum.expanding().max()
                        drawdown_series = cumsum - running_max
                        max_dd = abs(drawdown_series.min())
                        
                        # MAR ratio
                        mar = total_pl / max_dd if max_dd > 0 else 0
                        
                        # Profit factor
                        gross_profit = sim_df[sim_df['sim_pl'] > 0]['sim_pl'].sum()
                        gross_loss = abs(sim_df[sim_df['sim_pl'] < 0]['sim_pl'].sum())
                        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
                        
                        result = {
                            'TP': tp,
                            'SL': sl,
                            'Total P/L': total_pl,
                            'Avg P/L': avg_pl,
                            'Hit Rate': hit_rate,
                            'Payoff Ratio': payoff_ratio,
                            'Expectancy': expectancy,
                            'Sharpe Ratio': sharpe,
                            'Max DD': max_dd,
                            'MAR': mar,
                            'Profit Factor': profit_factor,
                            'sim_data': sim_df
                        }
                        
                        # Advanced stats if enabled
                        if advanced_stats:
                            # T-test
                            t_stat, p_value = stats.ttest_1samp(returns, 0)
                            result['T-stat'] = t_stat
                            result['P-value'] = p_value
                            
                            # Bootstrap CI
                            bootstrap_means = []
                            for _ in range(1000):
                                sample = np.random.choice(returns, size=len(returns), replace=True)
                                bootstrap_means.append(sample.mean())
                            ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])
                            result['CI 95%'] = f"[{ci_lower:.2f}, {ci_upper:.2f}]"
                            
                            # Normality test
                            jb_result = jarque_bera(returns)
                            # jarque_bera returns (JB statistic, p-value, skew, kurtosis)
                            jb_stat = jb_result[0]
                            jb_pvalue = jb_result[1]
                            result['JB p-value'] = jb_pvalue
                        
                        results.append(result)
                    
                    return pd.DataFrame(results)
                
                # Run simulation
                results_df = simulate_inversions(df, st.session_state.brackets)
                
                # Find best bracket
                metric_map = {
                    'Expectancy': 'Expectancy',
                    'Sharpe Ratio': 'Sharpe Ratio', 
                    'Profit Factor': 'Profit Factor',
                    'Hit Rate': 'Hit Rate',
                    'Total P/L': 'Total P/L'
                }
                best_idx = results_df[metric_map[edge_metric]].idxmax()
                
                # Display results
                st.header("📈 Simulation Results")
                
                # Highlight best
                def highlight_best(row):
                    if row.name == best_idx:
                        return ['background-color: lightgreen'] * len(row)
                    return [''] * len(row)
                
                display_cols = ['TP', 'SL', 'Total P/L', 'Avg P/L', 'Hit Rate', 'Payoff Ratio', 
                               'Expectancy', 'Sharpe Ratio', 'Max DD', 'MAR', 'Profit Factor']
                if advanced_stats:
                    display_cols.extend(['T-stat', 'P-value', 'CI 95%', 'JB p-value'])
                
                st.dataframe(
                    results_df[display_cols].style.apply(highlight_best, axis=1).format({
                        'Total P/L': '{:.2f}',
                        'Avg P/L': '{:.2f}',
                        'Hit Rate': '{:.2%}',
                        'Payoff Ratio': '{:.2f}',
                        'Expectancy': '{:.2f}',
                        'Sharpe Ratio': '{:.2f}',
                        'Max DD': '{:.2f}',
                        'MAR': '{:.2f}',
                        'Profit Factor': '{:.2f}'
                    })
                )
                
                # Best bracket details
                best_row = results_df.iloc[best_idx]
                st.success(f"🎯 Best bracket: TP={best_row['TP']}, SL={best_row['SL']} "
                          f"(optimized for {edge_metric}: {best_row[metric_map[edge_metric]]:.2f})")
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig, ax = plt.subplots(figsize=(8, 6))
                    best_data = best_row['sim_data']['sim_pl']
                    ax.hist(best_data, bins=30, alpha=0.7, color='blue', edgecolor='black')
                    ax.axvline(best_data.mean(), color='red', linestyle='--', label=f'Mean: {best_data.mean():.2f}')
                    ax.set_xlabel('P/L (points)')
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'P/L Distribution (TP={best_row["TP"]}, SL={best_row["SL"]})')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                with col2:
                    # Equity curve
                    fig, ax = plt.subplots(figsize=(8, 6))
                    equity = best_row['sim_data']['sim_pl'].cumsum()
                    ax.plot(equity, color='green', linewidth=2)
                    ax.fill_between(range(len(equity)), 0, equity, alpha=0.3, color='green')
                    ax.set_xlabel('Trade Number')
                    ax.set_ylabel('Cumulative P/L')
                    ax.set_title(f'Equity Curve (TP={best_row["TP"]}, SL={best_row["SL"]})')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                # Export options
                csv = results_df[display_cols].to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name="inverse_edge_results.csv",
                    mime="text/csv"
                )
                
                # Store in state for LLM
                st.session_state.df = best_row['sim_data']
                
# LLM Assistant
st.header("🤖 AI Assistant")
st.info("Ask questions about your simulation results!")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know about the results?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        if 'df' in st.session_state:
            response = f"I can see you have {len(st.session_state.df)} trades in your simulation. The data shows interesting patterns that could be explored further."
        else:
            response = "Please run a simulation first so I can analyze your results!"
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})