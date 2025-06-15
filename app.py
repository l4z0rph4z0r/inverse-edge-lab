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
                        
                        # Store original P/L for reference
                        sim_df['original_pl'] = sim_df['p/l (points)']
                        
                        # Calculate outcomes for inverse trades
                        outcomes = []
                        for _, row in sim_df.iterrows():
                            runup = abs(row['run-up (points)'])  # Absolute value
                            drawdown = abs(row['drawdown (points)'])  # Absolute value
                            dr_flag = row['d/r flag']
                            
                            # When we bet AGAINST the trader:
                            # - Their run-up becomes our drawdown
                            # - Their drawdown becomes our run-up
                            
                            if dr_flag == 'RP':  # Original trade had run-up first
                                # For inverse trade: we face drawdown first (their runup)
                                if runup >= sl:  # Their runup hits our stop loss
                                    outcomes.append(-sl)
                                elif drawdown >= tp:  # Their drawdown hits our take profit
                                    outcomes.append(tp)
                                else:
                                    # Neither TP nor SL hit, return inverse of original P/L
                                    outcomes.append(-row['original_pl'])
                            else:  # Original trade had drawdown first (DD)
                                # For inverse trade: we face run-up first (their drawdown)
                                if drawdown >= tp:  # Their drawdown hits our take profit
                                    outcomes.append(tp)
                                elif runup >= sl:  # Their runup hits our stop loss
                                    outcomes.append(-sl)
                                else:
                                    # Neither TP nor SL hit, return inverse of original P/L
                                    outcomes.append(-row['original_pl'])
                        
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
                
                # Explain the simulation
                with st.expander("ℹ️ How Inverse Trading Simulation Works"):
                    st.markdown("""
                    **Inverse Trading Strategy**: We bet AGAINST each trader's position.
                    
                    **Key Concepts**:
                    - When a trader goes LONG, we go SHORT (and vice versa)
                    - The trader's **run-up** (favorable movement) becomes our **adverse movement**
                    - The trader's **drawdown** (unfavorable movement) becomes our **favorable movement**
                    
                    **Simulation Logic**:
                    1. For each trade, we take the opposite position
                    2. We apply fixed Take Profit (TP) and Stop Loss (SL) brackets
                    3. Based on the D/R flag, we know which movement happened first:
                       - **RP (Run-up first)**: Trader saw profit first → We face loss first
                       - **DD (Drawdown first)**: Trader saw loss first → We see profit first
                    4. We check if our TP or SL gets hit based on these movements
                    
                    **Example**: If a trader lost 20 points, our inverse position would gain 20 points 
                    (unless limited by our TP bracket).
                    """)
                
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
                
                # Edge comparison
                original_total = df['p/l (points)'].sum()
                inverse_simple = -original_total  # Simple inverse without brackets
                inverse_optimized = best_row['Total P/L']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Traders' P/L", f"{original_total:.2f}", 
                             delta=None, delta_color="off")
                with col2:
                    st.metric("Simple Inverse P/L", f"{inverse_simple:.2f}", 
                             delta=f"{inverse_simple - original_total:.2f}", 
                             delta_color="normal")
                with col3:
                    st.metric("Optimized Inverse P/L", f"{inverse_optimized:.2f}", 
                             delta=f"{inverse_optimized - original_total:.2f}", 
                             delta_color="normal")
                
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
                
                # Store in state for LLM with current bracket info
                st.session_state.df = best_row['sim_data']
                st.session_state.TP = best_row['TP']
                st.session_state.SL = best_row['SL']
                st.session_state.results_df = results_df
                
# LLM Assistant
st.header("🤖 AI Assistant")
st.info("Ask questions about your simulation results - powered by Perplexity AI!")

# Perplexity API configuration
PERPLEXITY_API_KEY = "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"
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def query_perplexity(prompt, context_data=None):
    """Query Perplexity AI with context about the simulation results"""
    import requests
    
    # Build context message
    system_message = "You are an AI assistant specialized in analyzing inverse trading simulations. "
    
    if context_data is not None:
        # Extract comprehensive statistics from the data
        total_trades = len(context_data)
        wins = (context_data['sim_pl'] > 0).sum()
        losses = (context_data['sim_pl'] < 0).sum()
        breakeven = (context_data['sim_pl'] == 0).sum()
        avg_pl = context_data['sim_pl'].mean()
        total_pl = context_data['sim_pl'].sum()
        hit_rate = wins / total_trades if total_trades > 0 else 0
        
        # Calculate additional metrics
        avg_win = context_data[context_data['sim_pl'] > 0]['sim_pl'].mean() if wins > 0 else 0
        avg_loss = abs(context_data[context_data['sim_pl'] < 0]['sim_pl'].mean()) if losses > 0 else 0
        
        # Analyze original vs simulated
        original_total = context_data['original_pl'].sum()
        
        # Count TP/SL hits
        tp_hits = 0
        sl_hits = 0
        neither_hits = 0
        
        if 'TP' in st.session_state and 'SL' in st.session_state:
            current_tp = st.session_state.get('TP', 0)
            current_sl = st.session_state.get('SL', 0)
            tp_hits = (context_data['sim_pl'] == current_tp).sum()
            sl_hits = (context_data['sim_pl'] == -current_sl).sum()
            neither_hits = total_trades - tp_hits - sl_hits
        
        system_message += f"""
        Current inverse trading simulation results:
        
        OVERVIEW:
        - Total trades analyzed: {total_trades}
        - Original traders' total P/L: {original_total:.2f} points
        - Inverse strategy total P/L: {total_pl:.2f} points
        - Edge gained by inverting: {total_pl - (-original_total):.2f} points
        
        TRADE OUTCOMES:
        - Winning trades: {wins} ({hit_rate*100:.1f}%)
        - Losing trades: {losses} ({losses/total_trades*100 if total_trades > 0 else 0:.1f}%)
        - Breakeven trades: {breakeven}
        - Take Profit hits: {tp_hits}
        - Stop Loss hits: {sl_hits}
        - Neither TP/SL hit: {neither_hits}
        
        PERFORMANCE METRICS:
        - Average P/L per trade: {avg_pl:.2f} points
        - Average winning trade: {avg_win:.2f} points
        - Average losing trade: -{avg_loss:.2f} points
        - Win/Loss ratio: {avg_win/avg_loss if avg_loss > 0 else 0:.2f}
        
        The simulation tests betting AGAINST traders' positions using fixed TP/SL brackets.
        When a trader's position moves favorably (run-up), it becomes our adverse movement.
        When a trader's position moves unfavorably (drawdown), it becomes our favorable movement.
        
        Help the user understand if there's a statistical edge in betting against these traders.
        """
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(PERPLEXITY_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error connecting to AI service: {str(e)}"

if prompt := st.chat_input("What would you like to know about the results?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        if 'df' in st.session_state:
            with st.spinner("Thinking..."):
                response = query_perplexity(prompt, st.session_state.df)
        else:
            response = "Please run a simulation first so I can analyze your results!"
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})