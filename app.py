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
from translations import get_text

warnings.filterwarnings('ignore')

# Redis connection
REDIS_URL = os.getenv('UPSTASH_REDIS_REST_URL')
REDIS_TOKEN = os.getenv('UPSTASH_REDIS_REST_TOKEN')

if REDIS_URL and REDIS_TOKEN:
    r = redis.Redis.from_url(f"redis://default:{REDIS_TOKEN}@{REDIS_URL.replace('https://', '')}")
else:
    r = None  # Fallback for local dev

# Initialize language in session state
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# Page config
st.set_page_config(
    page_title=get_text("page_title", st.session_state.language), 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Language selector in the top right
col1, col2, col3 = st.columns([10, 1, 1])
with col3:
    language = st.selectbox(
        "🌐",
        options=["en", "it"],
        index=0 if st.session_state.language == "en" else 1,
        format_func=lambda x: "🇬🇧 EN" if x == "en" else "🇮🇹 IT",
        key="lang_selector"
    )
    if language != st.session_state.language:
        st.session_state.language = language
        st.rerun()

# Get current language
lang = st.session_state.language

# Simple authentication for demo
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title(get_text("login_title", lang))
    with st.form("login_form"):
        username = st.text_input(get_text("username", lang))
        password = st.text_input(get_text("password", lang), type="password")
        submitted = st.form_submit_button(get_text("login", lang))
        
        if submitted:
            # Simple hardcoded check for demo
            if username == "admin" and password == "admin123":
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error(get_text("invalid_credentials", lang))
    st.stop()

# Sidebar
with st.sidebar:
    st.header(get_text("simulation_controls", lang))
    if st.button(get_text("logout", lang)):
        st.session_state.authenticated = False
        st.rerun()
    
    # Bracket pairs
    st.subheader(get_text("bracket_pairs", lang))
    default_brackets = [(10, 10), (15, 10), (20, 10), (30, 15)]
    
    if 'brackets' not in st.session_state:
        st.session_state.brackets = default_brackets
    
    # Manual entry
    col1, col2 = st.columns(2)
    with col1:
        new_tp = st.number_input("TP", min_value=1, value=20)
    with col2:
        new_sl = st.number_input("SL", min_value=1, value=10)
    
    if st.button(get_text("add_bracket", lang)):
        st.session_state.brackets.append((new_tp, new_sl))
    
    # Display current brackets
    st.write(get_text("current_brackets", lang))
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
    rr_lock = st.checkbox(get_text("lock_rr", lang))
    if rr_lock:
        rr_ratio = st.slider(get_text("rr_ratio", lang), 0.5, 5.0, 2.0, 0.1)
    
    # Edge metric
    edge_metric = st.selectbox(
        get_text("optimize_metric", lang),
        ["Expectancy", "Sharpe Ratio", "Profit Factor", "Hit Rate", "Total P/L"]
    )
    
    # Advanced stats
    advanced_stats = st.checkbox(get_text("run_advanced_tests", lang))
    
    if advanced_stats:
        with st.expander(get_text("advanced_tests_title", lang)):
            st.markdown(get_text("advanced_tests_desc", lang))"""
            **Advanced tests validate if your edge is statistically significant or just luck:**
            
            • **T-Test**: Tests if average P/L is significantly different from zero
              - Null hypothesis: Mean P/L = 0 (no edge)
              - P-value < 0.05 suggests real edge (not random)
            
            • **Bootstrap Confidence Interval (95% CI)**: 
              - Resamples your trades 1000 times to estimate uncertainty
              - If CI doesn't include zero, edge is likely real
              - Example: CI [5.2, 12.8] means 95% confident true edge is between 5.2-12.8 points
            
            • **Jarque-Bera Normality Test**:
              - Tests if returns follow normal distribution
              - Important for risk models that assume normality
              - P-value < 0.05 indicates non-normal distribution (common in trading)
            
            **Why use these?** They help distinguish genuine edge from lucky streaks.
            """)
    
    # Run button
    run_sim = st.button(get_text("run_simulation", lang), type="primary", use_container_width=True)

# Main content
st.title(get_text("app_title", lang))
st.markdown(get_text("app_subtitle", lang))

# File upload
uploaded_files = st.file_uploader(
    get_text("upload_files", lang), 
    type=['csv', 'xlsx', 'xls'], 
    accept_multiple_files=True
)

# Mathematical explanation
with st.expander(get_text("math_framework_title", lang), expanded=False):
    st.markdown(get_text("math_framework_content", lang))"""
    ## How Inverse Trading Simulation Works
    
    ### 1. Core Concept
    We simulate taking the **opposite position** of each trader with fixed risk-reward brackets.
    
    **Key Insight**: When you bet against a trader:
    - Their **favorable movement** (run-up) → Your **adverse movement** (potential loss)
    - Their **unfavorable movement** (drawdown) → Your **favorable movement** (potential profit)
    
    ### 2. Mathematical Formulation
    
    For each trade `i` in the dataset:
    - `P/L_original[i]` = Original trader's profit/loss
    - `RP[i]` = Run-up (maximum favorable excursion for trader)
    - `DD[i]` = Drawdown (maximum adverse excursion for trader)
    - `D/R[i]` = Flag indicating which came first (RP or DD)
    
    **Inverse Position Calculation**:
    ```
    For our inverse position with brackets (TP, SL):
    
    IF D/R[i] == "RP" (trader saw profit first):
        IF RP[i] ≥ SL:  # Their profit hits our stop loss
            P/L_inverse[i] = -SL
        ELIF DD[i] ≥ TP:  # Their loss hits our take profit
            P/L_inverse[i] = +TP
        ELSE:  # Neither bracket hit
            P/L_inverse[i] = -P/L_original[i]
            
    ELIF D/R[i] == "DD" (trader saw loss first):
        IF DD[i] ≥ TP:  # Their loss hits our take profit
            P/L_inverse[i] = +TP
        ELIF RP[i] ≥ SL:  # Their profit hits our stop loss
            P/L_inverse[i] = -SL
        ELSE:  # Neither bracket hit
            P/L_inverse[i] = -P/L_original[i]
    ```
    
    ### 3. Key Metrics Calculated
    
    **Basic Metrics**:
    - **Total P/L** = Σ P/L_inverse[i]
    - **Hit Rate** = Count(P/L_inverse > 0) / Total Trades
    - **Expectancy** = (Hit Rate × Avg Win) - ((1 - Hit Rate) × Avg Loss)
    - **Sharpe Ratio** = Mean(Returns) / StdDev(Returns) × √252
    - **Profit Factor** = Gross Profits / Gross Losses
    
    **Risk Metrics**:
    - **Maximum Drawdown** = Largest peak-to-trough decline in equity curve
    - **MAR Ratio** = Total P/L / Max Drawdown
    
    ### 4. Why This Works
    
    If traders consistently lose money (negative edge), then:
    - Taking opposite positions should yield positive expectancy
    - Fixed brackets (TP/SL) can optimize this edge by:
      - Limiting losses when traders occasionally win big
      - Capturing consistent profits from their frequent losses
    
    ### 5. Example Scenario
    
    **Original Trade**: Trader loses 15 points
    - Run-up: 5 points (RP)
    - Drawdown: 20 points (DD)
    - D/R Flag: "RP" (saw 5 point profit before 20 point loss)
    
    **Our Inverse Trade** with TP=10, SL=10:
    - Trader's 5 point run-up = Our 5 point drawdown (< 10 SL, continues)
    - Trader's 20 point drawdown = Our 20 point run-up (> 10 TP, locked in)
    - **Result**: We make +10 points (TP hit)
    
    While simple inversion would gain +15, the bracket system:
    - Protects against occasional large trader wins
    - Provides consistent, controlled profits
    """)

st.markdown("---")

def load_data(files):
    dfs = []
    rome_tz = pytz.timezone('Europe/Rome')
    
    for file in files:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Show original columns for debugging
        st.write(get_text("file_label", lang, name=file.name))
        st.write(get_text("original_columns", lang, cols=list(df.columns)))
        
        # Normalize columns - make lowercase and strip spaces, handle numeric columns
        df.columns = [str(col).strip().lower() if pd.notna(col) else f'column_{i}' 
                      for i, col in enumerate(df.columns)]
        st.write(get_text("normalized_columns", lang, cols=list(df.columns)))
        
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
            st.error(get_text("missing_columns", lang, cols=missing_cols))
            st.info("Please ensure your file has these columns (exact or similar names):")
            for col in required_cols:
                st.write(f"- {col}")
            return None
        
        st.success(get_text("columns_found", lang))
        st.write(get_text("column_mapping", lang), col_mapping)
        
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
        st.success(get_text("loaded_trades", lang, count=len(df), files=len(uploaded_files)))
        
        # Data preview
        with st.expander(get_text("data_preview", lang)):
            st.dataframe(df.head(20))
        
        # Simulation
        if run_sim and st.session_state.brackets:
            with st.spinner(get_text("running_simulations", lang)):
                
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
                best_row = results_df.iloc[best_idx]  # Define best_row here
                
                # Display results
                st.header(get_text("simulation_results", lang))
                
                # Explain the simulation
                with st.expander(get_text("how_inverse_works_title", lang)):
                    st.markdown(get_text("how_inverse_works", lang))"""
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
                
                # Create tabs for basic and advanced analysis
                if advanced_stats:
                    tab1, tab2 = st.tabs([get_text("basic_analysis_tab", lang), get_text("advanced_stats_tab", lang)])
                else:
                    # If advanced stats not selected, just show basic analysis without tabs
                    tab1 = st.container()
                    tab2 = None
                
                # Highlight best function
                def highlight_best(row):
                    if row.name == best_idx:
                        return ['background-color: lightgreen'] * len(row)
                    return [''] * len(row)
                
                # Basic Analysis Tab
                with tab1:
                    st.markdown(get_text("basic_metrics_title", lang))
                    
                    # Basic columns
                    basic_cols = ['TP', 'SL', 'Total P/L', 'Avg P/L', 'Hit Rate', 'Payoff Ratio', 
                                 'Expectancy', 'Sharpe Ratio', 'Max DD', 'MAR', 'Profit Factor']
                    
                    st.dataframe(
                        results_df[basic_cols].style.apply(highlight_best, axis=1).format({
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
                
                # Advanced Statistics Tab (only if enabled)
                if tab2 is not None:
                    with tab2:
                        st.markdown(get_text("statistical_tests_title", lang))
                        
                        # Show results with statistical tests
                        stat_cols = ['TP', 'SL', 'Total P/L', 'Expectancy', 'T-stat', 'P-value', 'CI 95%', 'JB p-value']
                        
                        st.dataframe(
                            results_df[stat_cols].style.apply(highlight_best, axis=1).format({
                                'Total P/L': '{:.2f}',
                                'Expectancy': '{:.2f}',
                                'T-stat': '{:.3f}',
                                'P-value': '{:.4f}',
                                'JB p-value': '{:.4f}'
                            })
                        )
                        
                        # Detailed interpretation for best bracket
                        st.markdown(get_text("best_bracket_analysis", lang))
                        best_pvalue = best_row.get('P-value', 1.0)
                        best_ci = best_row.get('CI 95%', '[0, 0]')
                        best_jb = best_row.get('JB p-value', 1.0)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(get_text("edge_significance", lang))
                            if best_pvalue < 0.01:
                                st.success(get_text("highly_significant", lang, p=best_pvalue))
                                st.markdown(get_text("very_strong_evidence", lang))
                            elif best_pvalue < 0.05:
                                st.warning(get_text("significant", lang, p=best_pvalue))
                                st.markdown(get_text("good_evidence", lang))
                            else:
                                st.error(get_text("not_significant", lang, p=best_pvalue))
                                st.markdown(get_text("insufficient_evidence", lang))
                            
                            # Additional context
                            st.markdown(f"""
                            **T-statistic**: {best_row.get('T-stat', 0):.3f}
                            
                            Interpretation: The average P/L is {abs(best_row.get('T-stat', 0)):.1f} standard errors 
                            {'above' if best_row.get('T-stat', 0) > 0 else 'below'} zero.
                            """)
                        
                        with col2:
                            st.markdown(get_text("confidence_interval", lang))
                            st.info(get_text("true_edge_between", lang, ci=best_ci))
                            # Parse CI to check if it includes zero
                            try:
                                ci_str = best_ci.strip('[]')
                                ci_lower, ci_upper = map(float, ci_str.split(','))
                                if ci_lower > 0:
                                    st.success(get_text("ci_excludes_zero", lang))
                                elif ci_upper < 0:
                                    st.error(get_text("ci_negative", lang))
                                else:
                                    st.warning(get_text("ci_includes_zero", lang))
                                
                                # Additional context
                                st.markdown(f"""
                                {get_text("interval_width", lang, width=ci_upper - ci_lower)}
                                
                                {get_text("narrower_intervals", lang)}
                                """)
                            except:
                                pass
                        
                        st.markdown("---")
                        st.markdown(get_text("distribution_normality", lang))
                        if best_jb < 0.05:
                            st.info(get_text("non_normal_returns", lang, p=best_jb))
                            st.markdown("""
                            Non-normal distributions are common in trading and may indicate:
                            - Fat tails (extreme events more likely than normal distribution predicts)
                            - Skewness (asymmetric returns)
                            - Need for robust statistical methods
                            """)
                        else:
                            st.info(get_text("normal_returns", lang, p=best_jb))
                            st.markdown("Standard risk models and metrics are appropriate.")
                        
                        # Monte Carlo simulation preview
                        st.markdown("---")
                        st.markdown(get_text("bootstrap_title", lang))
                        st.markdown(get_text("bootstrap_desc", lang, count=len(best_row['sim_data'])))
                
                # Best bracket details (shown above tabs)
                st.success(get_text("best_bracket", lang, 
                                  tp=best_row['TP'], 
                                  sl=best_row['SL'],
                                  metric=edge_metric,
                                  value=best_row[metric_map[edge_metric]]))
                
                # Edge comparison (shown above tabs)
                original_total = df['p/l (points)'].sum()
                inverse_simple = -original_total  # Simple inverse without brackets
                inverse_optimized = best_row['Total P/L']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(get_text("original_pl", lang), f"{original_total:.2f}", 
                             delta=None, delta_color="off")
                with col2:
                    st.metric(get_text("simple_inverse_pl", lang), f"{inverse_simple:.2f}", 
                             delta=f"{inverse_simple - original_total:.2f}", 
                             delta_color="normal")
                with col3:
                    st.metric(get_text("optimized_inverse_pl", lang), f"{inverse_optimized:.2f}", 
                             delta=f"{inverse_optimized - original_total:.2f}", 
                             delta_color="normal")
                
                st.markdown("---")
                
                # Visualizations (shown below tabs)
                st.markdown(get_text("visualizations", lang))
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig, ax = plt.subplots(figsize=(8, 6))
                    best_data = best_row['sim_data']['sim_pl']
                    ax.hist(best_data, bins=30, alpha=0.7, color='blue', edgecolor='black')
                    ax.axvline(best_data.mean(), color='red', linestyle='--', 
                              label=get_text("mean_label", lang, value=best_data.mean()))
                    ax.set_xlabel('P/L (points)')
                    ax.set_ylabel(get_text("frequency", lang))
                    ax.set_title(get_text("pl_distribution", lang, tp=best_row["TP"], sl=best_row["SL"]))
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                with col2:
                    # Equity curve
                    fig, ax = plt.subplots(figsize=(8, 6))
                    equity = best_row['sim_data']['sim_pl'].cumsum()
                    ax.plot(equity, color='green', linewidth=2)
                    ax.fill_between(range(len(equity)), 0, equity, alpha=0.3, color='green')
                    ax.set_xlabel(get_text("trade_number", lang))
                    ax.set_ylabel(get_text("cumulative_pl", lang))
                    ax.set_title(get_text("equity_curve", lang, tp=best_row["TP"], sl=best_row["SL"]))
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                # Export options
                # Define all columns for export
                export_cols = ['TP', 'SL', 'Total P/L', 'Avg P/L', 'Hit Rate', 'Payoff Ratio', 
                              'Expectancy', 'Sharpe Ratio', 'Max DD', 'MAR', 'Profit Factor']
                if advanced_stats:
                    export_cols.extend(['T-stat', 'P-value', 'CI 95%', 'JB p-value'])
                
                csv = results_df[export_cols].to_csv(index=False)
                st.download_button(
                    label=get_text("download_csv", lang),
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
st.header(get_text("ai_assistant", lang))
st.info(get_text("ai_info", lang))

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

if prompt := st.chat_input(get_text("ai_prompt", lang)):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        if 'df' in st.session_state:
            with st.spinner("Thinking..."):
                response = query_perplexity(prompt, st.session_state.df)
        else:
            response = get_text("run_simulation_first", lang)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})