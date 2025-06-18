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
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO
import base64
from auth_manager import AuthManager
from admin_panel import show_admin_panel
import hashlib

warnings.filterwarnings('ignore')

# Redis connection
REDIS_URL = os.getenv('UPSTASH_REDIS_REST_URL')
REDIS_TOKEN = os.getenv('UPSTASH_REDIS_REST_TOKEN')

# For local development with Docker, use local Redis
if REDIS_URL and REDIS_URL.startswith('redis://'):
    # Local Redis URL from docker-compose
    r = redis.Redis.from_url(REDIS_URL)
elif REDIS_URL and REDIS_TOKEN and REDIS_URL.startswith('https://'):
    # Upstash Redis - for production
    r = redis.Redis.from_url(f"redis://default:{REDIS_TOKEN}@{REDIS_URL.replace('https://', '')}")
else:
    # Try to connect to local Redis container
    try:
        r = redis.Redis(host='redis', port=6379, decode_responses=True)
    except:
        r = None

# Initialize AuthManager
auth_manager = AuthManager(r)
# Create default admin if doesn't exist
auth_manager.create_admin_if_not_exists()

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

# Authentication check
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_role = None
    st.session_state.username = None

if not st.session_state.authenticated:
    st.title(get_text("login_title", lang))
    
    # Create tabs for login methods
    tab1, tab2 = st.tabs([get_text("admin_login", lang), get_text("token_login", lang)])
    
    with tab1:
        # Admin Login
        with st.form("admin_login_form"):
            admin_username = st.text_input(get_text("username", lang))
            admin_password = st.text_input(get_text("password", lang), type="password")
            admin_submitted = st.form_submit_button(get_text("login", lang))
            
            if admin_submitted:
                if auth_manager.verify_admin(admin_username, admin_password):
                    st.session_state.authenticated = True
                    st.session_state.username = admin_username
                    st.session_state.user_role = 'admin'
                    st.rerun()
                else:
                    st.error(get_text("invalid_credentials", lang))
    
    with tab2:
        # Token Login
        with st.form("token_login_form"):
            token = st.text_input(get_text("access_token", lang), placeholder="Enter your access token")
            user_username = st.text_input(get_text("choose_username", lang), placeholder="Choose a username")
            token_submitted = st.form_submit_button(get_text("login_with_token", lang))
            
            if token_submitted:
                if token and user_username:
                    token_info = auth_manager.use_token(token, user_username)
                    if token_info:
                        st.session_state.authenticated = True
                        st.session_state.username = user_username
                        st.session_state.user_role = 'user'
                        st.success(get_text("login_successful", lang))
                        st.rerun()
                    else:
                        st.error(get_text("invalid_token", lang))
                else:
                    st.error(get_text("token_username_required", lang))
    
    st.stop()

# Sidebar
with st.sidebar:
    st.header(get_text("simulation_controls", lang))
    
    # Show username and role
    st.write(f"👤 {st.session_state.username} ({st.session_state.user_role})")
    
    # Admin Panel button for admins
    if st.session_state.user_role == 'admin':
        if st.button("🔧 " + get_text("admin_panel", lang)):
            st.session_state.show_admin_panel = True
            st.rerun()
    
    # Logout button
    if st.button(get_text("logout", lang)):
        st.session_state.authenticated = False
        st.session_state.show_admin_panel = False
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
            st.markdown(get_text("advanced_tests_desc", lang))
    
    # Commission settings
    st.markdown("---")
    st.subheader(get_text("commission_settings", lang))
    
    # Contract settings
    col1, col2 = st.columns(2)
    with col1:
        contract_size = st.number_input(
            get_text("contract_size", lang),
            min_value=1,
            value=1,
            step=1,
            help=get_text("contract_size_help", lang)
        )
    
    with col2:
        commission_per_side = st.number_input(
            get_text("commission_per_side", lang),
            min_value=0.0,
            value=2.25,  # Common MNQ commission
            step=0.25,
            format="%.2f",
            help=get_text("commission_help", lang)
        )
    
    # Total commission per round trip
    total_commission = commission_per_side * 2  # Entry + Exit
    st.info(get_text("total_commission_info", lang, total=total_commission))
    
    # Point value setting
    st.markdown("---")
    point_value = st.number_input(
        get_text("point_value", lang),
        min_value=0.01,
        value=2.0,  # Default for MNQ
        step=0.01,
        format="%.2f",
        help=get_text("point_value_help", lang)
    )
    
    # Run button
    run_sim = st.button(get_text("run_simulation", lang), type="primary", use_container_width=True)

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

def generate_pdf_report(results_df, best_row, df, lang='en'):
    """Generate a PDF report with simulation results"""
    buffer = BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f4e79'),
        alignment=TA_CENTER,
        spaceAfter=30
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2e75b6'),
        spaceAfter=12
    )
    
    # Helper function to create chart images
    def create_chart_image(fig, width=6*inch, height=4*inch):
        """Convert matplotlib figure to reportlab Image"""
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        img = Image(img_buffer, width=width, height=height)
        plt.close(fig)
        return img
    
    # Title
    story.append(Paragraph(get_text("page_title", lang), title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Report generation date
    story.append(Paragraph(f"{get_text('report_date', lang)}: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Best Configuration Section
    story.append(Paragraph(get_text("best_config_title", lang), heading_style))
    
    config_data = [
        [get_text("col_tp", lang), f"{best_row['TP']:.1f}"],
        [get_text("col_sl", lang), f"{best_row['SL']:.1f}"],
        [get_text("col_gross_pl", lang), f"{best_row['Gross P/L']:.2f}"],
        [get_text("col_commission", lang), f"-${best_row['Commission']:.2f}"],
        [get_text("col_total_pl", lang), f"{best_row['Total P/L']:.2f}"],
        [get_text("col_avg_pl", lang), f"{best_row['Avg P/L']:.2f}"],
        [get_text("col_hit_rate", lang), f"{best_row['Hit Rate']:.1%}"],
        [get_text("col_payoff_ratio", lang), f"{best_row['Payoff Ratio']:.2f}"],
        [get_text("col_expectancy", lang), f"{best_row['Expectancy']:.2f}"],
        [get_text("col_sharpe", lang), f"{best_row['Sharpe Ratio']:.2f}"],
        [get_text("col_max_dd", lang), f"{best_row['Max DD']:.2f}"],
        [get_text("col_mar", lang), f"{best_row['MAR']:.2f}"],
        [get_text("col_profit_factor", lang), f"{best_row['Profit Factor']:.2f}"]
    ]
    
    # Add advanced stats if available
    if 'T-stat' in best_row:
        config_data.extend([
            ["", ""],  # Empty row for separation
            [get_text("advanced_stats_title", lang), ""],
            [get_text("col_tstat", lang), f"{best_row['T-stat']:.4f}"],
            [get_text("col_pvalue", lang), f"{best_row['P-value']:.4f}"],
            [get_text("col_ci95", lang), best_row['CI 95%']],
            [get_text("col_jb_pvalue", lang), f"{best_row['JB p-value']:.4f}"]
        ])
    
    config_table = Table(config_data, colWidths=[3*inch, 2*inch])
    config_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(config_table)
    story.append(Spacer(1, 0.5*inch))
    
    # All Results Summary
    story.append(Paragraph(get_text("all_results_title", lang), heading_style))
    
    # Prepare summary table data
    summary_cols = ['TP', 'SL', 'Total P/L', 'Hit Rate', 'Sharpe Ratio', 'MAR']
    summary_data = [[get_text(f"col_{col.lower().replace(' ', '_').replace('/', '_')}", lang) for col in summary_cols]]
    
    for _, row in results_df.head(10).iterrows():
        summary_data.append([
            f"{row['TP']:.1f}",
            f"{row['SL']:.1f}",
            f"{row['Total P/L']:.2f}",
            f"{row['Hit Rate']:.1%}",
            f"{row['Sharpe Ratio']:.2f}",
            f"{row['MAR']:.2f}"
        ])
    
    summary_table = Table(summary_data)
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(summary_table)
    story.append(PageBreak())
    
    # Charts Section
    story.append(Paragraph(get_text("charts_title", lang), heading_style))
    story.append(Spacer(1, 0.3*inch))
    
    # 1. TP/SL Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    pivot_data = results_df.pivot_table(
        index='SL', 
        columns='TP', 
        values='Total P/L',
        aggfunc='mean'
    )
    im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_xticklabels([f'{x:.1f}' for x in pivot_data.columns])
    ax.set_yticklabels([f'{y:.1f}' for y in pivot_data.index])
    ax.set_xlabel(get_text("col_tp", lang))
    ax.set_ylabel(get_text("col_sl", lang))
    ax.set_title(get_text("heatmap_title", lang))
    plt.colorbar(im, ax=ax, label=get_text("col_total_pl", lang))
    story.append(create_chart_image(fig, width=5*inch, height=3.75*inch))
    story.append(Spacer(1, 0.3*inch))
    
    # 2. Equity Curve for Best Configuration
    fig, ax = plt.subplots(figsize=(8, 6))
    equity = best_row['sim_data']['net_pl_points'].cumsum()
    ax.plot(equity, color='green', linewidth=2)
    ax.fill_between(range(len(equity)), 0, equity, alpha=0.3, color='green')
    ax.set_xlabel(get_text("trade_number", lang))
    ax.set_ylabel(get_text("cumulative_pl", lang))
    ax.set_title(get_text("equity_curve", lang, tp=best_row["TP"], sl=best_row["SL"]))
    ax.grid(True, alpha=0.3)
    story.append(create_chart_image(fig, width=5*inch, height=3.75*inch))
    story.append(Spacer(1, 0.3*inch))
    
    # 3. Performance Metrics Bar Chart
    if len(results_df) > 5:
        fig, ax = plt.subplots(figsize=(8, 6))
        top_5 = results_df.head(5)
        x = range(len(top_5))
        labels = [f"TP:{row['TP']:.1f}/SL:{row['SL']:.1f}" for _, row in top_5.iterrows()]
        
        ax.bar(x, top_5['Total P/L'], color='steelblue', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(get_text("col_total_pl", lang))
        ax.set_title(get_text("top_configs_title", lang))
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        story.append(create_chart_image(fig, width=5*inch, height=3.75*inch))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# Main content
# Check if admin panel should be shown
if st.session_state.get('show_admin_panel', False) and st.session_state.user_role == 'admin':
    show_admin_panel(auth_manager, st.session_state.username, lang)
    if st.button("← " + get_text("back_to_app", lang)):
        st.session_state.show_admin_panel = False
        st.rerun()
else:
    # Main application content
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
        st.markdown(get_text("math_framework_content", lang))

    st.markdown("---")

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
                    
                    def simulate_inversions(data, brackets, commission_per_rt=0.0, contracts=1, point_value=2.0):
                        results = []
                        
                        # Add progress bar for better user experience
                        progress_bar = st.progress(0)
                        total_brackets = len(brackets)
                        
                        for idx, (tp, sl) in enumerate(brackets):
                            # Update progress
                            progress = (idx + 1) / total_brackets
                            progress_bar.progress(progress, text=f"Testing bracket {idx + 1}/{total_brackets}: TP={tp}, SL={sl}")
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
                            
                            # Apply commissions
                            # Use the point_value parameter passed to the function
                            commission_in_points = commission_per_rt / point_value  # Convert dollars to points
                            
                            sim_df['gross_pl_points'] = sim_df['sim_pl'].copy()  # Gross P/L in points
                            sim_df['gross_pl_dollars'] = sim_df['gross_pl_points'] * point_value  # Gross P/L in dollars
                            sim_df['commission_dollars'] = commission_per_rt  # Commission in dollars
                            sim_df['commission_points'] = commission_in_points  # Commission in points
                            sim_df['net_pl_points'] = sim_df['gross_pl_points'] - commission_in_points  # Net P/L in points
                            sim_df['net_pl_dollars'] = sim_df['net_pl_points'] * point_value  # Net P/L in dollars
                            
                            # Calculate metrics using net P/L in points (for consistency with original data)
                            total_pl_points = sim_df['net_pl_points'].sum()
                            total_pl_dollars = sim_df['net_pl_dollars'].sum()
                            avg_pl_points = sim_df['net_pl_points'].mean()
                            wins = (sim_df['net_pl_points'] > 0).sum()
                            losses = (sim_df['net_pl_points'] < 0).sum()
                            hit_rate = wins / len(sim_df) if len(sim_df) > 0 else 0
                            
                            avg_win = sim_df[sim_df['net_pl_points'] > 0]['net_pl_points'].mean() if wins > 0 else 0
                            avg_loss = abs(sim_df[sim_df['net_pl_points'] < 0]['net_pl_points'].mean()) if losses > 0 else 1
                            payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0
                            
                            expectancy = (hit_rate * avg_win) - ((1 - hit_rate) * avg_loss)
                            
                            # Sharpe ratio (using net returns in points)
                            returns = sim_df['net_pl_points']
                            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                            
                            # Max drawdown (using net P/L in points)
                            cumsum = returns.cumsum()
                            running_max = cumsum.expanding().max()
                            drawdown_series = cumsum - running_max
                            max_dd = abs(drawdown_series.min())
                            
                            # MAR ratio
                            mar = total_pl_points / max_dd if max_dd > 0 else 0
                            
                            # Profit factor (using net P/L in points)
                            gross_profit = sim_df[sim_df['net_pl_points'] > 0]['net_pl_points'].sum()
                            gross_loss = abs(sim_df[sim_df['net_pl_points'] < 0]['net_pl_points'].sum())
                            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
                            
                            # Commission impact
                            total_commission_dollars = commission_per_rt * len(sim_df) * contracts
                            gross_total_pl_points = sim_df['gross_pl_points'].sum()
                            gross_total_pl_dollars = sim_df['gross_pl_dollars'].sum()
                            
                            result = {
                                'TP': tp,
                                'SL': sl,
                                'Gross P/L': gross_total_pl_points,  # In points
                                'Gross P/L $': gross_total_pl_dollars,  # In dollars
                                'Commission': total_commission_dollars,  # In dollars
                                'Total P/L': total_pl_points,  # Net P/L in points
                                'Total P/L $': total_pl_dollars,  # Net P/L in dollars
                                'Avg P/L': avg_pl_points,
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
                        
                        # Clear progress bar
                        progress_bar.empty()
                        
                        return pd.DataFrame(results)
                    
                    # Run simulation
                    results_df = simulate_inversions(
                        df, 
                        st.session_state.brackets,
                        commission_per_rt=total_commission,
                        contracts=contract_size,
                        point_value=point_value
                    )
                    
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
                        st.markdown(get_text("how_inverse_works", lang))
                    
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
                        
                        # Basic columns - show both points and dollars
                        basic_cols = ['TP', 'SL', 'Gross P/L', 'Gross P/L $', 'Commission', 'Total P/L', 'Total P/L $', 
                                     'Avg P/L', 'Hit Rate', 'Payoff Ratio', 'Expectancy', 'Sharpe Ratio', 'Max DD', 'MAR', 'Profit Factor']
                        
                        # Check if all columns exist
                        available_cols = [col for col in basic_cols if col in results_df.columns]
                        
                        st.dataframe(
                            results_df[available_cols].style.apply(highlight_best, axis=1).format({
                                'Gross P/L': '{:.2f}',
                                'Gross P/L $': '${:,.2f}',
                                'Commission': '${:,.2f}',
                                'Total P/L': '{:.2f}',
                                'Total P/L $': '${:,.2f}',
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
                    
                    # Commission Impact Summary
                    if total_commission > 0:
                        st.markdown("---")
                        st.subheader(get_text("commission_impact_title", lang))
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(get_text("gross_pl_metric", lang), f"{best_row['Gross P/L']:.2f}")
                        with col2:
                            st.metric(get_text("total_commission_metric", lang), f"-${best_row['Commission']:.2f}")
                        with col3:
                            st.metric(get_text("net_pl_metric", lang), f"{best_row['Total P/L']:.2f}")
                        with col4:
                            commission_impact_pct = (best_row['Commission'] / abs(best_row['Gross P/L']) * 100) if best_row['Gross P/L'] != 0 else 0
                            st.metric(get_text("commission_impact_pct", lang), f"{commission_impact_pct:.1f}%")
                        
                        st.info(get_text("commission_impact_info", lang, 
                                       contracts=int(contract_size),
                                       commission_per_rt=total_commission,
                                       total_trades=len(best_row['sim_data'])))
                    
                    st.markdown("---")
                    
                    # Visualizations (shown below tabs)
                    st.markdown(get_text("visualizations", lang))
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Histogram
                        fig, ax = plt.subplots(figsize=(8, 6))
                        best_data = best_row['sim_data']['net_pl_points']
                        ax.hist(best_data, bins=30, alpha=0.7, color='blue', edgecolor='black')
                        ax.axvline(best_data.mean(), color='red', linestyle='--', 
                                  label=get_text("mean_label", lang, value=best_data.mean()))
                        ax.set_xlabel('Net P/L (points)')
                        ax.set_ylabel(get_text("frequency", lang))
                        ax.set_title(get_text("pl_distribution", lang, tp=best_row["TP"], sl=best_row["SL"]))
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    with col2:
                        # Equity curve
                        fig, ax = plt.subplots(figsize=(8, 6))
                        equity = best_row['sim_data']['net_pl_points'].cumsum()
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
                    
                    # Export options in columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv = results_df[export_cols].to_csv(index=False)
                        st.download_button(
                            label=get_text("download_csv", lang),
                            data=csv,
                            file_name="inverse_edge_results.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        pdf_buffer = generate_pdf_report(results_df, best_row, df, lang)
                        st.download_button(
                            label=get_text("download_pdf", lang),
                            data=pdf_buffer,
                            file_name="inverse_edge_report.pdf",
                            mime="application/pdf"
                        )
                    
                    # Store in state for LLM with current bracket info
                    st.session_state.df = best_row['sim_data']
                    st.session_state.TP = best_row['TP']
                    st.session_state.SL = best_row['SL']
                    st.session_state.results_df = results_df
                    
                    # Calculation Preview Section
                    try:
                        st.markdown("---")
                        st.header("📊 " + get_text("calculation_preview", lang))
                        
                        with st.expander(get_text("show_calculation_details", lang), expanded=False):
                            st.markdown(get_text("calculation_preview_desc", lang))
                            
                            # Get the best bracket's simulation data
                            best_sim_data = best_row['sim_data']
                            
                            # Prepare preview data (first 100 trades)
                            preview_df = best_sim_data.head(100).copy()
                            
                            # Create display dataframe showing the calculation steps
                            display_df = pd.DataFrame({
                                get_text("trade_num", lang): range(1, len(preview_df) + 1),
                                get_text("original_pl", lang): preview_df['original_pl'].round(2),
                                get_text("runup", lang): preview_df['run-up (points)'].abs().round(2),
                                get_text("drawdown", lang): preview_df['drawdown (points)'].abs().round(2),
                                'D/R': preview_df['d/r flag'],
                                get_text("gross_inv_pl", lang): preview_df['gross_pl_points'].round(2),
                                get_text("commission", lang): preview_df['commission_points'].round(2),
                                get_text("net_inv_pl", lang): preview_df['net_pl_points'].round(2)
                            })
                            
                            # Add logic explanation column
                            logic_explanations = []
                            for _, row in preview_df.iterrows():
                                runup = abs(row['run-up (points)'])
                                drawdown = abs(row['drawdown (points)'])
                                dr_flag = row['d/r flag']
                                original_pl = row['original_pl']
                                gross_pl = row['gross_pl_points']
                                
                                if dr_flag == 'RP':
                                    if runup >= best_row['SL']:
                                        logic_explanations.append(f"RP: Run-up {runup:.1f} ≥ SL {best_row['SL']} → -{best_row['SL']}")
                                    elif drawdown >= best_row['TP']:
                                        logic_explanations.append(f"RP: Drawdown {drawdown:.1f} ≥ TP {best_row['TP']} → +{best_row['TP']}")
                                    else:
                                        logic_explanations.append(f"RP: No hit → -{original_pl:.1f}")
                                else:  # DD
                                    if drawdown >= best_row['TP']:
                                        logic_explanations.append(f"DD: Drawdown {drawdown:.1f} ≥ TP {best_row['TP']} → +{best_row['TP']}")
                                    elif runup >= best_row['SL']:
                                        logic_explanations.append(f"DD: Run-up {runup:.1f} ≥ SL {best_row['SL']} → -{best_row['SL']}")
                                    else:
                                        logic_explanations.append(f"DD: No hit → -{original_pl:.1f}")
                            
                            display_df[get_text("logic", lang)] = logic_explanations
                            
                            # Style the dataframe
                            def style_preview(row):
                                styles = [''] * len(row)
                                # Get the net P/L column index
                                net_pl_col = list(display_df.columns).index(get_text("net_inv_pl", lang))
                                # Highlight profitable trades in green
                                if row[get_text("net_inv_pl", lang)] > 0:
                                    styles[net_pl_col] = 'background-color: lightgreen'
                                elif row[get_text("net_inv_pl", lang)] < 0:
                                    styles[net_pl_col] = 'background-color: lightcoral'
                                return styles
                            
                            st.dataframe(
                                display_df.style.apply(style_preview, axis=1).format({
                                    get_text("original_pl", lang): '{:.2f}',
                                    get_text("runup", lang): '{:.2f}',
                                    get_text("drawdown", lang): '{:.2f}',
                                    get_text("gross_inv_pl", lang): '{:.2f}',
                                    get_text("commission", lang): '{:.2f}',
                                    get_text("net_inv_pl", lang): '{:.2f}'
                                }),
                                height=400,
                                use_container_width=True
                            )
                            
                            # Summary statistics for the preview
                            st.markdown("#### " + get_text("preview_summary", lang))
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                gross_pl_points = preview_df['gross_pl_points'].sum()
                                gross_pl_dollars = gross_pl_points * point_value * contract_size
                                st.metric(
                                    get_text("total_gross_pl", lang),
                                    f"{gross_pl_points:.2f} pts",
                                    delta=f"${gross_pl_dollars:.2f}"
                                )
                            
                            with col2:
                                total_comm_points = preview_df['commission_points'].sum()
                                total_comm_dollars = total_comm_points * point_value * contract_size
                                st.metric(
                                    get_text("total_commission", lang),
                                    f"{total_comm_points:.2f} pts",
                                    delta=f"-${total_comm_dollars:.2f}",
                                    delta_color="normal"
                                )
                            
                            with col3:
                                net_pl_points = preview_df['net_pl_points'].sum()
                                net_pl_dollars = net_pl_points * point_value * contract_size
                                st.metric(
                                    get_text("total_net_pl", lang),
                                    f"{net_pl_points:.2f} pts",
                                    delta=f"${net_pl_dollars:.2f}"
                                )
                            
                            with col4:
                                preview_hit_rate = (preview_df['net_pl_points'] > 0).sum() / len(preview_df) * 100
                                st.metric(
                                    get_text("hit_rate", lang),
                                    f"{preview_hit_rate:.1f}%"
                                )
                            
                            st.info(get_text("calculation_note", lang, tp=best_row['TP'], sl=best_row['SL']))
                    except Exception as e:
                        st.error(f"Error in calculation preview: {str(e)}")
                        st.write("Error details:", type(e).__name__)
                        import traceback
                        st.text(traceback.format_exc())
                    
# LLM Assistant
st.header(get_text("ai_assistant", lang))
st.info(get_text("ai_info", lang))

# Perplexity API configuration
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

# Check if API key is configured
if not PERPLEXITY_API_KEY:
    st.warning(get_text("api_key_missing", lang))
    st.info("Please set the PERPLEXITY_API_KEY environment variable to enable AI assistance.")

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