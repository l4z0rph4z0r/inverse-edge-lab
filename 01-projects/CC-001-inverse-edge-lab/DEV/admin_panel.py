# admin_panel.py
import streamlit as st
from auth_manager import AuthManager
from translations import get_text
from datetime import datetime
import pytz

def show_admin_panel(auth_manager: AuthManager, admin_username: str, lang: str):
    """Display the admin panel for managing access tokens"""
    
    st.header(get_text("admin_panel", lang))
    
    # User Statistics
    with st.expander(get_text("user_stats", lang), expanded=True):
        stats = auth_manager.get_user_stats()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(get_text("total_users", lang), stats["total_users"])
        with col2:
            st.metric(get_text("active_tokens_count", lang), stats["active_tokens"])
        with col3:
            st.metric(get_text("total_tokens_count", lang), stats["total_tokens"])
    
    # Token Generation
    st.subheader(get_text("generate_token", lang))
    
    with st.form("generate_token_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            token_name = st.text_input(
                get_text("token_name", lang),
                placeholder="e.g., John Doe Access"
            )
        
        with col2:
            expires_in = st.number_input(
                get_text("expires_in_days", lang),
                min_value=0,
                value=0,
                help="0 = " + get_text("unlimited", lang)
            )
        
        with col3:
            max_uses = st.number_input(
                get_text("max_uses", lang),
                min_value=0,
                value=0,
                help="0 = " + get_text("unlimited", lang)
            )
        
        submitted = st.form_submit_button(get_text("generate_token", lang))
        
        if submitted and token_name:
            token, token_data = auth_manager.create_access_token(
                created_by=admin_username,
                token_name=token_name,
                expires_in_days=expires_in if expires_in > 0 else None,
                max_uses=max_uses if max_uses > 0 else None
            )
            
            st.success(get_text("token_created", lang))
            st.info(get_text("copy_token", lang))
            st.code(token, language=None)
    
    # Token Management
    st.subheader(get_text("manage_tokens", lang))
    
    # Get all tokens
    tokens = auth_manager.list_tokens(admin_username)
    
    # Separate active and revoked tokens
    active_tokens = [t for t in tokens if t.get("is_active", True)]
    revoked_tokens = [t for t in tokens if not t.get("is_active", True)]
    
    # Display active tokens
    if active_tokens:
        st.write(f"### {get_text('active_tokens', lang)} ({len(active_tokens)})")
        
        for token in active_tokens:
            with st.expander(f"📌 {token['name']} - {token['token'][:8]}..."):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Token details
                    st.write(f"**{get_text('created_at', lang)}:** {format_datetime(token['created_at'])}")
                    
                    if token.get('expires_at'):
                        st.write(f"**{get_text('expires', lang)}:** {format_datetime(token['expires_at'])}")
                    else:
                        st.write(f"**{get_text('expires', lang)}:** {get_text('unlimited', lang)}")
                    
                    if token.get('max_uses'):
                        st.write(f"**{get_text('uses', lang)}:** {token.get('uses', 0)}/{token['max_uses']}")
                    else:
                        st.write(f"**{get_text('uses', lang)}:** {token.get('uses', 0)}/{get_text('unlimited', lang)}")
                    
                    if token.get('linked_user'):
                        st.write(f"**{get_text('linked_user', lang)}:** {token['linked_user']}")
                    
                    if token.get('last_used'):
                        st.write(f"**{get_text('last_used', lang)}:** {format_datetime(token['last_used'])}")
                
                with col2:
                    if st.button(get_text("revoke", lang), key=f"revoke_{token['token']}"):
                        if auth_manager.revoke_token(token['token'], admin_username):
                            st.success(get_text("token_revoked", lang))
                            st.rerun()
    
    # Display revoked tokens
    if revoked_tokens:
        with st.expander(f"{get_text('revoked_tokens', lang)} ({len(revoked_tokens)})"):
            for token in revoked_tokens:
                st.write(f"🚫 **{token['name']}** - {token['token'][:8]}...")
                if token.get('revoked_at'):
                    st.write(f"   {get_text('revoked', lang)}: {format_datetime(token['revoked_at'])}")

def format_datetime(dt_string: str) -> str:
    """Format datetime string for display"""
    try:
        dt = datetime.fromisoformat(dt_string)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except:
        return dt_string