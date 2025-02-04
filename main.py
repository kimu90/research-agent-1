import streamlit as st
import os
import logging
logging.getLogger('watchdog.observers.inotify_buffer').setLevel(logging.WARNING)
from datetime import datetime

from research_components.database import get_db_connection
from research_components.research import run_tool
from research_components.components import (
    display_analytics, 
    display_analysis,
    display_general_analysis,
    display_prompt_tracking,  # Add this import
    GeneralAgent, 
    PromptManager
)
from research_components.utils import setup_logging, load_research_history
from research_components.styles import apply_custom_styles

def main():
    # Configure the page
    st.set_page_config(
        page_title="Research Agent Analysis",
        layout="wide"
    )

    # Dark mode handling at the start
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False

    # Optional dark mode (moved from sidebar)
    if st.session_state.dark_mode:
        apply_custom_styles()

    # Initialize PromptManager
    prompt_manager = PromptManager(
        agent_type="general",
        config_path="prompts"
    )
    
    # Get available prompts
    available_prompts = prompt_manager.list_prompts()
    
    # Title and main layout
    st.title("🔍 Research Agent Analysis")

    # Main tabs for the entire interface
    main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
        "General Analytics", 
        "Data Analysis Agent", 
        "Summary Agent", 
        "Prompt Tracking"
    ])
    
    with main_tab1:
        # Load traces and display analytics immediately
        traces = load_research_history()
        display_general_analysis(traces)

    with main_tab2:
        # Load traces and display analytics immediately
        traces = load_research_history()
        db = get_db_connection()
        display_analysis(traces, db)

    with main_tab3:
        # Load traces and display analytics immediately
        traces = load_research_history()
        db = get_db_connection()
        display_analytics(traces, db)

    with main_tab4:
        # Prompt Tracking tab
        db = get_db_connection()
        display_prompt_tracking(db)

    # Footer
    st.markdown("---")
    st.markdown("""
    *Powered by AI Research Tools* | 
    [Documentation](https://github.com/yourusername/research-agent/docs)
    """)

# Main entry point
if __name__ == "__main__":
    # Set up logging
    setup_logging()
    
    # Run the main application
    try:
        main()
    except Exception as e:
        logging.error(f"Critical error in main application: {str(e)}")
        st.error(f"A critical error occurred: {str(e)}")