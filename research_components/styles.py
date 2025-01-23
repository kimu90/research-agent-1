import streamlit as st

def apply_custom_styles():
    st.markdown("""
        <style>
            /* Dark mode styles - abbreviated version */
            .stApp { background-color: #1E1E1E; color: #FFFFFF; }
            header[data-testid="stHeader"] { background-color: #1E1E1E; }
            [data-testid="stSidebar"] { background-color: #262626; }
            div[data-testid="stMetricValue"] { 
                background-color: #2C2C2C; 
                color: #FFFFFF; 
                padding: 1rem; 
                border-radius: 0.5rem; 
            }
            .stButton button { 
                background-color: #4A4A4A; 
                color: #FFFFFF; 
            }
            /* Additional minimal dark mode styles */
        </style>
    """, unsafe_allow_html=True)