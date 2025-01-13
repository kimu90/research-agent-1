import streamlit as st
import logging
import os
from dotenv import load_dotenv

# Import your research tools
from tools.research import (
    YouComSearch, 
    SimilarWebSearch, 
    ExaCompanySearch, 
    NewsSearch,
    AmazonSearch  # Added Amazon Search
)

# Set USER_AGENT

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

# Create a mock context for Streamlit
class StreamlitContext:
    def new_message(self):
        return StreamlitMessage()

class StreamlitMessage:
    def add(self, type, text=""):
        if type == "text":
            st.write(text)
        return self

    def notify(self):
        pass

def run_tool(tool_name, query, additional_params=None):
    """
    Run a specific research tool based on its name
    """
    context = StreamlitContext()
    
    try:
        if tool_name == "YouCom Search":
            tool = YouComSearch(include_summary=True)
            result = tool.invoke(input={"query": query})
        
        elif tool_name == "SimilarWeb Search":
            tool = SimilarWebSearch(include_summary=True)
            result = tool.invoke(input={
                "query": query,
                "entity_name": additional_params.get('entity_name', query)
            })
        
        elif tool_name == "Exa Company Search":
            tool = ExaCompanySearch(include_summary=True)
            result = tool.invoke(input={"query": query})
        
        elif tool_name == "News Search":
            tool = NewsSearch(include_summary=True)
            result = tool.invoke(input={"query": query})
            
        elif tool_name == "Amazon Search":  # Added Amazon Search handling
            tool = AmazonSearch(include_summary=True)
            result = tool.invoke(input={"query": query})
        
        else:
            st.error(f"Tool {tool_name} not found")
            return None

        return result
    
    except Exception as e:
        st.error(f"Error running {tool_name}: {e}")
        return None

def main():
    st.set_page_config(page_title="Research Agent Dashboard", layout="wide")
    
    # Title and description
    st.title("🔍 Research Agent Dashboard")
    st.markdown("""
    Explore the world of information with our intelligent research tools. 
    Select a tool from the sidebar, enter your query, and discover insights!
    """)

    # Sidebar for tool selection and configuration
    with st.sidebar:
        st.title("🛠 Research Tools")
        
        # Tool selection dropdown
        tool_options = [
            "YouCom Search", 
            "SimilarWeb Search", 
            "Exa Company Search", 
            "News Search",
            "Amazon Search"  # Added Amazon Search option
        ]
        selected_tool = st.selectbox("Choose a Research Tool", tool_options)
        
        # Additional parameters based on tool selection
        additional_params = {}
        if selected_tool == "SimilarWeb Search":
            additional_params['entity_name'] = st.text_input(
                "Entity Name", 
                help="Specific entity to search on SimilarWeb"
            )
        elif selected_tool == "Amazon Search":  # Added Amazon-specific parameters if needed
            additional_params['include_reviews'] = st.checkbox(
                "Include Customer Reviews",
                help="Include detailed customer reviews in the results"
            )
        
        # API Key checks
        st.markdown("### 🔑 API Status")
        api_checks = {
            "YouCom": os.getenv("YOUCOM_API_KEY") is not None,
            "EXA": os.getenv("EXA_API_KEY") is not None,
            "Brave": os.getenv("BRAVE_SEARCH_API_KEY") is not None,
            "OpenAI": os.getenv("OPENAI_API_KEY") is not None,
            "Groq": os.getenv("GROQ_API_KEY") is not None,
            "Rainforest": os.getenv("RAINFOREST_API_KEY") is not None  # Added Rainforest API check
        }
        
        for api, status in api_checks.items():
            if status:
                st.success(f"{api} API: Connected ✅")
            else:
                st.warning(f"{api} API: Not Configured ⚠️")

    # Main query input
    query = st.text_input(
        "Enter your search query" if selected_tool != "Amazon Search" 
        else "Enter product search query",
        key="query_input"
    )
    
    # Search button with improved styling
    col1, col2 = st.columns([3, 1])
    with col2:
        search_button = st.button("Run Research", type="primary")
    
    # Results display
    if search_button and query:
        # Clear previous results
        st.empty()
        
        # Show loading spinner
        with st.spinner(f"Researching with {selected_tool}..."):
            # Run the selected tool
            result = run_tool(selected_tool, query, additional_params)
            
            # Display results
            if result:
                # Create tabs for different views
                tab1, tab2 = st.tabs(["Summary", "Detailed Content"])
                
                with tab1:
                    st.subheader("Research Summary")
                    st.write(result.summary)
                
                with tab2:
                    for item in result.content:
                        with st.expander(f"📄 {item.title}"):
                            st.write(f"**URL:** {item.url}")
                            st.write(f"**Snippet:** {item.snippet}")
                            
                            # Special handling for Amazon results
                            if selected_tool == "Amazon Search":
                                st.write("**Product Details:**")
                                details = item.content.split('\n')
                                for detail in details:
                                    if detail.strip():
                                        st.write(detail.strip())
                            
                            # Full content toggle
                            if st.toggle(f"Show full content", key=f"content_{hash(item.url)}"):
                                st.text_area("Full Content", value=item.content, height=300)
            else:
                st.warning("No results found or an error occurred.")
        
    elif search_button and not query:
        st.warning("Please enter a query")

    # Footer
    st.markdown("---")
    st.markdown("""
    *Powered by AI Research Tools* | 
    [GitHub Repository](https://github.com/yourusername/research-agent)
    """)

if __name__ == "__main__":
    main()