import streamlit as st
import logging
import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Import research components
from research_agent import ResearchAgent
from tools import (
    MarineAgent, 
    GeneralAgent,
    AmazonAgent
    )
from research_agent.db.db import ContentDB
from research_agent.tracers import CustomTracer
from tools.research.common.model_schemas import ContentItem, ResearchToolOutput

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
    handlers=[
        logging.FileHandler("research_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize database connection
try:
    db = ContentDB(os.environ.get('DB_PATH', '/data/content.db'))
    logger.info("Database connection initialized successfully")
except Exception as e:
    logger.error(f"Error initializing database: {str(e)}")
    db = None

def capture_processing_steps(trace: Dict[str, Any], step_description: str) -> Dict[str, Any]:
    """Helper method to capture processing steps"""
    if 'processing_steps' not in trace:
        trace['processing_steps'] = []
    trace['processing_steps'].append(step_description)
    return trace

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

def save_trace(trace_data):
    """Save research trace to file"""
    try:
        with open('research_traces.jsonl', 'a') as f:
            f.write(json.dumps(trace_data) + '\n')
        logger.info(f"Trace saved successfully: {trace_data.get('query')}")
    except Exception as e:
        logger.error(f"Error saving trace: {str(e)}")

def run_tool(tool_name: str, query: str):
    """Run a specific research tool and track its execution"""
    context = StreamlitContext()
    start_time = datetime.now()
    
    trace = {
        "trace_id": str(os.urandom(16).hex()),
        "timestamp": start_time.isoformat(),
        "tool": tool_name,
        "query": query,
        "start_time": start_time.isoformat(),
        "tools_used": [tool_name],
        "duration": 0,
        "error": None,
        "success": False,
        "processing_steps": [],
        "content_new": 0,
        "content_reused": 0
    }
    
    # Capture initial step
    trace = capture_processing_steps(trace, f"Started research with {tool_name}")
    
    try:
        # Capture tool initialization step
        trace = capture_processing_steps(trace, "Initializing search tool")
        
        if tool_name == "Marine Agent":
            tool = MarineAgent(include_summary=True)
            trace = capture_processing_steps(trace, "Configured Marine Search")
            result = tool.invoke(input={"query": query})
        
         
        
        elif tool_name == "General Agent":
            tool = GeneralAgent(include_summary=True)
            trace = capture_processing_steps(trace, "Configured GeneralSearch")
            result = tool.invoke(input={"query": query})
            
        elif tool_name == "Amazon Agent":
            tool = AmazonAgent(include_summary=True)
            trace = capture_processing_steps(trace, "Configured Amazon Agent")
            result = tool.invoke(input={"query": query})
            
        
        
        else:
            st.error(f"Tool {tool_name} not found")
            trace = capture_processing_steps(trace, f"Error: Tool {tool_name} not found")
            return None, trace

        # Content processing steps
        if result and result.content and db:
            trace = capture_processing_steps(trace, f"Preparing to process {len(result.content)} content items")
            
            # Track new vs. reused content
            new_content = 0
            reused_content = 0
            
            for item in result.content:
                try:
                    # Attempt to upsert document and track if it's new or existing
                    is_new = db.upsert_doc(item)
                    if is_new:
                        new_content += 1
                    else:
                        reused_content += 1
                except Exception as e:
                    logger.error(f"Error storing results: {str(e)}")
                    st.warning(f"Could not save results to database: {str(e)}")
                    trace = capture_processing_steps(trace, f"Database storage error: {str(e)}")

            # Update trace with content tracking
            trace["content_new"] = new_content
            trace["content_reused"] = reused_content
            trace = capture_processing_steps(trace, 
                f"Content processed - New: {new_content}, Reused: {reused_content}"
            )

        # Update trace with success information
        end_time = datetime.now()
        trace["duration"] = (end_time - start_time).total_seconds()
        trace["success"] = True
        trace["content_count"] = len(result.content) if result and result.content else 0
        
        # Final success step
        trace = capture_processing_steps(trace, "Research completed successfully")
        
        save_trace(trace)
        
        return result, trace
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error running {tool_name}: {error_msg}")
        st.error(f"Error running {tool_name}: {error_msg}")
        
        # Update trace with error information
        end_time = datetime.now()
        trace["duration"] = (end_time - start_time).total_seconds()
        trace["error"] = error_msg
        trace["success"] = False
        
        # Detailed error tracking
        trace = capture_processing_steps(trace, f"Research failed: {error_msg}")
        save_trace(trace)
        
        return None, trace

def load_research_history():
    """Load and process research history"""
    try:
        if not os.path.exists('research_traces.jsonl'):
            return []
            
        with open('research_traces.jsonl', 'r') as f:
            traces = [json.loads(line) for line in f]
            st.sidebar.markdown("### 📊 Statistics")
            st.sidebar.write(f"Total researches: {len(traces)}")
            
            # Add more detailed statistics
            if traces:
                st.sidebar.write(f"Successful queries: {len([t for t in traces if t.get('success', False)])}")
                st.sidebar.write(f"Average duration: {sum(t.get('duration', 0) for t in traces)/len(traces):.2f}s")
                st.sidebar.write(f"New content found: {sum(t.get('content_new', 0) for t in traces)}")
                st.sidebar.write(f"Reused content: {sum(t.get('content_reused', 0) for t in traces)}")
            
            return traces
    except Exception as e:
        logger.error(f"Error loading traces: {str(e)}")
        return []

def display_research_results(result, selected_tool):
    """Display research results with visualizations"""
    if result:
        tab1, tab2, tab3 = st.tabs(["Summary", "Detailed Content", "Analytics"])
        
        with tab1:
            st.subheader("Research Summary")
            st.markdown(result.summary)
        
        with tab2:
            for item in result.content:
                with st.expander(f"📄 {item.title}", expanded=True):
                    st.write(f"**URL:** {item.url}")
                    st.write(f"**Snippet:** {item.snippet}")
                    
                    if selected_tool == "Amazon Agent":
                        st.write("**Product Details:**")
                        details = item.content.split('\n')
                        for detail in details:
                            if detail.strip():
                                st.write(detail.strip())
                    
                    st.markdown("---")
                    meta_col1, meta_col2 = st.columns(2)
                    with meta_col1:
                        st.write(f"Source: {item.source}")
                    with meta_col2:
                        st.write(f"Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    if st.toggle("Show full content", key=f"content_{hash(item.url)}"):
                        st.text_area("Full Content", value=item.content, height=300)
        
        with tab3:
            display_analytics()

def enhance_trace_visualization():
    """Enhanced visualization of research traces"""
    traces = load_research_history()
    if not traces:
        st.info("No research history available yet. Run some searches to see detailed analytics!")
        return

    # Processing Steps Analysis
    st.subheader("🔄 Processing Steps Analysis")
    
    # Collect and analyze processing steps across all traces
    all_steps = []
    for trace in traces:
        if 'processing_steps' in trace:
            all_steps.extend(trace['processing_steps'])
    
    # Count step frequencies
    if all_steps:
        step_counts = {}
        for step in all_steps:
            step_counts[step] = step_counts.get(step, 0) + 1
        
        # Create a bar chart of processing step frequencies
        steps_df = pd.DataFrame.from_dict(
            step_counts, 
            orient='index', 
            columns=['Frequency']
        ).reset_index()
        steps_df.columns = ['Processing Step', 'Frequency']
        steps_df = steps_df.sort_values('Frequency', ascending=False)
        
        fig_steps = px.bar(
            steps_df, 
            x='Processing Step', 
            y='Frequency',
            title="Frequency of Processing Steps",
            labels={'Processing Step': 'Step', 'Frequency': 'Occurrence'}
        )
        fig_steps.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_steps, use_container_width=True, key="processing_steps_chart")
    else:
        st.info("No processing steps recorded yet.")


    # Detailed Processing Step Breakdown
    st.subheader("📋 Detailed Processing Steps")
    
    # Create an expandable section for each unique processing step
    unique_steps = list(set(all_steps))
    for step in unique_steps[:10]:  # Limit to top 10 to prevent overwhelming display
        with st.expander(f"Step: {step}"):
            # Find traces containing this step
            related_traces = [
                trace for trace in traces 
                if step in trace.get('processing_steps', [])
            ]
            
            # Display related trace details
            if related_traces:
                st.write(f"Traces involving this step: {len(related_traces)}")
                
                # Create a DataFrame of related traces
                step_traces_df = pd.DataFrame([
                    {
                        'Query': trace.get('query', 'N/A'),
                        'Tool': trace.get('tool', 'N/A'),
                        'Timestamp': trace.get('timestamp', 'N/A'),
                        'Success': '✅' if trace.get('success', False) else '❌'
                    } for trace in related_traces
                ])
                
                st.dataframe(step_traces_df)
            else:
                st.info("No detailed information available for this step.")

def display_analytics():
    """Display research analytics dashboard"""
    traces = load_research_history()
    if not traces:
        st.info("No research history available yet. Run some searches to see analytics!")
        return

    # Existing analytics from previous implementation
    col1, col2 = st.columns(2)
    
    with col1:
        # Tool usage distribution
        tool_usage = {}
        for trace in traces:
            tool = trace.get('tool', 'Unknown')
            tool_usage[tool] = tool_usage.get(tool, 0) + 1
        
        fig_tools = px.pie(
            values=list(tool_usage.values()),
            names=list(tool_usage.keys()),
            title="Research Tools Distribution"
        )
        st.plotly_chart(fig_tools, use_container_width=True, key="tools_distribution_chart")
    
    with col2:
        # Success rate gauge
        success_count = len([t for t in traces if t.get('success', False)])
        success_rate = (success_count / len(traces)) * 100 if traces else 0
        fig_success = go.Figure(go.Indicator(
            mode="gauge+number",
            value=success_rate,
            title={'text': "Success Rate (%)"},
            gauge={'axis': {'range': [None, 100]}}
        ))
        st.plotly_chart(fig_success, use_container_width=True, key="success_rate_chart")

    # Research timeline
    st.subheader("Research Timeline")
    df = pd.DataFrame(traces)
    df['start_time'] = pd.to_datetime(df['start_time'])
    fig_timeline = px.line(
        df,
        x='start_time',
        y='duration',
        title='Query Duration Timeline'
    )
    st.plotly_chart(fig_timeline, use_container_width=True, key="timeline_chart")

    # Statistics summary
    # Statistics summary
    st.subheader("Research Statistics")
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    with stats_col1:
        st.metric("Total Researches", len(traces))
    with stats_col2:
        avg_duration = df['duration'].mean() if not df.empty else 0
        st.metric("Average Duration", f"{avg_duration:.2f}s")
    with stats_col3:
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Enhanced trace visualization with processing steps
    enhance_trace_visualization()

def main():
    st.set_page_config(page_title="Research Agent Dashboard", layout="wide")
    
    st.title("🔍 Research Agent Dashboard")
    st.markdown("""
    Explore the world of information with our intelligent research tools. 
    Select a tool from the sidebar and discover insights!
    """)

    with st.sidebar:
        st.title("🛠 Research Tools")
        
        tool_options = [
            
            "General Agent",
            "Amazon Agent",
            "Marine Agent",
            
        ]
        selected_tool = st.selectbox("Choose a Research Tool", tool_options)
        
       
        

    # Main research interface
    query = st.text_input("Enter your research query:", key="query_input")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        search_button = st.button("Run Research", type="primary")
    
    if search_button and query:
        with st.spinner(f"Researching with {selected_tool}..."):
            try:
                result, trace = run_tool(selected_tool, query)
                
                if result:
                    # Display research results
                    display_research_results(result, selected_tool)
                    # Display trace information
                    st.markdown("### 🔍 Research Trace")
                    
                    # Trace details
                    trace_col1, trace_col2 = st.columns(2)
                    with trace_col1:
                        st.write(f"**Tool Used:** {trace.get('tool', 'N/A')}")
                        st.write(f"**Query:** {trace.get('query', 'N/A')}")
                        st.write(f"**Timestamp:** {trace.get('timestamp', 'N/A')}")
                    
                    with trace_col2:
                        st.write(f"**Duration:** {trace.get('duration', 'N/A'):.2f} seconds")
                        st.write(f"**Success:** {'✅' if trace.get('success', False) else '❌'}")
                    
                    # Processing steps
                    if trace.get('processing_steps'):
                        st.subheader("Processing Steps")
                        processing_steps_container = st.container()
                        with processing_steps_container:
                            for step in trace['processing_steps']:
                                st.write(f"- {step}")
                    
                    # Error handling
                    if not trace.get('success', False):
                        st.error(f"Error Details: {trace.get('error', 'Unknown error')}")
                    
                    # Content summary
                    st.subheader("Content Summary")
                    content_col1, content_col2 = st.columns(2)
                    with content_col1:
                        st.metric("New Content", trace.get('content_new', 0))
                    with content_col2:
                        st.metric("Reused Content", trace.get('content_reused', 0))
                    
            except Exception as e:
                logger.error(f"Error during research: {str(e)}")
                st.error(f"An error occurred: {str(e)}")
    
    elif search_button and not query:
        st.warning("Please enter a query")

    
    # Footer
    st.markdown("---")
    st.markdown("""
    *Powered by AI Research Tools* | 
    [Documentation](https://github.com/yourusername/research-agent/docs)
    """)

    # Add a sidebar for additional controls and information
    with st.sidebar:
        st.markdown("### ℹ️ Research Agent Info")
        st.write("Version: 1.0.0")
        st.write("Last Updated: January 2025")
        
        # Trace management
        st.markdown("### 🗂️ Trace Management")
        if st.button("Clear Research History"):
            try:
                # Safely remove trace file
                if os.path.exists('research_traces.jsonl'):
                    os.remove('research_traces.jsonl')
                st.success("Research history cleared successfully!")
            except Exception as e:
                st.error(f"Error clearing history: {str(e)}")

# Logging configuration
def setup_logging():
    """Set up comprehensive logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('research_agent.log'),
            logging.StreamHandler()
        ]
    )
    
    # Add custom log levels if needed
    logging.addLevelName(logging.INFO, "🔵 INFO")
    logging.addLevelName(logging.WARNING, "🟠 WARNING")
    logging.addLevelName(logging.ERROR, "🔴 ERROR")

# Main entry point
if __name__ == "__main__":
    # Set up logging
    setup_logging()
    
    # Run the main application
    try:
        main()
    except Exception as e:
        logger.error(f"Critical error in main application: {str(e)}")
        st.error(f"A critical error occurred: {str(e)}")
        # Optionally, add more robust error handling or reporting