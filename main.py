
import pdb
import streamlit as st
import logging
import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore
import pandas as pd
# Import research components
from research_agent import ResearchAgent
from tools import GeneralAgent
from research_agent.db.db import ContentDB
from research_agent.tracers import CustomTracer, QueryTrace
from tools.research.common.model_schemas import ContentItem, ResearchToolOutput
from prompts.prompt_manager import PromptManager
from utils.token_tracking import TokenUsageTracker

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

   
def run_tool(tool_name: str, query: str, tool=None):
    """Run a specific research tool and track its execution and token usage"""
    pdb.set_trace()  # Breakpoint at the start of run_tool
    context = StreamlitContext()
    start_time = datetime.now()
    
    # Configure logging with more detail
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s - %(filename)s:%(lineno)d'
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting research tool execution - Tool: {tool_name}")
    logger.info(f"Query received: {query}")
    
    # Create a new QueryTrace
    trace = QueryTrace(query)
    trace.data["tool"] = tool_name
    trace.data["tools_used"] = [tool_name]
    trace.data["processing_steps"] = []
    trace.data["content_new"] = 0
    trace.data["content_reused"] = 0
    
    logger.info(f"Query trace initialized - ID: {trace.id if hasattr(trace, 'id') else 'N/A'}")
    
    # Capture initial step
    trace.data["processing_steps"].append(f"Started research with {tool_name}")
    
    try:
        # Capture tool initialization step
        logger.info("Initializing research tool")
        trace.data["processing_steps"].append("Initializing search tool")
        
        if tool_name == "General Agent":
            # Use the passed tool if available, otherwise create a new one
            if tool is None:
                logger.info("Creating GeneralAgent instance")
                tool = GeneralAgent(include_summary=True)
            
            logger.info("Recording prompt usage")
            trace.add_prompt_usage("general_agent_search", "general", "")
            
            logger.info("Invoking GeneralAgent with query")
            result = tool.invoke(input={"query": query})
            logger.info("GeneralAgent invocation completed")
            
        else:
            error_msg = f"Tool {tool_name} not found"
            logger.error(error_msg)
            st.error(error_msg)
            trace.data["processing_steps"].append(f"Error: {error_msg}")
            return None, trace

        # Content processing steps
        if result and result.content and db:
            content_count = len(result.content)
            logger.info(f"Processing {content_count} content items")
            trace.data["processing_steps"].append(f"Preparing to process {content_count} content items")
            
            # Track new vs. reused content
            new_content = 0
            reused_content = 0
            
            for idx, item in enumerate(result.content, 1):
                try:
                    logger.info(f"Processing content item {idx}/{content_count}")
                    # Attempt to upsert document and track if it's new or existing
                    is_new = db.upsert_doc(item)
                    if is_new:
                        new_content += 1
                        logger.info(f"New content item stored (ID: {getattr(item, 'id', 'N/A')})")
                    else:
                        reused_content += 1
                        logger.info(f"Existing content item updated (ID: {getattr(item, 'id', 'N/A')})")
                except Exception as e:
                    error_detail = f"Error storing content item {idx}: {str(e)}"
                    logger.error(error_detail, exc_info=True)
                    st.warning(f"Could not save results to database: {str(e)}")
                    trace.data["processing_steps"].append(f"Database storage error: {error_detail}")

            # Update trace with content tracking
            logger.info(f"Content processing completed - New: {new_content}, Reused: {reused_content}")
            trace.data["content_new"] = new_content
            trace.data["content_reused"] = reused_content
            trace.data["processing_steps"].append(
                f"Content processed - New: {new_content}, Reused: {reused_content}"
            )

        # Update trace with success information
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Processing completed in {duration:.2f} seconds")
        
        trace.data["duration"] = duration
        trace.data["success"] = True
        trace.data["content_count"] = len(result.content) if result and result.content else 0
        
        # Log final token usage
        token_stats = trace.token_tracker.get_usage_stats()
        logger.info(f"Final token usage stats: {token_stats}")
        
        if token_stats['tokens']['total'] > 0:
            usage_msg = f"Total tokens used: {token_stats['tokens']['total']}"
            logger.info(usage_msg)
            trace.data["processing_steps"].append(usage_msg)
        
        # Final success step
        logger.info("Research completed successfully")
        trace.data["processing_steps"].append("Research completed successfully")
        trace.data["end_time"] = datetime.now().isoformat()
        
        # Save the trace
        logger.info("Saving successful trace")
        tracer = CustomTracer()
        tracer.save_trace(trace)
        
        return result, trace
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error running {tool_name}: {error_msg}", exc_info=True)
        st.error(f"Error running {tool_name}: {error_msg}")
        
        # Update trace with error information
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Failed processing duration: {duration:.2f} seconds")
        
        trace.data["end_time"] = end_time.isoformat()
        trace.data["duration"] = duration
        trace.data["error"] = error_msg
        trace.data["success"] = False
        
        # Detailed error tracking
        error_step = f"Research failed: {error_msg}"
        logger.error(error_step)
        trace.data["processing_steps"].append(error_step)
        
        # Save the error trace
        logger.info("Saving error trace")
        tracer = CustomTracer()
        tracer.save_trace(trace)
        
        return None, trace
def update_token_stats(trace: QueryTrace, prompt_tokens: int, completion_tokens: int, 
                      model: str, prompt_id: Optional[str] = None) -> None:
    """
    Update token usage statistics in QueryTrace
    """
    try:
        # Ensure token tracker is initialized
        if not hasattr(trace, 'token_tracker'):
            trace.token_tracker = TokenUsageTracker()
        
        # Force update token usage
        trace.add_token_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_name=model,
            prompt_id=prompt_id
        )
        
        # Debug logging
        print("DEBUG: Updated token usage stats:")
        print(json.dumps(trace.token_tracker.get_usage_stats(), indent=2))
    except Exception as e:
        logging.error(f"Error updating token stats: {str(e)}")

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
                    
                    st.markdown("---")
                    meta_col1, meta_col2 = st.columns(2)
                    with meta_col1:
                        st.write(f"Source: {item.source}")
                    with meta_col2:
                        st.write(f"Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    if st.toggle("Show full content", key=f"content_{hash(item.url)}"):
                        st.text_area("Full Content", value=item.content, height=300)
        
        with tab3:
            # Load traces for analytics
            traces = load_research_history()
            display_analytics(traces)

def display_prompt_analytics(traces: List[QueryTrace]):
    """Display prompt usage analytics"""
    st.subheader("🔄 Prompt Usage Analysis")
    
    if not traces:
        st.info("No prompt history available yet.")
        return

    # Collect prompt usage data 
    prompt_usage = []
    for trace in traces:
        prompt_usage.extend(trace.data.get("prompts_used", []))

    if prompt_usage:
        try:
            # Create DataFrame and convert timestamp to datetime
            df = pd.DataFrame(prompt_usage)
            
            # Convert timestamp column to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Now group by timestamp
            usage_over_time = (
                df.groupby([
                    pd.Grouper(key='timestamp', freq='D'),
                    'prompt_id'
                ])
                .size()
                .reset_index(name='count')
            )

            # Create timeline chart
            fig_timeline = px.line(
                usage_over_time, 
                x='timestamp', 
                y='count', 
                color='prompt_id', 
                title='Prompt Usage Over Time',
                labels={'prompt_id': 'Prompt ID'}
            )
            st.plotly_chart(fig_timeline)
            
            # Most used prompts  
            st.subheader("Most Used Prompts")
            prompt_counts = df['prompt_id'].value_counts()
            fig_prompts = px.bar(
                prompt_counts, 
                x=prompt_counts.index, 
                y=prompt_counts.values,
                title='Most Used Prompts', 
                labels={'x': 'Prompt ID', 'y': 'Usage Count'}
            )
            st.plotly_chart(fig_prompts)
            
        except Exception as e:
            st.error(f"Error processing prompt analytics: {str(e)}")
            logging.error(f"Error in prompt analytics: {str(e)}")
            
            # Fallback to simple statistics
            st.write("Basic Prompt Usage Statistics:")
            prompt_ids = [p['prompt_id'] for p in prompt_usage]
            for prompt_id in set(prompt_ids):
                count = prompt_ids.count(prompt_id)
                st.write(f"- {prompt_id}: {count} uses")

def get_token_usage(trace: QueryTrace) -> Dict[str, Any]:
    """
    Get token usage statistics from a trace
    Args:
        trace: QueryTrace object to get stats from
    Returns:
        Dict containing token usage statistics
    """
    # Always use TokenUsageTracker as source of truth
    if hasattr(trace, 'token_tracker'):
        token_stats = trace.token_tracker.get_usage_stats()
        
        # Add debug logging
        print(f"DEBUG: Getting token usage for trace {trace.trace_id}")
        print(json.dumps(token_stats, indent=2))
        
        return token_stats
    
    # Fallback for older traces or error cases
    logging.warning(f"No TokenUsageTracker found for trace {trace.trace_id}")
    return {
        'total_usage': {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        },
        'usage_by_model': {},
        'usage_by_prompt': {},
        'usage_timeline': []
    }

def load_research_history() -> List[QueryTrace]:
    """
    Load and process research history into QueryTrace objects
    Returns:
        List of QueryTrace objects
    """
    try:
        if not os.path.exists('research_traces.jsonl'):
            return []
            
        traces = []
        with open('research_traces.jsonl', 'r') as f:
            for line in f:
                trace_data = json.loads(line)
                # Create QueryTrace object and populate it
                trace = QueryTrace(trace_data['query'])
                trace.data = trace_data
                # Initialize token tracker with historical data if available
                if 'token_usage' in trace_data:
                    token_usage = trace_data['token_usage']
                    if 'total_usage' in token_usage:
                        trace.token_tracker.prompt_tokens = token_usage['total_usage'].get('prompt_tokens', 0)
                        trace.token_tracker.completion_tokens = token_usage['total_usage'].get('completion_tokens', 0)
                        trace.token_tracker.total_tokens = token_usage['total_usage'].get('total_tokens', 0)
                    trace.token_tracker.usage_by_model = token_usage.get('usage_by_model', {})
                    trace.token_tracker.usage_by_prompt = token_usage.get('usage_by_prompt', {})
                traces.append(trace)
        
        return traces
    except Exception as e:
        logging.error(f"Error loading traces: {str(e)}")
        return []

def enhance_trace_visualization(traces: List[QueryTrace]):
    """
    Enhanced visualization of research traces
    Args:
        traces: List of QueryTrace objects to visualize
    """
    if not traces:
        st.info("No research history available yet. Run some searches to see detailed analytics!")
        return

    # Processing Steps Analysis
    st.subheader("🔄 Processing Steps Analysis")
    
    # Collect and analyze processing steps across all traces
    all_steps = []
    for trace in traces:
        steps = trace.data.get('processing_steps', [])
        if isinstance(steps, list):
            all_steps.extend(steps)
    
    # Count step frequencies
    if all_steps:
        step_counts = {}
        for step in all_steps:
            step_counts[step] = step_counts.get(step, 0) + 1
        
        # Create a bar chart of processing step frequencies
        steps_df = pd.DataFrame(
            list(step_counts.items()),
            columns=['Processing Step', 'Frequency']
        ).sort_values('Frequency', ascending=False)
        
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
                if step in trace.data.get('processing_steps', [])
            ]
            
            # Display related trace details
            if related_traces:
                st.write(f"Traces involving this step: {len(related_traces)}")
                
                # Create a DataFrame of related traces
                step_traces_df = pd.DataFrame([
                    {
                        'Query': trace.data.get('query', 'N/A'),
                        'Tool': trace.data.get('tool', 'N/A'),
                        'Timestamp': trace.data.get('start_time', 'N/A'),
                        'Success': '✅' if trace.data.get('success', False) else '❌',
                        'Duration': f"{trace.data.get('duration', 0):.2f}s"
                    } for trace in related_traces
                ])
                
                st.dataframe(
                    step_traces_df,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No detailed information available for this step.")

def display_token_usage(trace: QueryTrace, show_visualizations: bool = True):
    """
    Display token usage for a single trace
    Args:
        trace: QueryTrace object containing token usage data
        show_visualizations: Boolean to control whether to show detailed visualizations
    """
    # Always get stats from TokenUsageTracker for consistency
    token_stats = trace.token_tracker.get_usage_stats()

    # Extensive debugging
    print("Full token_stats structure:")
    print(json.dumps(token_stats, indent=2))
    print("Type of token_stats:", type(token_stats))
    print("Keys in token_stats:", list(token_stats.keys()) if isinstance(token_stats, dict) else "Not a dictionary")

    # Defensive programming to handle different possible structures
    try:
        # Try to extract tokens using the expected structure
        total_tokens = token_stats.get('tokens', {}).get('total', 0)
        prompt_tokens = token_stats.get('tokens', {}).get('input', 0)
        completion_tokens = token_stats.get('tokens', {}).get('output', 0)
        model = token_stats.get('model', 'Unknown')

        st.subheader("Token Usage Summary")

        cols = st.columns(3)
        with cols[0]:
            st.metric(
                "Total Tokens",
                f"{total_tokens:,}",
                help="Total tokens used in this research"
            )

        with cols[1]:
            st.metric(
                "Prompt Tokens",
                f"{prompt_tokens:,}",
                help="Tokens used in prompts"
            )

        with cols[2]:
            st.metric(
                "Completion Tokens",
                f"{completion_tokens:,}",
                help="Tokens used in completions"
            )

        # Additional error handling for visualization
        if show_visualizations:
            # Simplified visualization
            st.subheader("Model Usage")
            st.write(f"Model: {model}")
            st.write(f"Total Tokens: {total_tokens}")

    except Exception as e:
        st.error(f"Error processing token usage: {str(e)}")
        print("Unexpected token_stats structure:", token_stats)


def display_analytics(traces: List[QueryTrace]):
   """
   Display research analytics dashboard with token usage metrics
   Args:
       traces: List[QueryTrace objects]
   """
   if not traces:
       st.info("No research history available yet. Run some searches to see analytics!")
       return

   tab1, tab2, tab3 = st.tabs(["General Analytics", "Token Usage", "Processing Steps"])
   
   with tab1:
       # Convert traces to DataFrame for analysis
       df = pd.DataFrame([
           {
               'date': datetime.fromisoformat(t.data['start_time']).date(),
               'success': t.data.get('success', False),
               'duration': t.data.get('duration', 0),
               'content_new': t.data.get('content_new', 0), 
               'content_reused': t.data.get('content_reused', 0),
               'query': t.data.get('query', '')
           }
           for t in traces
       ])
       
       # Success Rate Over Time
       st.subheader("📈 Success Rate Over Time")
       success_by_date = (
           df.groupby('date')
           .agg({
               'success': ['count', lambda x: x.sum() / len(x) * 100]
           })
           .reset_index()
       )
       success_by_date.columns = ['date', 'total', 'success_rate']
       
       fig_success_timeline = px.line(
           success_by_date,
           x='date',
           y='success_rate',
           title='Success Rate Trend',
           labels={'success_rate': 'Success Rate (%)', 'date': 'Date'}
       )
       fig_success_timeline.update_layout(yaxis_range=[0, 100])
       st.plotly_chart(fig_success_timeline, use_container_width=True)

       # Statistics Summary Cards
       st.subheader("📊 Research Statistics")
       stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
       
       with stats_col1:
           st.metric("Total Researches", len(traces))
       with stats_col2:
           avg_duration = df['duration'].mean() if not df.empty else 0
           st.metric("Average Duration", f"{avg_duration:.2f}s")
       with stats_col3:
           success_rate = (df['success'].sum() / len(df)) * 100 if not df.empty else 0
           st.metric("Success Rate", f"{success_rate:.1f}%")
       with stats_col4:
           total_content = df['content_new'].sum() + df['content_reused'].sum()
           st.metric("Total Content Processed", total_content)

   with tab2:
       st.subheader("🎯 Token Usage Analytics")
       
       # Collect token usage data from TokenUsageTracker
       token_data = []
       for trace in traces:
           try:
               stats = trace.token_tracker.get_usage_stats()
               timeline = trace.data.get('token_usage', {}).get('usage_timeline', [])
               
               for entry in timeline:
                   try:
                       # Use .get() with default values to avoid KeyError
                       timestamp = entry.get('timestamp')
                       if not timestamp:  # If no timestamp, use current time
                           timestamp = datetime.now().isoformat()
                           
                       token_data.append({
                           'timestamp': datetime.fromisoformat(timestamp),
                           'prompt_tokens': entry.get('prompt_tokens', 0),
                           'completion_tokens': entry.get('completion_tokens', 0),
                           'total_tokens': entry.get('prompt_tokens', 0) + entry.get('completion_tokens', 0),
                           'model': entry.get('model', 'unknown'),
                           'prompt_id': entry.get('prompt_id', 'Unknown'),
                           'query': trace.data.get('query', 'No query')
                       })
                   except Exception as e:
                       logging.warning(f"Error processing timeline entry: {str(e)}")
                       continue
                       
           except Exception as e:
               logging.warning(f"Error processing token data for trace {trace.trace_id}: {str(e)}")
               continue

       try:
           if token_data:
               token_df = pd.DataFrame(token_data)
               
               # Token Usage Statistics
               token_stats_col1, token_stats_col2, token_stats_col3 = st.columns(3)
               
               with token_stats_col1:
                   total_tokens = token_df['total_tokens'].sum()
                   st.metric(
                       "Total Tokens Used",
                       f"{total_tokens:,}",
                       help="Sum of all input and output tokens"
                   )
               
               with token_stats_col2:
                   avg_tokens = token_df.groupby('query')['total_tokens'].sum().mean()
                   st.metric(
                       "Avg Tokens per Query",
                       f"{avg_tokens:,.0f}",
                       help="Average token usage per research query"
                   )
               
               with token_stats_col3:
                   completion_sum = token_df['completion_tokens'].sum()
                   token_ratio = (token_df['prompt_tokens'].sum() / completion_sum 
                                if completion_sum > 0 else 0)
                   st.metric(
                       "Prompt/Completion Ratio",
                       f"{token_ratio:.2f}",
                       help="Ratio of prompt tokens to completion tokens"
                   )

               # Token Usage Over Time
               st.subheader("Token Usage Trends")
               token_df_daily = token_df.set_index('timestamp').resample('D').sum().reset_index()
               fig_token_timeline = px.line(
                   token_df_daily,
                   x='timestamp',
                   y=['prompt_tokens', 'completion_tokens', 'total_tokens'],
                   title='Daily Token Usage',
                   labels={
                       'timestamp': 'Date',
                       'value': 'Tokens',
                       'variable': 'Token Type'
                   }
               )
               st.plotly_chart(fig_token_timeline, use_container_width=True)

               # Token Usage by Model
               if len(token_df['model'].unique()) > 1:  # Only show if multiple models
                   st.subheader("Token Usage by Model")
                   model_usage = token_df.groupby('model').agg({
                       'total_tokens': 'sum',
                       'prompt_tokens': 'sum',
                       'completion_tokens': 'sum'
                   }).reset_index()
                   
                   fig_model_usage = px.bar(
                       model_usage,
                       x='model',
                       y=['prompt_tokens', 'completion_tokens'],
                       title='Token Distribution by Model',
                       barmode='stack',
                       labels={
                           'model': 'Model',
                           'value': 'Tokens',
                           'variable': 'Token Type'
                       }
                   )
                   st.plotly_chart(fig_model_usage, use_container_width=True)

               # Token Usage by Prompt Type
               st.subheader("Token Usage by Prompt Type")
               prompt_usage = token_df.groupby('prompt_id').agg({
                   'total_tokens': 'sum'
               }).sort_values('total_tokens', ascending=False).head(10)
               
               fig_prompt_usage = px.pie(
                   prompt_usage,
                   values='total_tokens',
                   names=prompt_usage.index,
                   title='Top 10 Prompts by Token Usage'
               )
               st.plotly_chart(fig_prompt_usage, use_container_width=True)

               # Most Token-Intensive Queries
               st.subheader("Most Token-Intensive Queries")
               query_usage = token_df.groupby('query')['total_tokens'].sum().sort_values(ascending=False).head(5)
               fig_query_usage = px.bar(
                   x=query_usage.index,
                   y=query_usage.values,
                   title='Top 5 Token-Intensive Queries',
                   labels={'x': 'Query', 'y': 'Total Tokens'}
               )
               fig_query_usage.update_layout(xaxis_tickangle=-45)
               st.plotly_chart(fig_query_usage, use_container_width=True)
           else:
               st.info("No token usage data available yet. Run some searches to see token analytics!")
       
       except Exception as e:
           logging.error(f"Error processing token analytics: {str(e)}")
           st.error(f"Error displaying token analytics: {str(e)}")

   with tab3:
       enhance_trace_visualization(traces)
# At the top of your script, before the main function
_db_instance = None

def get_db_connection():
    global _db_instance
    if _db_instance is None:
        try:
            _db_instance = ContentDB(os.environ.get('DB_PATH', '/data/content.db'))
            logger.info("Database connection initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            _db_instance = None
    return _db_instance

def main():
    pdb.set_trace()  # Breakpoint at the start of main function
    st.set_page_config(page_title="Research Agent Dashboard", layout="wide")
    
    st.title("🔍 Research Agent Dashboard")
    st.markdown("""
    Explore the world of information with our intelligent research tools. 
    Select a tool from the sidebar and discover insights!
    """)

    # Sidebar configuration
    with st.sidebar:
        st.title("🛠 Research Tools")
        
        # Tool Selection
        tool_options = ["General Agent"]
        selected_tool = st.selectbox("Choose a Research Tool", tool_options)
        pdb.set_trace()  # Breakpoint at the start of main function

        
        # Initialize PromptManager
        prompt_manager = PromptManager(
            agent_type="general",
            config_path="prompts"
        )
        
        # Get available prompts
        available_prompts = prompt_manager.list_prompts()
        
        # Add prompt selector if prompts are available
        if available_prompts:
            st.markdown("### 📝 Select Research Style")
            selected_prompt_id = st.selectbox(
                "Choose Research Approach",
                options=available_prompts,
                format_func=lambda x: prompt_manager.get_prompt(x).metadata.get('description', x)
            )
            pdb.set_trace()

            # Show prompt details
            if selected_prompt_id:
                prompt = prompt_manager.get_prompt(selected_prompt_id)
                with st.expander("Prompt Details"):
                    st.markdown(f"**Use Case:** {prompt.metadata.get('use_case', 'General research')}")
                    st.markdown(f"**Type:** {prompt.metadata.get('type', 'Not specified')}")
        else:
            st.warning("No prompts found. Please check the prompts directory.")

    # Main research interface
    query = st.text_input("Enter your research query:", key="query_input")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        search_button = st.button("Run Research", type="primary")
    
    if search_button and query:
        pdb.set_trace()
        with st.spinner(f"Researching with {selected_tool}..."):
            try:
                # Get selected prompt
                current_prompt = prompt_manager.get_prompt(selected_prompt_id)
                
                # Initialize tool with selected prompt
                tool = GeneralAgent(
                    include_summary=True,
                    custom_prompt=current_prompt
                )
                pdb.set_trace()

                # Pass the initialized tool to run_tool
                result, trace = run_tool(
                    tool_name=selected_tool, 
                    query=query, 
                    tool=tool
                )
                
                if result:
                    pdb.set_trace()

                    # Display research results
                    display_research_results(result, selected_tool)
                    pdb.set_trace()

                    # Display trace information
                    st.markdown("### 🔍 Research Trace")
                    
                    # Trace details
                    trace_col1, trace_col2 = st.columns(2)
                    with trace_col1:
                        st.write(f"**Tool Used:** {trace.data.get('tool', 'N/A')}")
                        st.write(f"**Query:** {trace.data.get('query', 'N/A')}")
                        st.write(f"**Timestamp:** {trace.data.get('start_time', 'N/A')}")
                    
                    with trace_col2:
                        st.write(f"**Duration:** {trace.data.get('duration', 'N/A'):.2f} seconds")
                        st.write(f"**Success:** {'✅' if trace.data.get('success', False) else '❌'}")
                    
                    # Processing steps
                    if trace.data.get('processing_steps'):
                        st.subheader("Processing Steps")
                        processing_steps_container = st.container()
                        with processing_steps_container:
                            for step in trace.data['processing_steps']:
                                st.write(f"- {step}")
                    
                    # Error handling
                    if not trace.data.get('success', False):
                        st.error(f"Error Details: {trace.data.get('error', 'Unknown error')}")
                    
                    # Content summary
                    st.subheader("Content Summary")
                    content_col1, content_col2 = st.columns(2)
                    with content_col1:
                        st.metric("New Content", trace.data.get('content_new', 0))
                    with content_col2:
                        st.metric("Reused Content", trace.data.get('content_reused', 0))
                    
                    # Display token usage
                    display_token_usage(trace)
                    
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

    # Sidebar additional info
    with st.sidebar:
        st.markdown("### ℹ️ Research Agent Info")
        st.write("Version: 1.0.0")
        st.write("Last Updated: January 2025")
        
        # Trace management
        st.markdown("### 🗂️ Trace Management")
        if st.button("Clear Research History"):
            try:
                if os.path.exists('research_traces.jsonl'):
                    os.remove('research_traces.jsonl')
                st.success("Research history cleared successfully!")
            except Exception as e:
                st.error(f"Error clearing history: {str(e)}")

# For database connection
db = get_db_connection()

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
    
    # Add custom log levels
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