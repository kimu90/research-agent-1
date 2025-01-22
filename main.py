import pdb
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
from tools import GeneralAgent
from research_agent.db.db import ContentDB
from research_agent.tracers import CustomTracer, QueryTrace
from tools.research.common.model_schemas import ContentItem, ResearchToolOutput
from prompts.prompt_manager import PromptManager
from utils.token_tracking import TokenUsageTracker
from utils.evaluation import create_factual_accuracy_evaluator
from utils.source_coverage import create_source_coverage_evaluator
from utils.logical_coherence import create_logical_coherence_evaluator
from utils.answer_relevance import create_answer_relevance_evaluator

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

# Database connection management
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
def display_token_usage(trace: QueryTrace, show_visualizations: bool = True):
    """Display token usage for a single trace"""
    token_stats = trace.token_tracker.get_usage_stats()
    
    # Extract token counts
    token_counts = token_stats.get('tokens', {})
    prompt_tokens = token_counts.get('input', 0)
    completion_tokens = token_counts.get('output', 0)
    total_tokens = token_counts.get('total', 0)
    
    st.subheader("Token Usage Summary")
    
    cols = st.columns(3)
    with cols[0]:
        st.metric("Total Tokens", f"{total_tokens:,}")
    with cols[1]:
        st.metric("Prompt Tokens", f"{prompt_tokens:,}")
    with cols[2]:
        st.metric("Completion Tokens", f"{completion_tokens:,}")
    
    if show_visualizations:
        # Processing metrics
        st.subheader("Processing Metrics")
        proc_cols = st.columns(2)
        with proc_cols[0]:
            st.metric(
                "Processing Time (s)",
                f"{token_stats.get('processing', {}).get('time', 0):.2f}"
            )
        with proc_cols[1]:
            st.metric(
                "Token Speed (tokens/s)",
                f"{token_stats.get('processing', {}).get('speed', 0):.2f}"
            )
        
        # Cost analysis
        st.subheader("Cost Analysis")
        st.metric(
            "Total Cost ($)",
            f"{token_stats.get('cost', 0):.6f}"
        )

def display_research_results(result, selected_tool):
    """Display research results with analytics tab"""
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
            # Pass the global db connection
            display_analytics(traces, db)

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
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Group by timestamp
            usage_over_time = (
                df.groupby([pd.Grouper(key='timestamp', freq='D'), 'prompt_id'])
                .size()
                .reset_index(name='count')
            )

            # Timeline chart
            fig_timeline = px.line(
                usage_over_time, 
                x='timestamp', 
                y='count', 
                color='prompt_id', 
                title='Prompt Usage Over Time'
            )
            st.plotly_chart(fig_timeline)
            
            # Most used prompts
            st.subheader("Most Used Prompts")
            prompt_counts = df['prompt_id'].value_counts()
            fig_prompts = px.bar(
                prompt_counts,
                x=prompt_counts.index,
                y=prompt_counts.values,
                title='Most Used Prompts'
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

def load_research_history() -> List[QueryTrace]:
    """Load and process research history into QueryTrace objects"""
    try:
        if not os.path.exists('research_traces.jsonl'):
            return []
            
        traces = []
        with open('research_traces.jsonl', 'r') as f:
            for line in f:
                trace_data = json.loads(line)
                trace = QueryTrace(trace_data['query'])
                trace.data = trace_data
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
    """Enhanced visualization of research traces"""
    if not traces:
        st.info("No research history available yet. Run some searches to see detailed analytics!")
        return

    st.subheader("🔄 Processing Steps Analysis")
    
    # Collect and analyze processing steps
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
        
        steps_df = pd.DataFrame(
            list(step_counts.items()),
            columns=['Processing Step', 'Frequency']
        ).sort_values('Frequency', ascending=False)
        
        fig_steps = px.bar(
            steps_df,
            x='Processing Step',
            y='Frequency',
            title="Frequency of Processing Steps"
        )
        fig_steps.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_steps, use_container_width=True)
    else:
        st.info("No processing steps recorded yet.")

    # Detailed Processing Step Breakdown
    st.subheader("📋 Detailed Processing Steps")
    
    unique_steps = list(set(all_steps))
    for step in unique_steps[:10]:
        with st.expander(f"Step: {step}"):
            related_traces = [
                trace for trace in traces
                if step in trace.data.get('processing_steps', [])
            ]
            
            if related_traces:
                st.write(f"Traces involving this step: {len(related_traces)}")
                
                step_traces_df = pd.DataFrame([
                    {
                        'Query': trace.data.get('query', 'N/A'),
                        'Tool': trace.data.get('tool', 'N/A'),
                        'Timestamp': trace.data.get('start_time', 'N/A'),
                        'Success': '✅' if trace.data.get('success', False) else '❌',
                        'Duration': f"{trace.data.get('duration', 0):.2f}s"
                    } for trace in related_traces
                ])
                
                st.dataframe(step_traces_df, use_container_width=True, hide_index=True)
            else:
                st.info("No detailed information available for this step.")

def display_analytics(traces: List[QueryTrace], content_db: ContentDB):
    """
    Display research analytics dashboard with properly organized sections
    """
    if not traces:
        st.info("No research history available yet. Run some searches to see analytics!")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "General Analytics", 
        "Token Usage", 
        "Processing Steps",
        "Content Validation"
    ])
   
    with tab1:
        # Only show success rate and research statistics
        df = pd.DataFrame([
            {
                'date': datetime.fromisoformat(t.data['start_time']).date(),
                'success': t.data.get('success', False),
                'duration': t.data.get('duration', 0),
                'content_new': t.data.get('content_new', 0), 
                'content_reused': t.data.get('content_reused', 0)
            }
            for t in traces
        ])
       
        # Success Rate Over Time
        st.subheader("📈 Success Rate Over Time")
        success_by_date = df.groupby('date').agg({
            'success': ['count', lambda x: x.sum() / len(x) * 100]
        }).reset_index()
        success_by_date.columns = ['date', 'total', 'success_rate']
        
        fig_success = px.line(
            success_by_date,
            x='date',
            y='success_rate',
            title='Success Rate Trend'
        )
        st.plotly_chart(fig_success, use_container_width=True)

        # Basic Statistics
        st.subheader("📊 Research Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Researches", len(traces))
        with col2:
            st.metric("Average Duration", f"{df['duration'].mean():.2f}s")
        with col3:
            success_rate = (df['success'].sum() / len(df)) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        with col4:
            total_content = df['content_new'].sum() + df['content_reused'].sum()
            st.metric("Total Content Processed", total_content)

    with tab2:
        # Token Usage tab - only show essential metrics here
        st.subheader("Token Usage Summary")
        
        # Main metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tokens", "1,589")
        with col2:
            st.metric("Prompt Tokens", "648")
        with col3:
            st.metric("Completion Tokens", "941")
            
        # Processing metrics
        st.subheader("Processing Metrics")
        proc_col1, proc_col2 = st.columns(2)
        with proc_col1:
            st.metric("Processing Time (s)", "0.50")
        with proc_col2:
            st.metric("Token Speed (tokens/s)", "3,178.00")
            
        # Cost analysis
        st.subheader("Cost Analysis")
        st.metric("Total Cost ($)", "0.001112")

    with tab3:
        enhance_trace_visualization(traces)

    with tab4:
        st.subheader("🔍 Content Validation Analysis")
        
        validation_tab1, validation_tab2, validation_tab3, validation_tab4 = st.tabs([
            "Factual Accuracy",
            "Source Coverage",
            "Logical Coherence",
            "Answer Relevance"
        ])

        with validation_tab1:
            try:
                accuracy_evals = content_db.get_accuracy_evaluations(limit=50)
                if accuracy_evals:
                    accuracy_df = pd.DataFrame(accuracy_evals)
                    
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    with metrics_col1:
                        avg_factual_score = accuracy_df['factual_score'].mean()
                        st.metric("Avg Factual Score", f"{avg_factual_score:.2f}")
                    with metrics_col2:
                        avg_citation_accuracy = accuracy_df['citation_accuracy'].mean()
                        st.metric("Avg Citation Accuracy", f"{avg_citation_accuracy:.2%}")
                    with metrics_col3:
                        total_evaluated_sources = accuracy_df['total_sources'].sum()
                        st.metric("Total Sources Evaluated", f"{total_evaluated_sources:,}")
                    
                    fig_score_dist = px.histogram(
                        accuracy_df,
                        x='factual_score',
                        title='Distribution of Factual Scores',
                        nbins=20
                    )
                    st.plotly_chart(fig_score_dist, use_container_width=True)
                else:
                    st.info("No factual accuracy data available yet.")
            except Exception as e:
                st.error(f"Error displaying factual accuracy analytics: {str(e)}")

        with validation_tab2:
            try:
                source_coverage_evals = content_db.get_source_coverage_evaluations(limit=50)
                if source_coverage_evals:
                    coverage_df = pd.DataFrame(source_coverage_evals)
                    
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    with metrics_col1:
                        st.metric("Avg Coverage Score", f"{coverage_df['coverage_score'].mean():.2f}")
                    with metrics_col2:
                        st.metric("Source Diversity", f"{coverage_df['diversity_score'].mean():.2f}")
                    with metrics_col3:
                        st.metric("Total Sources", f"{coverage_df['total_sources'].sum():,}")
                    
                    fig_coverage_dist = px.histogram(
                        coverage_df,
                        x='coverage_score',
                        title='Source Coverage Score Distribution'
                    )
                    st.plotly_chart(fig_coverage_dist, use_container_width=True)
                else:
                    st.info("No source coverage data available yet.")
            except Exception as e:
                st.error(f"Error displaying source coverage analytics: {str(e)}")

        with validation_tab3:
            try:
                coherence_evals = content_db.get_logical_coherence_evaluations(limit=50)
                if coherence_evals:
                    coherence_df = pd.DataFrame(coherence_evals)
                    
                    metrics_col1, metrics_col2 = st.columns(2)
                    with metrics_col1:
                        st.metric("Avg Coherence Score", f"{coherence_df['coherence_score'].mean():.2f}")
                    with metrics_col2:
                        argument_structure_ratio = (coherence_df['has_argument_structure'].sum() / len(coherence_df)) * 100
                        st.metric("Clear Arguments", f"{argument_structure_ratio:.1f}%")
                    
                    fig_coherence_dist = px.histogram(
                        coherence_df,
                        x='coherence_score',
                        title='Logical Coherence Score Distribution'
                    )
                    st.plotly_chart(fig_coherence_dist, use_container_width=True)
                else:
                    st.info("No logical coherence data available yet.")
            except Exception as e:
                st.error(f"Error displaying logical coherence analytics: {str(e)}")

        with validation_tab4:
            try:
                relevance_evals = content_db.get_answer_relevance_evaluations(limit=50)
                if relevance_evals:
                    relevance_df = pd.DataFrame(relevance_evals)
                    
                    metrics_col1, metrics_col2 = st.columns(2)
                    with metrics_col1:
                        st.metric("Avg Relevance Score", f"{relevance_df['relevance_score'].mean():.2f}")
                    with metrics_col2:
                        st.metric("Semantic Similarity", f"{relevance_df['semantic_similarity'].mean():.2f}")
                    
                    fig_relevance_dist = px.histogram(
                        relevance_df,
                        x='relevance_score',
                        title='Answer Relevance Score Distribution'
                    )
                    st.plotly_chart(fig_relevance_dist, use_container_width=True)
                else:
                    st.info("No answer relevance data available yet.")
            except Exception as e:
                st.error(f"Error displaying answer relevance analytics: {str(e)}")
def run_tool(tool_name: str, query: str, tool=None):
    """
    Run a specific research tool and track its execution and token usage
    
    Args:
        tool_name (str): Name of the research tool to use
        query (str): Research query
        tool (Optional): Preinitialized tool instance
    
    Returns:
        Tuple[Optional[ResearchToolOutput], QueryTrace]: Research result and trace
    """
    context = StreamlitContext()
    start_time = datetime.now()
    
    # Detailed logging configuration
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
    trace.data["processing_steps"].append(f"Started research with {tool_name}")
    
    try:
        # Initialize evaluation tools
        try:
            accuracy_evaluator = create_factual_accuracy_evaluator()
            source_coverage_evaluator = create_source_coverage_evaluator()
            coherence_evaluator = create_logical_coherence_evaluator()
            relevance_evaluator = create_answer_relevance_evaluator()
        except Exception as eval_init_error:
            logger.error(f"Evaluation tools initialization failed: {eval_init_error}")
            accuracy_evaluator = source_coverage_evaluator = coherence_evaluator = relevance_evaluator = None

        # Tool initialization and research
        if tool_name == "General Agent":
            # Create or use existing tool
            if tool is None:
                logger.info("Creating GeneralAgent instance")
                tool = GeneralAgent(include_summary=True)
            
            logger.info("Recording prompt usage")
            trace.add_prompt_usage("general_agent_search", "general", "")
            
            logger.info("Invoking GeneralAgent with query")
            result = tool.invoke(input={"query": query})
            logger.info("GeneralAgent invocation completed")
            
            # Content processing steps
            if result and result.content:
                try:
                    # Get database connection
                    db = get_db_connection()
                    
                    content_count = len(result.content)
                    logger.info(f"Processing {content_count} content items")
                    trace.data["processing_steps"].append(f"Preparing to process {content_count} content items")
                    
                    # Track new vs. reused content
                    new_content = 0
                    reused_content = 0
                    
                    for idx, item in enumerate(result.content, 1):
                        try:
                            logger.info(f"Processing content item {idx}/{content_count}")
                            if db:
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
                except Exception as content_processing_error:
                    logger.error(f"Content processing failed: {content_processing_error}")
                    trace.data["processing_steps"].append(f"Content processing error: {content_processing_error}")
            
            # Comprehensive evaluation
            if result and accuracy_evaluator and source_coverage_evaluator and coherence_evaluator and relevance_evaluator:
                try:
                    # Get database connection
                    db = get_db_connection()
                    
                    # Factual Accuracy Evaluation
                    factual_score, accuracy_details = accuracy_evaluator.evaluate_factual_accuracy(result)
                    trace.data['factual_accuracy'] = {
                        'score': factual_score,
                        'details': accuracy_details
                    }
                    if db:
                        db.store_accuracy_evaluation({
                            'query': query,
                            'timestamp': datetime.now().isoformat(),
                            'factual_score': factual_score,
                            **accuracy_details
                        })

                    # Source Coverage Evaluation
                    coverage_score, coverage_details = source_coverage_evaluator.evaluate_source_coverage(result)
                    trace.data['source_coverage'] = {
                        'score': coverage_score,
                        'details': coverage_details
                    }
                    
                    if db:
                        db.store_source_coverage({
                            'query': query,
                            'timestamp': datetime.now().isoformat(),
                            'coverage_score': coverage_score,
                            **coverage_details
                        })
                    
                    # Logical Coherence Evaluation
                    coherence_score, coherence_details = coherence_evaluator.evaluate_logical_coherence(result)
                    trace.data['logical_coherence'] = {
                        'score': coherence_score,
                        'details': coherence_details
                    }
                    
                    if db:
                        db.store_logical_coherence({
                            'query': query,
                            'timestamp': datetime.now().isoformat(),
                            'coherence_score': coherence_score,
                            **coherence_details
                        })
                    
                    # Answer Relevance Evaluation
                    relevance_score, relevance_details = relevance_evaluator.evaluate_answer_relevance(result, query)
                    trace.data['answer_relevance'] = {
                        'score': relevance_score,
                        'details': relevance_details
                    }
                    
                    if db:
                        db.store_answer_relevance({
                            'query': query,
                            'timestamp': datetime.now().isoformat(),
                            'relevance_score': relevance_score,
                            **relevance_details
                        })
                    
                except Exception as eval_error:
                    logger.error(f"Evaluation process failed: {eval_error}")
                    trace.data['evaluation_error'] = str(eval_error)
            
            # Update trace with success information
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Processing completed in {duration:.2f} seconds")
            
            trace.data["duration"] = duration
            trace.data["success"] = True
            trace.data["content_count"] = len(result.content) if result and result.content else 0
            
            # Log final token usage
            try:
                token_stats = trace.token_tracker.get_usage_stats()
                logger.info(f"Final token usage stats: {token_stats}")
                
                if token_stats['tokens']['total'] > 0:
                    usage_msg = f"Total tokens used: {token_stats['tokens']['total']}"
                    logger.info(usage_msg)
                    trace.data["processing_steps"].append(usage_msg)
            except Exception as token_error:
                logger.warning(f"Could not retrieve token stats: {token_error}")
            
            # Final success step
            logger.info("Research completed successfully")
            trace.data["processing_steps"].append("Research completed successfully")
            trace.data["end_time"] = datetime.now().isoformat()
            
            # Save the trace
            try:
                logger.info("Saving successful trace")
                tracer = CustomTracer()
                tracer.save_trace(trace)
            except Exception as trace_save_error:
                logger.error(f"Failed to save trace: {trace_save_error}")
            
            return result, trace
        
        else:
            # Unsupported tool
            error_msg = f"Tool {tool_name} not found"
            logger.error(error_msg)
            st.error(error_msg)
            trace.data["processing_steps"].append(f"Error: {error_msg}")
            trace.data['error'] = error_msg
            trace.data['success'] = False
            
            return None, trace
    
    except Exception as e:
        # Comprehensive error handling
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
        try:
            logger.info("Saving error trace")
            tracer = CustomTracer()
            tracer.save_trace(trace)
        except Exception as trace_save_error:
            logger.error(f"Failed to save error trace: {trace_save_error}")
        
        return None, trace
def main():
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
        with st.spinner(f"Researching with {selected_tool}..."):
            try:
                # Get selected prompt
                current_prompt = prompt_manager.get_prompt(selected_prompt_id)
                
                # Initialize tool with selected prompt
                tool = GeneralAgent(
                    include_summary=True,
                    custom_prompt=current_prompt
                )

                # Run the research
                result, trace = run_tool(
                    tool_name=selected_tool, 
                    query=query, 
                    tool=tool
                )
                
                if result:
                    # Display research results
                    display_research_results(result, selected_tool)
                    
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

# Initialize database connection
db = get_db_connection()

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