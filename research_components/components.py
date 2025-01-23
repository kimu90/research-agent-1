import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from typing import List

from research_agent.db.db import ContentDB
from research_agent.tracers import QueryTrace
from tools import GeneralAgent
from prompts.prompt_manager import PromptManager
from .database import get_db_connection
from .utils import load_research_history



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