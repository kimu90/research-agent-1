import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from typing import List

from .db import ContentDB
from research_agent.tracers import QueryTrace
from tools import GeneralAgent
from prompt.prompt_manager import PromptManager
from .utils import load_research_history
import logging
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime
from typing import List
import time
from typing import Dict, Any

# Set up logger
logger = logging.getLogger(__name__)


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

db_path = "./data/content.db"  # Set your database path
content_db = ContentDB(db_path)


def display_analytics(traces: List[QueryTrace], content_db: ContentDB):
    """
    Display detailed analytics including token usage, processing steps, and content validation.
    """
    if not traces:
        st.info("No research history available yet. Run some searches to see analytics!")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Factual Accuracy",
        "Source Coverage",
        "Logical Coherence",
        "Answer Relevance",
        "Automated Tests"
    ])

    with tab1:
        try:
            accuracy_evals = content_db.get_accuracy_evaluations(limit=50)
            if accuracy_evals:
                accuracy_df = pd.DataFrame(accuracy_evals)
                
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                with metrics_col1:
                    avg_factual_score = accuracy_df['factual_score'].mean()
                    st.metric("Avg Factual Score", f"{avg_factual_score:.2f}")
                with metrics_col2:
                    avg_citation_accuracy = accuracy_df['citation_accuracy'].mean()
                    st.metric("Avg Citation Accuracy", f"{avg_citation_accuracy:.2%}")
                with metrics_col3:
                    verified_claims = accuracy_df['verified_claims'].sum()
                    st.metric("Verified Claims", f"{verified_claims:,}")
                with metrics_col4:
                    contradicting_claims = accuracy_df['contradicting_claims'].sum()
                    st.metric("Contradicting Claims", f"{contradicting_claims:,}")
                
                fig_score_dist = px.histogram(
                    accuracy_df,
                    x='factual_score',
                    title='Distribution of Factual Scores',
                    nbins=20
                )
                st.plotly_chart(fig_score_dist, use_container_width=True)
                
                fig_timeline = px.line(
                    accuracy_df.sort_values('timestamp'),
                    x='timestamp',
                    y='factual_score',
                    title='Factual Accuracy Over Time'
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
            else:
                st.info("No factual accuracy data available yet.")
        except Exception as e:
            st.error(f"Error displaying factual accuracy analytics: {str(e)}")
    with tab2:
        try:
            coverage_evals = content_db.get_source_coverage_evaluations(limit=50)
            if coverage_evals:
                coverage_df = pd.DataFrame(coverage_evals)
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                with metrics_col1:
                    avg_coverage = coverage_df['coverage_score'].mean()
                    st.metric("Avg Coverage Score", f"{avg_coverage:.2f}")
                with metrics_col2:
                    avg_diversity = coverage_df['diversity_score'].mean()
                    st.metric("Avg Source Diversity", f"{avg_diversity:.2f}")
                with metrics_col3:
                    unique_domains = coverage_df['unique_domains'].sum()
                    st.metric("Unique Domains", f"{unique_domains:,}")
                with metrics_col4:
                    avg_source_depth = coverage_df['source_depth'].mean()
                    st.metric("Avg Source Depth", f"{avg_source_depth:.2f}")
                
                fig = px.histogram(
                    coverage_df,
                    x='coverage_score',
                    title='Coverage Score Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                fig_timeline = px.line(
                    coverage_df.sort_values('timestamp'),
                    x='timestamp',
                    y='coverage_score',
                    title='Source Coverage Over Time'
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
            else:
                st.info("No source coverage data available yet.")
        except Exception as e:
            st.error(f"Error displaying coverage analytics: {str(e)}")

    with tab3:
        logger.info("Processing Logical Coherence tab")
        try:
            coherence_evals = content_db.get_logical_coherence_evaluations(limit=50)
            logger.debug(f"Retrieved {len(coherence_evals) if coherence_evals else 0} coherence evaluations")
            
            if coherence_evals:
                coherence_df = pd.DataFrame(coherence_evals)
                logger.debug(f"Created DataFrame with {len(coherence_df)} rows")
                logger.debug(f"Available columns: {coherence_df.columns.tolist()}")
                
                # Handle timestamp conversion
                if 'timestamp' in coherence_df.columns:
                    if coherence_df['timestamp'].dtype != 'datetime64[ns]':
                        logger.debug("Converting timestamps to datetime")
                        coherence_df['timestamp'] = pd.to_datetime(coherence_df['timestamp'])
                
                # Display metrics with safe column access
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                
                # Coherence Score
                with metrics_col1:
                    if 'coherence_score' in coherence_df.columns:
                        avg_coherence = coherence_df['coherence_score'].mean()
                        st.metric("Avg Coherence Score", f"{avg_coherence:.2f}")
                        logger.info(f"Average coherence score: {avg_coherence:.2f}")
                    else:
                        st.metric("Avg Coherence Score", "N/A")
                        logger.warning("coherence_score column not found")
                
                # Topic Coherence
                with metrics_col2:
                    score_column = next((col for col in coherence_df.columns if 'topic' in col.lower()), None)
                    if score_column:
                        avg_topic = coherence_df[score_column].mean()
                        st.metric("Topic Score", f"{avg_topic:.2f}")
                        logger.info(f"Average {score_column}: {avg_topic:.2f}")
                    else:
                        st.metric("Topic Score", "N/A")
                        logger.warning("No topic-related column found")
                
                # Logical Fallacies
                with metrics_col3:
                    if 'logical_fallacies_count' in coherence_df.columns:
                        fallacies = coherence_df['logical_fallacies_count'].sum()
                        st.metric("Logical Fallacies", f"{fallacies:,}")
                        logger.info(f"Total logical fallacies: {fallacies}")
                    else:
                        st.metric("Logical Fallacies", "N/A")
                        logger.warning("logical_fallacies_count column not found")
                
                # Idea Progression
                with metrics_col4:
                    if 'idea_progression_score' in coherence_df.columns:
                        progression = coherence_df['idea_progression_score'].mean()
                        st.metric("Idea Progression", f"{progression:.2f}")
                        logger.info(f"Average idea progression: {progression:.2f}")
                    else:
                        st.metric("Idea Progression", "N/A")
                        logger.warning("idea_progression_score column not found")

                # Create visualizations if coherence_score exists
                if 'coherence_score' in coherence_df.columns:
                    try:
                        logger.debug("Creating coherence score histogram")
                        fig_hist = px.histogram(
                            coherence_df,
                            x='coherence_score',
                            title='Coherence Score Distribution',
                            nbins=20,
                            labels={'coherence_score': 'Coherence Score', 'count': 'Count'}
                        )
                        fig_hist.update_layout(
                            xaxis_title="Coherence Score",
                            yaxis_title="Count"
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                        
                        if 'timestamp' in coherence_df.columns:
                            logger.debug("Creating coherence timeline")
                            fig_timeline = px.line(
                                coherence_df.sort_values('timestamp'),
                                x='timestamp',
                                y='coherence_score',
                                title='Logical Coherence Over Time'
                            )
                            fig_timeline.update_layout(
                                xaxis_title="Time",
                                yaxis_title="Coherence Score",
                                hovermode='x unified'
                            )
                            st.plotly_chart(fig_timeline, use_container_width=True)
                    except Exception as viz_error:
                        logger.error(f"Error creating visualizations: {str(viz_error)}", exc_info=True)
                        st.error("Unable to display coherence visualizations")

                # Display detailed statistics
                try:
                    logger.debug("Creating detailed statistics view")
                    with st.expander("View Detailed Statistics"):
                        # Get available columns from the required set
                        available_columns = ['timestamp', 'query']
                        score_columns = [col for col in coherence_df.columns if 'score' in col.lower()]
                        available_columns.extend(score_columns)
                        
                        detailed_df = coherence_df[available_columns].sort_values('timestamp', ascending=False)
                        
                        format_dict = {col: '{:.2f}' for col in score_columns}
                        st.dataframe(detailed_df.style.format(format_dict))
                        logger.debug(f"Displayed detailed statistics for {len(detailed_df)} entries")
                except Exception as stats_error:
                    logger.error(f"Error displaying detailed statistics: {str(stats_error)}", exc_info=True)
                    st.error("Unable to display detailed statistics")

            else:
                logger.info("No coherence data available")
                st.info("No logical coherence data available yet.")

        except Exception as e:
            error_msg = f"Error processing coherence analytics: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.error(error_msg)

    with tab4:
        try:
            relevance_evals = content_db.get_answer_relevance_evaluations(limit=50)
            if relevance_evals:
                relevance_df = pd.DataFrame(relevance_evals)
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                with metrics_col1:
                    avg_relevance = relevance_df['relevance_score'].mean()
                    st.metric("Avg Relevance Score", f"{avg_relevance:.2f}")
                with metrics_col2:
                    semantic_similarity = relevance_df['semantic_similarity'].mean()
                    st.metric("Semantic Similarity", f"{semantic_similarity:.2f}")
                with metrics_col3:
                    info_density = relevance_df['information_density'].mean()
                    st.metric("Info Density", f"{info_density:.2f}")
                with metrics_col4:
                    context_alignment = relevance_df['context_alignment_score'].mean()
                    st.metric("Context Alignment", f"{context_alignment:.2f}")
                
                fig = px.histogram(
                    relevance_df,
                    x='relevance_score',
                    title='Relevance Score Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                fig_timeline = px.line(
                    relevance_df.sort_values('timestamp'),
                    x='timestamp',
                    y='relevance_score',
                    title='Answer Relevance Over Time'
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
            else:
                st.info("No answer relevance data available yet.")
        except Exception as e:
            st.error(f"Error displaying relevance analytics: {str(e)}")
                

    with tab5:
        try:
            automated_test_results = content_db.get_test_results(limit=50)
            if automated_test_results:
                test_df = pd.DataFrame(automated_test_results)
                
                test_metrics_col1, test_metrics_col2, test_metrics_col3, test_metrics_col4 = st.columns(4)
                with test_metrics_col1:
                    avg_rouge1 = test_df['rouge1_score'].mean()
                    st.metric("Avg ROUGE-1", f"{avg_rouge1:.3f}")
                with test_metrics_col2:
                    avg_rouge2 = test_df['rouge2_score'].mean()
                    st.metric("Avg ROUGE-2", f"{avg_rouge2:.3f}")
                with test_metrics_col3:
                    avg_semantic = test_df['semantic_similarity'].mean()
                    st.metric("Semantic Similarity", f"{avg_semantic:.3f}")
                with test_metrics_col4:
                    avg_hallucination = test_df['hallucination_score'].mean()
                    st.metric("Hallucination Score", f"{avg_hallucination:.3f}")
                
                fig_scores = px.box(
                    test_df,
                    y=['rouge1_score', 'rouge2_score', 'semantic_similarity', 'hallucination_score'],
                    title="Test Score Distributions"
                )
                st.plotly_chart(fig_scores, use_container_width=True)
                
                fig_trend = px.line(
                    test_df.sort_values('timestamp'),
                    x='timestamp',
                    y='overall_score',
                    title="Overall Test Score Trend"
                )
                st.plotly_chart(fig_trend, use_container_width=True)
                
                if any(len(segs) > 0 for segs in test_df['suspicious_segments']):
                    st.subheader("Recent Suspicious Segments")
                    for idx, row in test_df.head(5).iterrows():
                        if row['suspicious_segments']:
                            st.markdown(f"**Query:** {row['query']}")
                            for segment in row['suspicious_segments']:
                                st.warning(f"• {segment}")
                            st.markdown("---")
            else:
                st.info("No automated test results available yet.")
        except Exception as e:
            st.error(f"Error displaying automated test results: {str(e)}")

        

def display_general_analysis(traces: List[QueryTrace]):
    """
    Display general analytics metrics including success rate and basic statistics.
    """
    if not traces:
        st.info("No research history available yet. Run some searches to see analytics!")
        return
        
    df = pd.DataFrame([{
        'date': datetime.fromisoformat(t.data['start_time']).date(),
        'success': t.data.get('success', False),
        'duration': t.data.get('duration', 0),
        'content_new': t.data.get('content_new', 0), 
        'content_reused': t.data.get('content_reused', 0)
    } for t in traces])
   
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

def safe_json_loads(value):
    if isinstance(value, (str, bytes, bytearray)):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return {}

def display_analysis(traces: List[QueryTrace], content_db: ContentDB):
    logger.info("Starting display_analysis")
    if not traces:
        st.info("No analysis history available.")
        return

    try:
        # Get and preprocess evaluations
        analysis_evals = content_db.get_analysis_evaluations(limit=50)
        
        if analysis_evals:
            for eval_data in analysis_evals:
                eval_data['analytical_elements'] = safe_json_loads(eval_data.get('analytical_elements'))
                eval_data['validation_checks'] = safe_json_loads(eval_data.get('validation_checks'))
                eval_data['calculation_examples'] = safe_json_loads(eval_data.get('calculation_examples'))
                # Convert numeric fields
                for field in ['numerical_accuracy', 'query_understanding', 'data_validation', 
                            'reasoning_transparency', 'overall_score', 'term_coverage']:
                    eval_data[field] = float(eval_data.get(field, 0))

        tab1, tab2, tab3, tab4 = st.tabs([
            "Analysis Overview", "Numerical Accuracy", 
            "Query Understanding", "Validation & Reasoning"
        ])

        with tab1:
            try:
                df = pd.DataFrame([{
                    'date': datetime.fromisoformat(t.data['start_time']).date(),
                    'success': bool(t.data.get('success', False)),
                    'duration': float(t.data.get('duration', 0)),
                    'overall_score': next((e['overall_score'] for e in analysis_evals 
                        if e['timestamp'] == t.data.get('start_time')), 0.0)
                } for t in traces if isinstance(t.data, dict) and 'start_time' in t.data])
                
                success_by_date = df.groupby('date').agg({
                    'success': ['count', lambda x: x.sum() / len(x) * 100],
                    'overall_score': 'mean'
                }).reset_index()
                success_by_date.columns = ['date', 'total_analyses', 'success_rate', 'avg_score']

                fig = px.line(success_by_date, x='date', y=['success_rate', 'avg_score'])
                st.plotly_chart(fig, use_container_width=True)

                cols = st.columns(4)
                cols[0].metric("Total Analyses", len(df))
                cols[1].metric("Average Duration", f"{df['duration'].mean():.2f}s")
                cols[2].metric("Success Rate", f"{df['success'].mean()*100:.1f}%")
                cols[3].metric("Avg Score", f"{df['overall_score'].mean():.2f}")

            except Exception as e:
                logger.error(f"Error in overview: {str(e)}")
                st.error("Error processing overview")

        with tab2:
            try:
                if analysis_evals:
                    metrics_df = pd.DataFrame([{
                        'timestamp': e['timestamp'],
                        'numerical_accuracy': e['numerical_accuracy'],
                        'term_coverage': e['term_coverage']
                    } for e in analysis_evals])

                    cols = st.columns(2)
                    cols[0].metric("Numerical Accuracy", f"{metrics_df['numerical_accuracy'].mean():.2f}")
                    cols[1].metric("Term Coverage", f"{metrics_df['term_coverage'].mean():.2f}")

                    fig = px.line(metrics_df.sort_values('timestamp'),
                                x='timestamp', y='numerical_accuracy')
                    st.plotly_chart(fig, use_container_width=True)

                    for eval_data in analysis_evals[:5]:
                        examples = eval_data['calculation_examples']
                        if examples:
                            with st.expander(f"Query: {eval_data['query']}"):
                                for ex in examples:
                                    st.code(ex)

            except Exception as e:
                logger.error(f"Error in accuracy: {str(e)}")
                st.error("Error processing accuracy data")

        with tab3:
            try:
                if analysis_evals:
                    df = pd.DataFrame([{
                        'timestamp': e['timestamp'],
                        'query_understanding': e['query_understanding'],
                        'analytical_elements': len(e['analytical_elements'])
                    } for e in analysis_evals])

                    cols = st.columns(2)
                    cols[0].metric("Query Understanding", f"{df['query_understanding'].mean():.2f}")
                    cols[1].metric("Analytical Elements", f"{df['analytical_elements'].mean():.1f}")

                    fig = px.line(df.sort_values('timestamp'),
                                x='timestamp', y='query_understanding')
                    st.plotly_chart(fig, use_container_width=True)

                    for eval_data in analysis_evals[:5]:
                        elements = eval_data['analytical_elements']
                        if elements:
                            with st.expander(f"Query: {eval_data['query']}"):
                                for k, v in elements.items():
                                    st.markdown(f"**{k}:** {v}")

            except Exception as e:
                logger.error(f"Error in understanding: {str(e)}")
                st.error("Error processing understanding data")

        with tab4:
            try:
                if analysis_evals:
                    df = pd.DataFrame([{
                        'timestamp': e['timestamp'],
                        'data_validation': e['data_validation'],
                        'reasoning_transparency': e['reasoning_transparency']
                    } for e in analysis_evals])

                    cols = st.columns(2)
                    cols[0].metric("Data Validation", f"{df['data_validation'].mean():.2f}")
                    cols[1].metric("Reasoning", f"{df['reasoning_transparency'].mean():.2f}")

                    fig = px.box(df, y=['data_validation', 'reasoning_transparency'])
                    st.plotly_chart(fig, use_container_width=True)

                    for eval_data in analysis_evals[:5]:
                        checks = eval_data['validation_checks']
                        if checks:
                            with st.expander(f"Query: {eval_data['query']}"):
                                for check_type, details in checks.items():
                                    st.markdown(f"**{check_type}:** {details}")

            except Exception as e:
                logger.error(f"Error in validation: {str(e)}")
                st.error("Error processing validation data")

    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        st.error("Error displaying metrics")
def display_prompt_tracking(content_db: ContentDB):
    """
    Display comprehensive prompt tracking and usage analytics
    """
    st.subheader("🔄 Prompt Tracking & Usage Analytics")

    # Retrieve query traces
    query_traces = content_db.get_query_traces(limit=100)
    
    if not query_traces:
        st.info("No prompt usage data available yet.")
        return

    try:
        # Create DataFrame from query traces with error handling
        traces_df = pd.DataFrame(query_traces)
        
        # Ensure timestamp is converted correctly
        if 'timestamp' in traces_df.columns:
            traces_df['timestamp'] = pd.to_datetime(traces_df['timestamp'], errors='coerce')
        else:
            st.warning("No timestamp column found in query traces.")
            return

        # Remove any rows with invalid timestamps
        traces_df = traces_df.dropna(subset=['timestamp'])

        # Verify required columns exist
        required_columns = ['tool', 'success', 'duration']
        for col in required_columns:
            if col not in traces_df.columns:
                st.warning(f"Column '{col}' not found in query traces.")
                return

        # Top-level Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Queries", len(traces_df))
        with col2:
            success_rate = (traces_df['success'].sum() / len(traces_df)) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        with col3:
            avg_duration = traces_df['duration'].mean()
            st.metric("Avg Duration", f"{avg_duration:.2f}s")
        with col4:
            unique_tools = traces_df['tool'].nunique()
            st.metric("Unique Tools", unique_tools)

        # Visualization 1: Tool Usage Distribution
        st.subheader("Tool Usage Distribution")
        tool_usage = traces_df['tool'].value_counts()
        fig_tool_usage = px.bar(
            x=tool_usage.index, 
            y=tool_usage.values, 
            title="Prompt Usage by Tool",
            labels={'x': 'Tool', 'y': 'Number of Queries'}
        )
        st.plotly_chart(fig_tool_usage, use_container_width=True)

        # Visualization 2: Success Rate Over Time
        st.subheader("Success Rate Over Time")
        daily_success = traces_df.groupby(pd.Grouper(key='timestamp', freq='D'))['success'].mean()
        fig_success_trend = px.line(
            x=daily_success.index, 
            y=daily_success.values, 
            title="Daily Success Rate Trend",
            labels={'x': 'Date', 'y': 'Success Rate'}
        )
        st.plotly_chart(fig_success_trend, use_container_width=True)

        # Detailed Query Traces
        st.subheader("Detailed Query Traces")
        
        # Display the detailed dataframe
        st.dataframe(
            traces_df[['timestamp', 'tool', 'success', 'duration']].style.format({
                'timestamp': lambda x: x.strftime('%Y-%m-%d %H:%M'),
                'success': lambda x: '✅' if x else '❌',
                'duration': '{:.2f}s'.format
            }),
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Error processing prompt tracking data: {str(e)}")
        logging.error(f"Prompt tracking error: {str(e)}", exc_info=True)