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

def show_metric_with_tooltip(label, value, help_text):
    """
    Create a metric with an additional tooltip explaining its meaning.
    
    :param label: The label for the metric
    :param value: The value to display
    :param help_text: Explanatory text shown in the tooltip
    """
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.metric(label=label, value=value, help=help_text)




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
                    show_metric_with_tooltip(
                        "Avg Factual Score", 
                        f"{avg_factual_score:.2f}", 
                        "Average score measuring the factual accuracy of responses based on verified information sources"
                    )
                with metrics_col2:
                    avg_citation_accuracy = accuracy_df['citation_accuracy'].mean()
                    show_metric_with_tooltip(
                        "Avg Citation Accuracy", 
                        f"{avg_citation_accuracy:.2%}", 
                        "Percentage of citations that correctly reference their source materials and context"
                    )
                with metrics_col3:
                    verified_claims = accuracy_df['verified_claims'].sum()
                    show_metric_with_tooltip(
                        "Verified Claims", 
                        f"{verified_claims:,}", 
                        "Number of statements that have been independently verified against trusted sources"
                    )
                with metrics_col4:
                    contradicting_claims = accuracy_df['contradicting_claims'].sum()
                    show_metric_with_tooltip(
                        "Contradicting Claims", 
                        f"{contradicting_claims:,}", 
                        "Count of claims that conflict with other statements or established facts"
                    )
                
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
                    show_metric_with_tooltip(
                        "Avg Coverage Score", 
                        f"{avg_coverage:.2f}", 
                        "Overall score representing the breadth and depth of source coverage"
                    )
                with metrics_col2:
                    avg_diversity = coverage_df['diversity_score'].mean()
                    show_metric_with_tooltip(
                        "Avg Source Diversity", 
                        f"{avg_diversity:.2f}", 
                        "Measure of the variety and range of sources used in research"
                    )
                with metrics_col3:
                    unique_domains = coverage_df['unique_domains'].sum()
                    show_metric_with_tooltip(
                        "Unique Domains", 
                        f"{unique_domains:,}", 
                        "Total number of distinct domains referenced in research"
                    )
                with metrics_col4:
                    avg_source_depth = coverage_df['source_depth'].mean()
                    show_metric_with_tooltip(
                        "Avg Source Depth", 
                        f"{avg_source_depth:.2f}", 
                        "Average level of detail and comprehensiveness of sources"
                    )
                
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
                        show_metric_with_tooltip(
                            "Avg Coherence Score", 
                            f"{avg_coherence:.2f}", 
                            "Measure of logical consistency and smooth progression of ideas"
                        )
                        logger.info(f"Average coherence score: {avg_coherence:.2f}")
                    else:
                        st.metric("Avg Coherence Score", "N/A")
                        logger.warning("coherence_score column not found")
                
                # Topic Coherence
                with metrics_col2:
                    score_column = next((col for col in coherence_df.columns if 'topic' in col.lower()), None)
                    if score_column:
                        avg_topic = coherence_df[score_column].mean()
                        show_metric_with_tooltip(
                            "Topic Score", 
                            f"{avg_topic:.2f}", 
                            "Evaluation of how well the content maintains topical relevance"
                        )
                        logger.info(f"Average {score_column}: {avg_topic:.2f}")
                    else:
                        st.metric("Topic Score", "N/A")
                        logger.warning("No topic-related column found")
                
                # Logical Fallacies
                with metrics_col3:
                    if 'logical_fallacies_count' in coherence_df.columns:
                        fallacies = coherence_df['logical_fallacies_count'].sum()
                        show_metric_with_tooltip(
                            "Logical Fallacies", 
                            f"{fallacies:,}", 
                            "Total number of detected logical inconsistencies or fallacious reasoning"
                        )
                        logger.info(f"Total logical fallacies: {fallacies}")
                    else:
                        st.metric("Logical Fallacies", "N/A")
                        logger.warning("logical_fallacies_count column not found")
                
                # Idea Progression
                with metrics_col4:
                    if 'idea_progression_score' in coherence_df.columns:
                        progression = coherence_df['idea_progression_score'].mean()
                        show_metric_with_tooltip(
                            "Idea Progression", 
                            f"{progression:.2f}", 
                            "Score indicating the smoothness and logical flow of ideas"
                        )
                        logger.info(f"Average idea progression: {progression:.2f}")
                    else:
                        st.metric("Idea Progression", "N/A")
                        logger.warning("idea_progression_score column not found")

                # Rest of the Logical Coherence tab content remains the same
                # (keeping the previous visualization and detailed statistics code)

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
                    show_metric_with_tooltip(
                        "Avg Relevance Score", 
                        f"{avg_relevance:.2f}", 
                        "Overall measure of how pertinent and directly answering the response is"
                    )
                with metrics_col2:
                    semantic_similarity = relevance_df['semantic_similarity'].mean()
                    show_metric_with_tooltip(
                        "Semantic Similarity", 
                        f"{semantic_similarity:.2f}", 
                        "Degree of semantic closeness between query and response"
                    )
                with metrics_col3:
                    info_density = relevance_df['information_density'].mean()
                    show_metric_with_tooltip(
                        "Info Density", 
                        f"{info_density:.2f}", 
                        "Measure of information compactness and substantiveness"
                    )
                with metrics_col4:
                    context_alignment = relevance_df['context_alignment_score'].mean()
                    show_metric_with_tooltip(
                        "Context Alignment", 
                        f"{context_alignment:.2f}", 
                        "How well the response maintains the original context and intent"
                    )
                
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
                    show_metric_with_tooltip(
                        "Avg ROUGE-1", 
                        f"{avg_rouge1:.3f}", 
                        "Evaluation of unigram (single word) overlap between generated and reference text"
                    )
                with test_metrics_col2:
                    avg_rouge2 = test_df['rouge2_score'].mean()
                    show_metric_with_tooltip(
                        "Avg ROUGE-2", 
                        f"{avg_rouge2:.3f}", 
                        "Evaluation of bigram (two-word) overlap between generated and reference text"
                    )
                with test_metrics_col3:
                    avg_semantic = test_df['semantic_similarity'].mean()
                    show_metric_with_tooltip(
                        "Semantic Similarity", 
                        f"{avg_semantic:.3f}", 
                        "Measure of semantic closeness between generated and reference text"
                    )
                with test_metrics_col4:
                    avg_hallucination = test_df['hallucination_score'].mean()
                    show_metric_with_tooltip(
                        "Hallucination Score", 
                        f"{avg_hallucination:.3f}", 
                        "Indicator of generated content's factual accuracy and likelihood of generating fictional information"
                    )
                
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

    st.subheader("📊 Research Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_metric_with_tooltip(
            "Total Researches", 
            str(len(traces)), 
            "Total number of research tasks performed"
        )
    with col2:
        show_metric_with_tooltip(
            "Average Duration", 
            f"{df['duration'].mean():.2f}s", 
            "Average time taken to complete a research task"
        )
    with col3:
        success_rate = (df['success'].sum() / len(df)) * 100
        show_metric_with_tooltip(
            "Success Rate", 
            f"{success_rate:.1f}%", 
            "Percentage of research tasks completed successfully"
        )
    with col4:
        total_content = df['content_new'].sum() + df['content_reused'].sum()
        show_metric_with_tooltip(
            "Total Content Processed", 
            str(total_content), 
            "Total amount of new and reused content generated"
        )

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
                
                # Metrics displayed horizontally
                cols = st.columns(4)
                with cols[0]:
                    show_metric_with_tooltip(
                        "Total Analyses", 
                        str(len(df)), 
                        "Total number of analysis tasks performed"
                    )
                with cols[1]:
                    show_metric_with_tooltip(
                        "Average Duration", 
                        f"{df['duration'].mean():.2f}s", 
                        "Average time taken to complete an analysis task"
                    )
                with cols[2]:
                    show_metric_with_tooltip(
                        "Success Rate", 
                        f"{df['success'].mean()*100:.1f}%", 
                        "Percentage of analysis tasks completed successfully"
                    )
                with cols[3]:
                    show_metric_with_tooltip(
                        "Avg Score", 
                        f"{df['overall_score'].mean():.2f}", 
                        "Overall average performance score across all analyses"
                    )
                
                # Line graph 
                fig = px.line(success_by_date, x='date', y=['success_rate', 'avg_score'])
                st.plotly_chart(fig, use_container_width=True)
                
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
                    with cols[0]:
                        show_metric_with_tooltip(
                            "Numerical Accuracy", 
                            f"{metrics_df['numerical_accuracy'].mean():.2f}", 
                            "Precision of numerical calculations and data interpretation"
                        )
                    with cols[1]:
                        show_metric_with_tooltip(
                            "Term Coverage", 
                            f"{metrics_df['term_coverage'].mean():.2f}", 
                            "Breadth of terminology and concepts covered in the analysis"
                        )

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
                    with cols[0]:
                        show_metric_with_tooltip(
                            "Query Understanding", 
                            f"{df['query_understanding'].mean():.2f}", 
                            "Depth of comprehension and interpretation of user queries"
                        )
                    with cols[1]:
                        show_metric_with_tooltip(
                            "Analytical Elements", 
                            f"{df['analytical_elements'].mean():.1f}", 
                            "Average number of key analytical components identified in each query"
                        )

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
                    with cols[0]:
                        show_metric_with_tooltip(
                            "Data Validation", 
                            f"{df['data_validation'].mean():.2f}", 
                            "Thoroughness and accuracy of data verification processes"
                        )
                    with cols[1]:
                        show_metric_with_tooltip(
                            "Reasoning", 
                            f"{df['reasoning_transparency'].mean():.2f}", 
                            "Clarity and explainability of analytical reasoning"
                        )

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
    Display comprehensive prompt tracking and usage analytics with detailed evaluation metrics
    """
    st.subheader("🔍 Detailed Prompt Tracking & Performance Analytics")

    # Filtering options
    col1, col2 = st.columns(2)
    with col1:
        query_filter = st.text_input("Filter by Query", placeholder="Enter partial query to filter")
    with col2:
        tool_filter = st.selectbox("Filter by Tool", 
            ["All", "General Agent", "Analysis Agent"], 
            index=0)

    # Retrieve query traces with optional filtering
    query_traces = content_db.get_query_traces(query=query_filter if query_filter else None, limit=100)
    
    if not query_traces:
        st.info("No prompt usage data available.")
        return

    try:
        # Aggregate evaluation data
        accuracy_evals = content_db.get_accuracy_evaluations(query=query_filter)
        relevance_evals = content_db.get_answer_relevance_evaluations(query=query_filter)
        coherence_evals = content_db.get_logical_coherence_evaluations(query=query_filter)
        source_coverage_evals = content_db.get_source_coverage_evaluations(query=query_filter)

        # Convert to DataFrame for easier manipulation
        traces_df = pd.DataFrame(query_traces)

        # Top-level Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            show_metric_with_tooltip(
                "Total Queries",
                 str(len(traces_df)),
                 "Total number of queries processed across all tools"
            )
        with col2:
            success_rate = (traces_df['success'].sum() / len(traces_df)) * 100
            show_metric_with_tooltip(
                "Success Rate",
                 f"{success_rate:.1f}%",
                 "Percentage of queries that were successfully completed"
            )
        with col3:
            avg_duration = traces_df['duration'].mean()
            show_metric_with_tooltip(
                "Avg Duration",
                 f"{avg_duration:.2f}s",
                 "Average time taken to complete a query"
            )
        with col4:
            unique_tools = traces_df['tool'].nunique()
            show_metric_with_tooltip(
                "Unique Tools",
                 str(unique_tools),
                 "Number of different tools or methods used in queries"
            )

        # Combine evaluation data
        eval_data = {}
        for eval_type, evals in [
            ('accuracy', accuracy_evals),
            ('relevance', relevance_evals),
            ('coherence', coherence_evals),
            ('source_coverage', source_coverage_evals)
        ]:
            for eval_item in evals:
                query = eval_item['query']
                if query not in eval_data:
                    eval_data[query] = {}
                eval_data[query][eval_type] = eval_item

        # Create a comprehensive dataframe
        detailed_traces = []
        for trace in query_traces:
            query = trace['query']
            
            # Apply tool filter
            if tool_filter != "All" and trace['tool'] != tool_filter:
                continue

            # Combine trace data with evaluation metrics
            trace_detail = {
                'Timestamp': trace.get('timestamp', 'N/A'),
                'Query': query,
                'Tool': trace.get('tool', 'N/A'),
                'Success': '✅' if trace.get('success', False) else '❌',
                'Duration (s)': round(trace.get('duration', 0), 2),
                'Content New': trace.get('content_new', 0),
                'Factual Accuracy': 'N/A',
                'Relevance Score': 'N/A',
                'Coherence Score': 'N/A',
                'Source Coverage': 'N/A'
            }

            # Add evaluation metrics if available
            if query in eval_data:
                eval_metrics = eval_data[query]
                
                if 'accuracy' in eval_metrics:
                    trace_detail['Factual Accuracy'] = f"{eval_metrics['accuracy'].get('factual_score', 0):.2f}"
                
                if 'relevance' in eval_metrics:
                    trace_detail['Relevance Score'] = f"{eval_metrics['relevance'].get('relevance_score', 0):.2f}"
                
                if 'coherence' in eval_metrics:
                    trace_detail['Coherence Score'] = f"{eval_metrics['coherence'].get('coherence_score', 0):.2f}"
                
                if 'source_coverage' in eval_metrics:
                    trace_detail['Source Coverage'] = f"{eval_metrics['source_coverage'].get('coverage_score', 0):.2f}"

            detailed_traces.append(trace_detail)

        # Display the detailed dataframe
        if detailed_traces:
            df = pd.DataFrame(detailed_traces)
            st.dataframe(df, use_container_width=True)

            # Additional Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Evaluation Scores Distribution")
                eval_columns = ['Factual Accuracy', 'Relevance Score', 'Coherence Score', 'Source Coverage']
                fig_box = px.box(df, y=eval_columns, 
                    title="Distribution of Evaluation Metrics")
                st.plotly_chart(fig_box, use_container_width=True)
            
            with col2:
                st.subheader("Success Rate by Tool")
                tool_success = df.groupby('Tool')['Success'].apply(lambda x: (x == '✅').mean() * 100)
                fig_tool_success = px.bar(
                    x=tool_success.index, 
                    y=tool_success.values,
                    title="Success Rate Percentage by Tool",
                    labels={'x': 'Tool', 'y': 'Success Rate (%)'}
                )
                st.plotly_chart(fig_tool_success, use_container_width=True)

        else:
            st.warning("No matching prompt traces found.")

    except Exception as e:
        st.error(f"Error processing prompt tracking data: {str(e)}")
        logging.error(f"Prompt tracking error: {str(e)}", exc_info=True)