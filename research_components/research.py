import logging
import streamlit as st
from datetime import datetime

from tools import GeneralAgent
from research_agent.tracers import CustomTracer, QueryTrace
from research_components.database import get_db_connection
from utils.evaluation import create_factual_accuracy_evaluator
from utils.source_coverage import create_source_coverage_evaluator
from utils.logical_coherence import create_logical_coherence_evaluator
from utils.answer_relevance import create_answer_relevance_evaluator


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
    context = st.session_state
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
