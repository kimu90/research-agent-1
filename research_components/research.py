import logging
logging.getLogger('watchdog.observers.inotify_buffer').setLevel(logging.WARNING)
from datetime import datetime
from tools import GeneralAgent, AnalysisAgent
from research_agent.tracers import CustomTracer, QueryTrace
from utils.source_coverage import create_source_coverage_evaluator
from utils.evaluation import create_factual_accuracy_evaluator
from utils.answer_relevance import create_answer_relevance_evaluator
from utils.logical_coherence import create_logical_coherence_evaluator
from utils.automated_tests import create_automated_test_evaluator
from utils.analysis_evaluator import create_analysis_evaluator
from .db import ContentDB
import json
from typing import Optional, Dict, Any, Union, List
from tools.research.common.model_schemas import ContentItem

def convert_content_items(value: Any) -> Any:
    """Convert ContentItem objects to their string representation."""
    if isinstance(value, ContentItem):
        return value.get_text_content()
    elif isinstance(value, list):
        return [convert_content_items(item) for item in value]
    elif isinstance(value, dict):
        return {k: convert_content_items(v) for k, v in value.items()}
    elif hasattr(value, 'get_text_content'):
        return value.get_text_content()
    return value

def safe_store(store_func, data: Dict[str, Any], logger: logging.Logger) -> Optional[int]:
    """Safely store data with proper error handling."""
    try:
        result = store_func(data)
        if result == -1:
            logger.error(f"Failed to store data using {store_func.__name__}")
            return None
        return result
    except Exception as e:
        logger.error(f"Error in {store_func.__name__}: {str(e)}", exc_info=True)
        return None

def run_tool(
    tool_name: str, 
    query: str, 
    dataset: str = None, 
    analysis_type: str = None, 
    tool=None,
    prompt_name: str = "research.txt"
):
    """
    Execute a research or analysis tool with comprehensive tracing and evaluation.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s - %(filename)s:%(lineno)d',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('research_tool.log', mode='a')
        ]
    )
    logger = logging.getLogger(__name__)
    
    start_time = datetime.now()
    db = ContentDB("./data/content.db")    
    logger.info(f"Starting tool execution - Tool: {tool_name}")
    logger.info(f"Query received: {query}")
    logger.info(f"Prompt selected: {prompt_name}")
    
    trace = QueryTrace(query)
    trace.data.update({
        "tool": tool_name,
        "tools_used": [tool_name],
        "processing_steps": [],
        "content_new": 0,
        "content_reused": 0,
        "prompt_used": prompt_name
    })

    try:
        # Initialize evaluators based on tool type
        try:
            if tool_name == "General Agent":
                accuracy_evaluator = create_factual_accuracy_evaluator()
                source_coverage_evaluator = create_source_coverage_evaluator()
                coherence_evaluator = create_logical_coherence_evaluator()
                relevance_evaluator = create_answer_relevance_evaluator()
                automated_test_evaluator = create_automated_test_evaluator()
                analysis_evaluator = None
            else:
                analysis_evaluator = create_analysis_evaluator()
                accuracy_evaluator = source_coverage_evaluator = coherence_evaluator = None
                relevance_evaluator = None
                automated_test_evaluator = None
                
        except Exception as eval_init_error:
            logger.error(f"Evaluation tools initialization failed: {eval_init_error}")
            accuracy_evaluator = source_coverage_evaluator = coherence_evaluator = relevance_evaluator = automated_test_evaluator = analysis_evaluator = None

        if tool_name == "General Agent":
            if tool is None:
                tool = GeneralAgent(
                    include_summary=True, 
                    prompt_name=prompt_name
                )
            
            trace.add_prompt_usage("general_agent_search", "general", prompt_name)
            result = tool.invoke(input={"query": query})
            
            if result:
                try:
                    content_count = len(result.content) if result.content else 0
                    trace.data["processing_steps"].append(f"Preparing to process {content_count} content items")
                    trace.data.update({
                        "content_new": content_count,
                        "content_reused": 0,
                        "processing_steps": [f"Content processed - New: {content_count}, Reused: 0"]
                    })
                except Exception as content_processing_error:
                    logger.error(f"Content processing failed: {content_processing_error}")
                    trace.data["processing_steps"].append(f"Content processing error: {content_processing_error}")
            
                # Run all evaluations for General Agent
                try:
                    if accuracy_evaluator:
                        factual_score, accuracy_details = accuracy_evaluator.evaluate_factual_accuracy(result)
                        accuracy_details = convert_content_items(accuracy_details)
                        trace.data['factual_accuracy'] = {
                            'score': factual_score,
                            'details': accuracy_details
                        }
                        trace.token_tracker.add_usage(100, 50, "llama3-70b-8192", "factual_accuracy")
                        safe_store(db.store_accuracy_evaluation, {
                            'query': query,
                            'timestamp': datetime.now().isoformat(),
                            'factual_score': factual_score,
                            **accuracy_details
                        }, logger)

                    if source_coverage_evaluator:
                        coverage_score, coverage_details = source_coverage_evaluator.evaluate_source_coverage(result)
                        coverage_details = convert_content_items(coverage_details)
                        trace.data['source_coverage'] = {
                            'score': coverage_score,
                            'details': coverage_details
                        }
                        trace.token_tracker.add_usage(100, 50, "llama3-70b-8192", "source_coverage")
                        safe_store(db.store_source_coverage, {
                            'query': query,
                            'coverage_score': coverage_score,
                            **coverage_details
                        }, logger)

                    if coherence_evaluator:
                        coherence_score, coherence_details = coherence_evaluator.evaluate_logical_coherence(result)
                        coherence_details = convert_content_items(coherence_details)
                        trace.data['logical_coherence'] = {
                            'score': coherence_score,
                            'details': coherence_details
                        }
                        trace.token_tracker.add_usage(100, 50, "llama3-70b-8192", "logical_coherence")
                        safe_store(db.store_logical_coherence, {
                            'query': query,
                            'coherence_score': coherence_score,
                            **coherence_details
                        }, logger)

                    if relevance_evaluator:
                        relevance_score, relevance_details = relevance_evaluator.evaluate_answer_relevance(result, query)
                        relevance_details = convert_content_items(relevance_details)
                        trace.data['answer_relevance'] = {
                            'score': relevance_score,
                            'details': relevance_details
                        }
                        trace.token_tracker.add_usage(100, 50, "llama3-70b-8192", "answer_relevance")
                        
                        safe_store(db.store_answer_relevance, {
                            'query': query,
                            'relevance_score': relevance_score,
                            'semantic_similarity': float(relevance_details.get('semantic_similarity', 0)),
                            'entity_coverage': float(relevance_details.get('entity_coverage', 0)),
                            'keyword_coverage': float(relevance_details.get('keyword_coverage', 0)),
                            'topic_focus': float(relevance_details.get('topic_focus', 0)),
                            'off_topic_sentences': relevance_details.get('off_topic_sentences', []),
                            'total_sentences': int(relevance_details.get('total_sentences', 0)),
                            'query_match_percentage': float(relevance_details.get('query_match_percentage', 0)),
                            'information_density': float(relevance_details.get('information_density', 0)),
                            'context_alignment_score': float(relevance_details.get('context_alignment_score', 0))
                        }, logger)

                    if automated_test_evaluator:
                        # Handle ContentItem objects in the content list
                        if isinstance(result.content, list):
                            content_texts = []
                            for item in result.content:
                                if isinstance(item, ContentItem):
                                    content_texts.append(item.get_text_content())
                                else:
                                    content_texts.append(str(item))
                            content_for_test = ' '.join(content_texts)
                        else:
                            content_for_test = str(result.content)

                        logger.debug(f"Prepared content for automated tests, length: {len(content_for_test)}")
                        test_score, test_details = automated_test_evaluator.evaluate_automated_tests(content_for_test, query)
                        test_details = convert_content_items(test_details)
                        trace.data['automated_tests'] = {
                            'score': test_score,
                            'details': test_details
                        }
                        trace.token_tracker.add_usage(100, 50, "llama3-70b-8192", "automated_tests")
                        safe_store(db.store_test_results, {
                            'query': query,
                            'test_score': test_score,
                            **test_details
                        }, logger)

                except Exception as eval_error:
                    logger.error(f"Research evaluation failed: {eval_error}", exc_info=True)
                    trace.data['evaluation_error'] = str(eval_error)

        elif tool_name == "Analysis Agent":
            if tool is None:
                tool = AnalysisAgent(
                    data_folder="./data",
                    prompt_name=prompt_name
                )
            
            trace.add_prompt_usage("analysis_agent", "analysis", "")
            result = tool.invoke_analysis(input={
                "query": query,
                "dataset": dataset
            })
            
            if result:
                try:
                    evaluation_data = {
                        'query': query,
                        'timestamp': datetime.now().isoformat(),
                        'analysis': convert_content_items(result.analysis),
                    }

                    if analysis_evaluator:
                        analysis_metrics = analysis_evaluator.evaluate_analysis(result, query)
                        analysis_metrics = convert_content_items(analysis_metrics)
                        trace.data['analysis_metrics'] = analysis_metrics
                        
                        evaluation_data.update({
                            'numerical_accuracy': float(analysis_metrics.get('numerical_accuracy', {}).get('score', 0.0)),
                            'query_understanding': float(analysis_metrics.get('query_understanding', {}).get('score', 0.0)),
                            'data_validation': float(analysis_metrics.get('data_validation', {}).get('score', 0.0)),
                            'reasoning_transparency': float(analysis_metrics.get('reasoning_transparency', {}).get('score', 0.0)),
                            'overall_score': float(analysis_metrics.get('overall_score', 0.0)),
                            'metrics_details': analysis_metrics,
                            'calculation_examples': analysis_metrics.get('numerical_accuracy', {}).get('details', {}).get('calculation_examples', []),
                            'term_coverage': float(analysis_metrics.get('query_understanding', {}).get('details', {}).get('term_coverage', 0.0)),
                            'analytical_elements': analysis_metrics.get('query_understanding', {}).get('details', {}),
                            'validation_checks': analysis_metrics.get('data_validation', {}).get('details', {}),
                            'explanation_patterns': analysis_metrics.get('reasoning_transparency', {}).get('details', {})
                        })
                        
                    safe_store(db.store_analysis_evaluation, evaluation_data, logger)
                    
                    trace.data.update({
                        "processing_steps": ["Analysis completed successfully"],
                        "analysis_metrics": evaluation_data
                    })
                    
                except Exception as eval_error:
                    logger.error(f"Analysis evaluation failed: {eval_error}", exc_info=True)
                    trace.data['evaluation_error'] = str(eval_error)

        else:
            error_msg = f"Tool {tool_name} not found"
            logger.error(error_msg)
            trace.data.update({
                "processing_steps": [f"Error: {error_msg}"],
                "error": error_msg,
                "success": False
            })
            db.close()
            return None, trace

        # Finalize execution
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        trace.data.update({
            "duration": duration,
            "success": True if result else False,
            "end_time": end_time.isoformat()
        })
        
        try:
            token_stats = trace.token_tracker.get_usage_stats()
            logger.info(f"Final token usage stats: {token_stats}")
            
            if token_stats['tokens']['total'] > 0:
                usage_msg = f"Total tokens used: {token_stats['tokens']['total']}"
                logger.info(usage_msg)
                trace.data["processing_steps"].append(usage_msg)
        except Exception as token_error:
            logger.warning(f"Could not retrieve token stats: {token_error}")
        
        logger.info(f"{tool_name} completed successfully")
        trace.data["processing_steps"].append(f"{tool_name} completed successfully")
        
        try:
            tracer = CustomTracer()
            tracer.save_trace(trace)
        except Exception as trace_save_error:
            logger.error(f"Failed to save trace: {trace_save_error}")
        
        db.close()
        return result, trace
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error running {tool_name}: {error_msg}", exc_info=True)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        trace.data.update({
            "end_time": end_time.isoformat(),
            "duration": duration,
            "error": error_msg,
            "success": False,
            "processing_steps": [f"Execution failed: {error_msg}"]
        })
        
        try:
            tracer = CustomTracer()
            tracer.save_trace(trace)
        except Exception as trace_save_error:
            logger.error(f"Failed to save error trace: {trace_save_error}")
        
        db.close()
        return None, trace