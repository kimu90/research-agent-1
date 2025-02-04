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
    print(f"\n{'='*50}\nStarting Tool Execution\n{'='*50}")
    print(f"Tool: {tool_name}")
    print(f"Query: {query}")
    print(f"Prompt: {prompt_name}\n")
    
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
        "prompt_used": prompt_name,
        "query": query
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
                print("Successfully initialized all evaluators for General Agent")
            else:
                analysis_evaluator = create_analysis_evaluator()
                accuracy_evaluator = source_coverage_evaluator = coherence_evaluator = None
                relevance_evaluator = None
                automated_test_evaluator = None
                print("Successfully initialized Analysis Evaluator")
                
        except Exception as eval_init_error:
            logger.error(f"Evaluation tools initialization failed: {eval_init_error}")
            print(f"Error initializing evaluators: {eval_init_error}")
            accuracy_evaluator = source_coverage_evaluator = coherence_evaluator = relevance_evaluator = automated_test_evaluator = analysis_evaluator = None

        result = None
        
        if tool_name == "General Agent":
            if tool is None:
                tool = GeneralAgent(
                    include_summary=True, 
                    prompt_name=prompt_name
                )
            
            trace.add_prompt_usage("general_agent_search", "general", prompt_name)
            print("\nExecuting General Agent query...")
            result = tool.invoke(input={"query": query})
            
            if result:
                try:
                    # Process content for storage first
                    content_to_store = result.content
                    if isinstance(content_to_store, list):
                        processed_content = [
                            item.get_text_content() if hasattr(item, 'get_text_content') 
                            else str(item) for item in content_to_store
                        ]
                    else:
                        processed_content = (
                            content_to_store.get_text_content() 
                            if hasattr(content_to_store, 'get_text_content') 
                            else str(content_to_store)
                        )
                        
                    # Store with proper metadata
                    content_result_data = {
                        'query_id': trace.data.get('query_id'),
                        'trace_id': trace.data.get('trace_id'),
                        'content': processed_content,
                        'content_type': tool_name.lower().replace(' ', '_'),
                        'evaluation_metrics': {
                            'factual_accuracy': trace.data.get('factual_accuracy', {}),
                            'source_coverage': trace.data.get('source_coverage', {}),
                            'logical_coherence': trace.data.get('logical_coherence', {}),
                            'answer_relevance': trace.data.get('answer_relevance', {}),
                            'automated_tests': trace.data.get('automated_tests', {}),
                            'analysis_metrics': trace.data.get('analysis_metrics', {})
                        }
                    }
                    
                    content_result_id = db.store_content_result(content_result_data)
                    if content_result_id:
                        logger.info(f"Successfully stored content with ID: {content_result_id}")
                    else:
                        logger.error("Failed to store content result")
                        
                except Exception as content_error:
                    logger.error(f"Error processing content: {str(content_error)}", exc_info=True)
                    trace.data['content_error'] = str(content_error)

                # Continue with rest of processing...
                try:
                    content_count = len(result.content) if result.content else 0
                    trace.data.update({
                        "content_new": content_count,
                        "content_reused": 0,
                        "processing_steps": [f"Content processed - New: {content_count}, Reused: 0"]
                    })
                    
                    # Run evaluations...
                    if accuracy_evaluator:
                        # Factual accuracy evaluation
                        factual_score, accuracy_details = accuracy_evaluator.evaluate_factual_accuracy(result)
                        accuracy_details = convert_content_items(accuracy_details)
                        trace.data['factual_accuracy'] = {'score': factual_score, 'details': accuracy_details}
                        trace.token_tracker.add_usage(100, 50, "llama3-70b-8192", "factual_accuracy")
                        
                    if source_coverage_evaluator:
                        # Source coverage evaluation
                        coverage_score, coverage_details = source_coverage_evaluator.evaluate_source_coverage(result)
                        coverage_details = convert_content_items(coverage_details)
                        trace.data['source_coverage'] = {'score': coverage_score, 'details': coverage_details}
                        trace.token_tracker.add_usage(100, 50, "llama3-70b-8192", "source_coverage")
                        
                    if coherence_evaluator:
                        # Coherence evaluation
                        coherence_score, coherence_details = coherence_evaluator.evaluate_logical_coherence(result)
                        coherence_details = convert_content_items(coherence_details)
                        trace.data['logical_coherence'] = {'score': coherence_score, 'details': coherence_details}
                        trace.token_tracker.add_usage(100, 50, "llama3-70b-8192", "logical_coherence")
                        
                    if relevance_evaluator:
                        # Relevance evaluation
                        relevance_score, relevance_details = relevance_evaluator.evaluate_answer_relevance(result, query)
                        relevance_details = convert_content_items(relevance_details)
                        trace.data['answer_relevance'] = {'score': relevance_score, 'details': relevance_details}
                        trace.token_tracker.add_usage(100, 50, "llama3-70b-8192", "answer_relevance")
                        
                    if automated_test_evaluator:
                        # Automated tests
                        content_for_test = ' '.join([str(item) for item in result.content]) if isinstance(result.content, list) else str(result.content)
                        test_score, test_details = automated_test_evaluator.evaluate_automated_tests(content_for_test, query)
                        test_details = convert_content_items(test_details)
                        trace.data['automated_tests'] = {'score': test_score, 'details': test_details}
                        trace.token_tracker.add_usage(100, 50, "llama3-70b-8192", "automated_tests")
                        
                except Exception as eval_error:
                    logger.error(f"Evaluation error: {str(eval_error)}", exc_info=True)
                    trace.data['evaluation_error'] = str(eval_error)

        elif tool_name == "Analysis Agent":
            if tool is None:
                tool = AnalysisAgent(data_folder="./data", prompt_name=prompt_name)
            
            trace.add_prompt_usage("analysis_agent", "analysis", "")
            print("\nExecuting Analysis Agent...")
            
            try:
                result = tool.invoke_analysis(input={"query": query, "dataset": dataset})
                
                if result:
                    try:
                        content_to_store = result.analysis
                        
                        content_result_data = {
                            'query_id': trace.data.get('query_id'),
                            'trace_id': trace.data.get('trace_id'),
                            'content': content_to_store,
                            'content_type': tool_name.lower().replace(' ', '_')
                        }
                        
                        if analysis_evaluator:
                            try:
                                analysis_metrics = analysis_evaluator.evaluate_analysis(result, query)
                                analysis_metrics = convert_content_items(analysis_metrics)
                                trace.data['analysis_metrics'] = analysis_metrics
                                
                                content_result_data['evaluation_metrics'] = {
                                    'analysis_metrics': analysis_metrics
                                }
                                
                            except Exception as eval_error:
                                logger.error(f"Analysis evaluation error: {str(eval_error)}", exc_info=True)
                                trace.data['evaluation_error'] = str(eval_error)
                        
                        content_result_id = db.store_content_result(content_result_data)
                        if content_result_id:
                            logger.info(f"Successfully stored content with ID: {content_result_id}")
                        else:
                            logger.error("Failed to store content result")
                            
                    except Exception as content_error:
                        logger.error(f"Error processing content: {str(content_error)}", exc_info=True)
                        trace.data['content_error'] = str(content_error)
                            
            except Exception as analysis_error:
                logger.error(f"Analysis error: {str(analysis_error)}", exc_info=True)
                trace.data['analysis_error'] = str(analysis_error)

        else:
            error_msg = f"Tool {tool_name} not found"
            logger.error(error_msg)
            trace.data.update({
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
        
        # Save trace
        try:
            tracer = CustomTracer()
            tracer.save_trace(trace)
        except Exception as trace_error:
            logger.error(f"Error saving trace: {str(trace_error)}", exc_info=True)
        
        db.close()
        return result, trace

    except Exception as e:
        # Main error handling
        error_msg = str(e)
        logger.error(f"Error in run_tool: {error_msg}", exc_info=True)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        trace.data.update({
            "end_time": end_time.isoformat(),
            "duration": duration,
            "error": error_msg,
            "success": False
        })
        
        db.close()
        return None, trace