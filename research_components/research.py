import logging
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

def run_tool(tool_name: str, query: str, dataset: str = None, analysis_type: str = None, tool=None):
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
    
    trace = QueryTrace(query)
    trace.data.update({
        "tool": tool_name,
        "tools_used": [tool_name],
        "processing_steps": [],
        "content_new": 0,
        "content_reused": 0
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
                tool = GeneralAgent(include_summary=True)
            
            trace.add_prompt_usage("general_agent_search", "general", "")
            result = tool.invoke(input={"query": query})
            
            if result:
                # Process research results
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
                        trace.data['factual_accuracy'] = {
                            'score': factual_score,
                            'details': accuracy_details
                        }
                        trace.token_tracker.add_usage(100, 50, "llama3-70b-8192", "factual_accuracy")
                        db.store_accuracy_evaluation({
                            'query': query,
                            'timestamp': datetime.now().isoformat(),
                            'factual_score': factual_score,
                            **accuracy_details
                        })

                    if source_coverage_evaluator:
                        coverage_score, coverage_details = source_coverage_evaluator.evaluate_source_coverage(result)
                        trace.data['source_coverage'] = {
                            'score': coverage_score,
                            'details': coverage_details
                        }
                        trace.token_tracker.add_usage(100, 50, "llama3-70b-8192", "source_coverage")
                        db.store_source_coverage({
                            'query': query,
                            'coverage_score': coverage_score,
                            **coverage_details
                        })

                    if coherence_evaluator:
                        coherence_score, coherence_details = coherence_evaluator.evaluate_logical_coherence(result)
                        trace.data['logical_coherence'] = {
                            'score': coherence_score,
                            'details': coherence_details
                        }
                        trace.token_tracker.add_usage(100, 50, "llama3-70b-8192", "logical_coherence")
                        db.store_logical_coherence({
                            'query': query,
                            'coherence_score': coherence_score,
                            **coherence_details
                        })

                    if relevance_evaluator:
                        relevance_score, relevance_details = relevance_evaluator.evaluate_answer_relevance(result, query)
                        trace.data['answer_relevance'] = {
                            'score': relevance_score,
                            'details': relevance_details
                        }
                        trace.token_tracker.add_usage(100, 50, "llama3-70b-8192", "answer_relevance")
                        
                        db.store_answer_relevance({
                            'query': query,
                            'relevance_score': relevance_score,
                            'query_match_percentage': relevance_details.get('query_match_percentage', 0),
                            'semantic_similarity': relevance_details.get('similarity', 0),
                            'keyword_coverage': relevance_details.get('keyword_overlap', 0),
                            'entity_coverage': relevance_details.get('entity_coverage', 0),
                            'topic_focus': relevance_details.get('topic_focus', 0),
                            'off_topic_sentences': json.dumps(relevance_details.get('off_topic_sentences', [])),
                            'total_sentences': len(result.content) if hasattr(result, 'content') else 0,
                            'information_density': relevance_details.get('information_density', 0),
                            'context_alignment_score': relevance_details.get('context_alignment_score', 0)
                        })

                    if automated_test_evaluator:
                        content_for_test = ' '.join(result.content) if isinstance(result.content, list) else str(result.content)
                        test_score, test_details = automated_test_evaluator.evaluate_automated_tests(content_for_test, query)
                        trace.data['automated_tests'] = {
                            'score': test_score,
                            'details': test_details
                        }
                        trace.token_tracker.add_usage(100, 50, "llama3-70b-8192", "automated_tests")
                        db.store_test_results(query, test_score, test_details)

                except Exception as eval_error:
                    logger.error(f"Research evaluation failed: {eval_error}")
                    trace.data['evaluation_error'] = str(eval_error)

        elif tool_name == "Analysis Agent":
            if tool is None:
                tool = AnalysisAgent(data_folder="./data")  # Initialize with data folder
            
            trace.add_prompt_usage("analysis_agent", "analysis", "")
            result = tool.invoke_analysis(input={
                "query": query,
                "dataset": dataset  # Add the dataset parameter
            })
            
            if result:
                try:
                    evaluation_data = {
                        'query': query,
                        'timestamp': datetime.now().isoformat(),
                        'analysis': result.analysis,
                    }

                    # Run analysis-specific evaluations
                    if analysis_evaluator:
                        analysis_metrics = analysis_evaluator.evaluate_analysis(result, query)
                        trace.data['analysis_metrics'] = analysis_metrics
                        
                        evaluation_data.update({
                            'numerical_accuracy': analysis_metrics.get('numerical_accuracy', {}).get('score', 0.0),
                            'query_understanding': analysis_metrics.get('query_understanding', {}).get('score', 0.0),
                            'data_validation': analysis_metrics.get('data_validation', {}).get('score', 0.0),
                            'reasoning_transparency': analysis_metrics.get('reasoning_transparency', {}).get('score', 0.0),
                            'overall_score': analysis_metrics.get('overall_score', 0.0),
                            'metrics_details': json.dumps(analysis_metrics),
                            'calculation_examples': json.dumps(analysis_metrics.get('numerical_accuracy', {}).get('details', {}).get('calculation_examples', [])),
                            'term_coverage': analysis_metrics.get('query_understanding', {}).get('details', {}).get('term_coverage', 0.0),
                            'analytical_elements': json.dumps(analysis_metrics.get('query_understanding', {}).get('details', {})),
                            'validation_checks': json.dumps(analysis_metrics.get('data_validation', {}).get('details', {})),
                            'explanation_patterns': json.dumps(analysis_metrics.get('reasoning_transparency', {}).get('details', {}))
                        })
                        
                    # Store complete analysis evaluation
                    db.store_analysis_evaluation(evaluation_data)
                    
                    trace.data.update({
                        "processing_steps": ["Analysis completed successfully"],
                        "analysis_metrics": evaluation_data
                    })
                    
                except Exception as eval_error:
                    logger.error(f"Analysis evaluation failed: {eval_error}")
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