import logging
import streamlit as st
from datetime import datetime

from tools import GeneralAgent
from research_agent.tracers import CustomTracer, QueryTrace
from utils.source_coverage import create_source_coverage_evaluator
from utils.evaluation import create_factual_accuracy_evaluator
from utils.answer_relevance import create_answer_relevance_evaluator
from utils.logical_coherence import create_logical_coherence_evaluator

def run_tool(tool_name: str, query: str, tool=None):
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
    
    logger.info(f"Starting research tool execution - Tool: {tool_name}")
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
        # Evaluation tools initialization
        try:
            accuracy_evaluator = create_factual_accuracy_evaluator()
            source_coverage_evaluator = create_source_coverage_evaluator()
            coherence_evaluator = create_logical_coherence_evaluator()
            relevance_evaluator = create_answer_relevance_evaluator()
        except Exception as eval_init_error:
            logger.error(f"Evaluation tools initialization failed: {eval_init_error}")
            accuracy_evaluator = source_coverage_evaluator = coherence_evaluator = relevance_evaluator = None

        if tool_name == "General Agent":
            if tool is None:
                tool = GeneralAgent(include_summary=True)
            
            trace.add_prompt_usage("general_agent_search", "general", "")
            
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
            
            # Comprehensive evaluation with relaxed validation
            try:
                factual_score = 0.5  # Default score
                coverage_score = 0.5
                coherence_score = 0.5
                relevance_score = 0.5
                
                # Factual accuracy evaluation
                try:
                    factual_score, accuracy_details = accuracy_evaluator.evaluate_factual_accuracy(result)
                    trace.data['factual_accuracy'] = {
                        'score': factual_score,
                        'details': accuracy_details
                    }
                    trace.token_tracker.add_usage(100, 50, "llama3-70b-8192", "factual_accuracy")                
                except Exception as e:
                    logger.warning(f"Factual accuracy evaluation failed: {e}")
                
                # Source coverage evaluation
                try:
                    coverage_score, coverage_details = source_coverage_evaluator.evaluate_source_coverage(result)
                    trace.data['source_coverage'] = {
                        'score': coverage_score,
                        'details': coverage_details
                    }
                    trace.token_tracker.add_usage(100, 50, "llama3-70b-8192", "source_coverage")                
                except Exception as e:
                    logger.warning(f"Source coverage evaluation failed: {e}")
                
                # Logical coherence evaluation
                try:
                    coherence_score, coherence_details = coherence_evaluator.evaluate_logical_coherence(result)
                    trace.data['logical_coherence'] = {
                        'score': coherence_score,
                        'details': coherence_details
                    }
                    trace.token_tracker.add_usage(100, 50, "llama3-70b-8192", "logical_coherence")
                except Exception as e:
                    logger.warning(f"Logical coherence evaluation failed: {e}")
                
                # Answer relevance evaluation
                try:
                    relevance_score, relevance_details = relevance_evaluator.evaluate_answer_relevance(result, query)
                    trace.data['answer_relevance'] = {
                        'score': relevance_score,
                        'details': relevance_details
                    }
                    trace.token_tracker.add_usage(100, 50, "llama3-70b-8192", "answer_relevance")
                except Exception as e:
                    logger.warning(f"Answer relevance evaluation failed: {e}")
                
            except Exception as eval_error:
                logger.error(f"Evaluation process failed: {eval_error}")
                trace.data['evaluation_error'] = str(eval_error)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            trace.data.update({
                "duration": duration,
                "success": True,
                "content_count": len(result.content) if result and result.content else 0
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
            
            logger.info("Research completed successfully")
            trace.data["processing_steps"].append("Research completed successfully")
            trace.data["end_time"] = datetime.now().isoformat()
            
            try:
                tracer = CustomTracer()
                tracer.save_trace(trace)
            except Exception as trace_save_error:
                logger.error(f"Failed to save trace: {trace_save_error}")
            
            return result, trace
        
        else:
            error_msg = f"Tool {tool_name} not found"
            logger.error(error_msg)
            trace.data.update({
                "processing_steps": [f"Error: {error_msg}"],
                "error": error_msg,
                "success": False
            })
            
            return None, trace
    
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
            "processing_steps": [f"Research failed: {error_msg}"]
        })
        
        try:
            tracer = CustomTracer()
            tracer.save_trace(trace)
        except Exception as trace_save_error:
            logger.error(f"Failed to save error trace: {trace_save_error}")
        
        return None, trace