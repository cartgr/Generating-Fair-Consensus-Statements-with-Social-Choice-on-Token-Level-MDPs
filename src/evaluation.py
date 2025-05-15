import numpy as np
import pandas as pd
import json
import time
import argparse
import yaml
import os
import logging
from pathlib import Path
from scipy.spatial.distance import cosine
from typing import Dict, List, Tuple, Optional, Union, Any
from pydantic import BaseModel, Field

# Set OpenAI logging to WARNING level to suppress debug messages
import logging
openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.WARNING)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

# Try importing OpenAI for LLM judge functionality, but make it optional
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .utils import (
    get_prompt_logprobs,
    get_embedding,
)

# Set up logger
logger = logging.getLogger(__name__)

# Pydantic models for O3 API responses
class RankingResponse(BaseModel):
    """Pydantic model for ranking response from O3 API."""
    reasoning: str 
    ranking: list[int]
    method_ranking: Dict[str, int] = Field(default_factory=dict)  # Make this optional with default empty dict

class ScoreResponse(BaseModel):
    """Pydantic model for LLM judge scoring response."""
    reasoning: str
    score: int


class StatementEvaluator:
    """
    A class to evaluate consensus statements using various metrics.
    Supports both basic metrics (avg_logprob, cosine_similarity) and
    advanced metrics (LLM-as-judge, multiple welfare metrics).
    """
    
    def __init__(
        self,
        evaluation_model: str,
        llm_judge_model: Optional[str] = None,
        openai_client: Optional[Any] = None,
        include_llm_judge: bool = False,
        include_comparative_ranking: bool = True,  # Default to True to run by default
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        verbose: bool = True,
    ):
        """
        Initialize the StatementEvaluator with the specified models and options.
        
        Args:
            evaluation_model: The model to use for logprob-based evaluation
            llm_judge_model: The OpenAI model to use for LLM-as-judge evaluation
            openai_client: An optional pre-initialized OpenAI client
            include_llm_judge: Whether to include LLM-as-judge evaluation
            include_comparative_ranking: Whether to include comparative statement ranking
            embedding_model: The model to use for embedding calculations
            verbose: Whether to print detailed evaluation information
        """
        self.evaluation_model = evaluation_model
        self.embedding_model = embedding_model
        self.verbose = verbose
        
        # Setup LLM judge and comparative ranking if requested
        self.include_llm_judge = include_llm_judge
        self.include_comparative_ranking = include_comparative_ranking
        self.llm_judge_model = llm_judge_model
        self.openai_client = openai_client
        
        # Check if we need the OpenAI client (for either LLM judge or comparative ranking)
        needs_openai = self.include_llm_judge or self.include_comparative_ranking
        
        if needs_openai:
            if not OPENAI_AVAILABLE:
                if self.verbose:
                    print("Warning: OpenAI package not available. LLM Judge and comparative ranking will be disabled.")
                self.include_llm_judge = False
                self.include_comparative_ranking = False
            elif not self.openai_client and not self.llm_judge_model:
                if self.verbose:
                    print("Warning: No OpenAI client or model provided. LLM Judge and comparative ranking will be disabled.")
                self.include_llm_judge = False
                self.include_comparative_ranking = False
            elif not self.openai_client:
                # Try to initialize OpenAI client if not provided
                try:
                    load_dotenv()
                    api_key = os.getenv("OPENAI_API_KEY")
                    if api_key:
                        self.openai_client = OpenAI(api_key=api_key)
                        if self.verbose:
                            if self.include_llm_judge and self.include_comparative_ranking:
                                print(f"Initialized OpenAI client for LLM Judge and comparative ranking ({self.llm_judge_model}).")
                            elif self.include_llm_judge:
                                print(f"Initialized OpenAI client for LLM Judge ({self.llm_judge_model}).")
                            elif self.include_comparative_ranking:
                                print(f"Initialized OpenAI client for comparative ranking ({self.llm_judge_model}).")
                    else:
                        if self.verbose:
                            print("Warning: OPENAI_API_KEY not found. LLM Judge and comparative ranking will be disabled.")
                        self.include_llm_judge = False
                        self.include_comparative_ranking = False
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Failed to initialize OpenAI client: {e}. LLM Judge and comparative ranking will be disabled.")
                    self.include_llm_judge = False
                    self.include_comparative_ranking = False
    
    def evaluate_statement(
        self,
        statement: str,
        issue: str,
        agent_opinions: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Evaluates a single statement using the configured metrics.
        
        Args:
            statement: The statement to evaluate
            issue: The issue being discussed
            agent_opinions: Dictionary mapping agent names to their opinions
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        if self.verbose:
            print(f"Evaluating statement: '{statement}'")
            print(f"Using evaluation model: '{self.evaluation_model}'")
        
        results = {}
        epsilon = 1e-9  # Small value for log nash stability
        
        # Store individual agent utilities/scores
        agent_cosine_similarities = {}
        agent_avg_probs = {}  # Utility for logprob-based welfare (avg probability)
        agent_avg_logprobs = {}  # Store the simple average logprob metric
        
        # --- Embedding/Logprob/Cosine Evaluation ---
        if self.verbose:
            print("  Calculating embedding/logprob/cosine metrics...")
        
        try:
            # --- Calculate Statement Embedding (once) ---
            if self.verbose:
                print("    Calculating embedding for the statement...")
            
            statement_embedding = get_embedding(statement, model=self.embedding_model)
            if statement_embedding is None:
                if self.verbose:
                    print("      Warning: Could not get embedding for the statement. Similarities will be None.")
                results["statement_embedding"] = None
            else:
                if self.verbose:
                    print(f"      Statement embedding calculated (length: {len(statement_embedding)}).")
                results["statement_embedding"] = statement_embedding
            
            # --- Calculate metrics for each agent ---
            for agent_id, opinion in agent_opinions.items():
                # --- Log Probability Calculation ---
                if self.verbose:
                    print(f"    Calculating logprobs for Agent: {agent_id}")
                
                system_prompt = (
                    f"Issue: {issue}. "
                    f"Agent's Opinion: {opinion}. "
                    "Here is a consensus statement that perfectly aligns with the agent's opinion:"
                )
                user_prompt = statement
                
                tokens, logprobs = get_prompt_logprobs(
                    model=self.evaluation_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                
                if self.verbose:
                    print(f"      tokens: {len(tokens) if tokens else 0}, logprobs: {len(logprobs) if logprobs else 0}")
                
                # --- Calculate Avg Logprob and Avg Probability (Utility) ---
                avg_logprob_for_agent = None
                agent_utility_avg_prob = None
                raw_logprobs = logprobs  # Use the logprobs directly from get_prompt_logprobs
                
                if raw_logprobs:
                    valid_logprobs = [lp for lp in raw_logprobs if lp is not None]
                    if valid_logprobs:
                        avg_logprob_for_agent = np.mean(valid_logprobs)
                        if self.verbose:
                            print(f"      Avg Logprob for {agent_id}: {avg_logprob_for_agent:.4f}")
                        
                        try:
                            probabilities = np.exp(np.array(valid_logprobs))
                            avg_prob = np.mean(probabilities)
                            agent_utility_avg_prob = avg_prob
                            if self.verbose:
                                print(f"      Avg Probability (Utility) for {agent_id}: {agent_utility_avg_prob:.4f}")
                        except Exception as e:
                            if self.verbose:
                                print(f"      Error calculating avg probability for {agent_id}: {e}")
                            agent_utility_avg_prob = None
                    else:
                        if self.verbose:
                            print(f"      Warning: No valid logprobs returned for agent {agent_id}.")
                else:
                    if self.verbose:
                        print(f"      Warning: Could not retrieve logprobs for agent {agent_id}.")
                
                # Store both types of logprob metrics (averaged log prob and averaged prob)
                results[f"avg_logprob_{agent_id}"] = avg_logprob_for_agent
                results[f"utility_avg_logprob_{agent_id}"] = avg_logprob_for_agent
                agent_avg_probs[agent_id] = agent_utility_avg_prob
                
                # --- Cosine Similarity Calculation ---
                if self.verbose:
                    print(f"    Calculating similarity for Agent: {agent_id}")
                
                agent_opinion_embedding = get_embedding(opinion, model=self.embedding_model)
                cosine_sim = None
                
                if statement_embedding is not None and agent_opinion_embedding is not None:
                    try:
                        stmt_emb_np = np.array(statement_embedding)
                        agent_emb_np = np.array(agent_opinion_embedding)
                        
                        if np.linalg.norm(stmt_emb_np) == 0 or np.linalg.norm(agent_emb_np) == 0:
                            if self.verbose:
                                print(f"      Warning: Zero vector encountered for agent {agent_id}. Similarity set to 0.")
                            cosine_sim = 0.0
                        else:
                            similarity = 1 - cosine(stmt_emb_np, agent_emb_np)
                            if np.isnan(similarity):
                                if self.verbose:
                                    print(f"      Warning: Cosine similarity resulted in NaN for agent {agent_id}. Setting to None.")
                                cosine_sim = None
                            else:
                                cosine_sim = similarity
                                if self.verbose:
                                    print(f"      Cosine similarity for {agent_id}: {cosine_sim:.4f}")
                    except Exception as e:
                        if self.verbose:
                            print(f"      Error calculating cosine similarity for agent {agent_id}: {e}")
                        cosine_sim = None
                elif statement_embedding is None:
                    if self.verbose:
                        print(f"      Skipping similarity for {agent_id} (statement embedding failed).")
                else:
                    if self.verbose:
                        print(f"      Warning: Could not get embedding for agent {agent_id}'s opinion. Similarity set to None.")
                
                # Store similarity metrics
                results[f"cosine_similarity_{agent_id}"] = cosine_sim
                results[f"utility_cosine_similarity_{agent_id}"] = cosine_sim
                agent_cosine_similarities[agent_id] = cosine_sim
            
            # --- Calculate Welfare Metrics for Cosine Similarity ---
            if self.verbose:
                print("\n  Calculating Cosine Similarity Welfare Metrics...")
            
            valid_similarities = [
                s for s in agent_cosine_similarities.values() 
                if s is not None and np.isfinite(s)
            ]
            
            if valid_similarities:
                # Egalitarian welfare (minimum utility)
                results["egalitarian_welfare_cosine"] = min(valid_similarities)
                results["utility_egalitarian_welfare_cosine"] = min(valid_similarities)
                
                # Utilitarian welfare (sum of utilities)
                results["utilitarian_welfare_cosine"] = sum(valid_similarities)
                results["utility_utilitarian_welfare_cosine"] = sum(valid_similarities)
                
                # Log Nash welfare (sum of log utilities, with epsilon to avoid log(0))
                positive_similarities = [max(s, epsilon) for s in valid_similarities]
                results["log_nash_welfare_cosine"] = sum(np.log(positive_similarities))
                results["utility_log_nash_welfare_cosine"] = sum(np.log(positive_similarities))
                
                if self.verbose:
                    print(f"    Egalitarian (Cosine): {results['egalitarian_welfare_cosine']:.4f}")
                    print(f"    Utilitarian (Cosine): {results['utilitarian_welfare_cosine']:.4f}")
                    print(f"    Log Nash (Cosine):    {results['log_nash_welfare_cosine']:.4f}")
            else:
                results["egalitarian_welfare_cosine"] = np.nan
                results["utilitarian_welfare_cosine"] = np.nan
                results["log_nash_welfare_cosine"] = np.nan
                results["utility_egalitarian_welfare_cosine"] = np.nan
                results["utility_utilitarian_welfare_cosine"] = np.nan
                results["utility_log_nash_welfare_cosine"] = np.nan
                
                if self.verbose:
                    print("    Could not calculate Cosine Welfare (no valid similarities).")
            
            # --- Calculate Welfare Metrics for Average Probability ---
            if self.verbose:
                print("\n  Calculating Average Probability Welfare Metrics...")
            
            valid_avg_probs = [
                p for p in agent_avg_probs.values() 
                if p is not None and np.isfinite(p)
            ]
            
            # Calculate average logprobs for perplexity
            valid_avg_logprobs = {}
            for agent_id, avg_logprob in results.items():
                if agent_id.startswith("avg_logprob_"):
                    agent_name = agent_id.replace("avg_logprob_", "")
                    if avg_logprob is not None and np.isfinite(avg_logprob):
                        valid_avg_logprobs[agent_name] = avg_logprob
            
            # Calculate perplexity (exp(-logprob)) for each agent
            if valid_avg_logprobs:
                for agent_name, avg_logprob in valid_avg_logprobs.items():
                    perplexity = np.exp(-avg_logprob)
                    results[f"perplexity_{agent_name}"] = perplexity
                    if self.verbose:
                        print(f"    Perplexity for {agent_name}: {perplexity:.4f}")
            
            if valid_avg_probs:
                # Egalitarian welfare (minimum utility)
                results["egalitarian_welfare_avg_prob"] = min(valid_avg_probs)
                results["utility_egalitarian_welfare_logprob"] = min(valid_avg_probs)
                
                # Utilitarian welfare (sum of utilities)
                results["utilitarian_welfare_avg_prob"] = sum(valid_avg_probs)
                results["utility_utilitarian_welfare_logprob"] = sum(valid_avg_probs)
                
                # Log Nash welfare (sum of log utilities, with epsilon to avoid log(0))
                positive_avg_probs = [max(p, epsilon) for p in valid_avg_probs]
                results["log_nash_welfare_avg_prob"] = sum(np.log(positive_avg_probs))
                results["utility_log_nash_welfare_logprob"] = sum(np.log(positive_avg_probs))
                
                if self.verbose:
                    print(f"    Egalitarian (Avg Prob): {results['egalitarian_welfare_avg_prob']:.4f}")
                    print(f"    Utilitarian (Avg Prob): {results['utilitarian_welfare_avg_prob']:.4f}")
                    print(f"    Log Nash (Avg Prob):    {results['log_nash_welfare_avg_prob']:.4f}")
            else:
                results["egalitarian_welfare_avg_prob"] = np.nan
                results["utilitarian_welfare_avg_prob"] = np.nan
                results["log_nash_welfare_avg_prob"] = np.nan
                results["utility_egalitarian_welfare_logprob"] = np.nan
                results["utility_utilitarian_welfare_logprob"] = np.nan
                results["utility_log_nash_welfare_logprob"] = np.nan
                
                if self.verbose:
                    print("    Could not calculate Avg Prob Welfare (no valid avg probabilities).")
            
            # Calculate perplexity-based welfare metrics
            if valid_avg_logprobs:
                # Calculate perplexity for each agent
                agent_perplexities = {agent: np.exp(-logprob) for agent, logprob in valid_avg_logprobs.items()}
                
                # Egalitarian welfare (minimum utility - for perplexity, higher is worse, so max is egalitarian)
                # Since perplexity is a "cost" rather than a "utility", we report the worst (maximum) value
                # Lower perplexity is better, so max perplexity is the min utility
                results["egalitarian_welfare_perplexity"] = max(agent_perplexities.values())
                
                # Utilitarian welfare (sum) - for perplexity, we invert the values since higher is worse
                results["utilitarian_welfare_perplexity"] = sum(agent_perplexities.values())
                
                # Log Nash welfare - for perplexity, we use product of inverse perplexities
                inverse_perplexities = [1.0 / max(perp, epsilon) for perp in agent_perplexities.values()]
                results["log_nash_welfare_perplexity"] = sum(np.log(inverse_perplexities))
                
                if self.verbose:
                    print(f"\n  Perplexity-Based Welfare Metrics:")
                    print(f"    Egalitarian (Perplexity, lower is better): {results['egalitarian_welfare_perplexity']:.4f}")
                    print(f"    Utilitarian (Perplexity Sum): {results['utilitarian_welfare_perplexity']:.4f}")
                    print(f"    Log Nash (Inverse Perplexity): {results['log_nash_welfare_perplexity']:.4f}")
            else:
                results["egalitarian_welfare_perplexity"] = np.nan
                results["utilitarian_welfare_perplexity"] = np.nan
                results["log_nash_welfare_perplexity"] = np.nan
                
                if self.verbose:
                    print("    Could not calculate Perplexity Welfare (no valid avg logprobs).")
        
        except Exception as e:
            if self.verbose:
                print(f"  ERROR calculating embedding/logprob/cosine metrics: {e}")
            
            results["error_embedding_logprob"] = str(e)
            
            # Ensure welfare keys exist even on error
            for welfare_type in ["egalitarian", "utilitarian", "log_nash"]:
                for metric_type in ["cosine", "avg_prob"]:
                    results.setdefault(f"{welfare_type}_welfare_{metric_type}", np.nan)
                    if welfare_type == "egalitarian":
                        results.setdefault(f"utility_{welfare_type}_welfare_{'logprob' if metric_type == 'avg_prob' else metric_type}", np.nan)
                    elif welfare_type == "utilitarian":
                        results.setdefault(f"utility_{welfare_type}_welfare_{'logprob' if metric_type == 'avg_prob' else metric_type}", np.nan)
                    elif welfare_type == "log_nash":
                        results.setdefault(f"utility_{welfare_type}_welfare_{'logprob' if metric_type == 'avg_prob' else metric_type}", np.nan)
        
        # --- LLM-as-Judge Evaluation (optional) ---
        agent_llm_scores = {}
        
        if self.include_llm_judge and self.openai_client and self.llm_judge_model:
            if self.verbose:
                print(f"\n  Evaluating with LLM Judge ({self.llm_judge_model})...")
            
            judge_start_time = time.time()
            
            for agent_id, opinion in agent_opinions.items():
                if self.verbose:
                    print(f"    Judging for Agent {agent_id}...")
                
                system_prompt = (
                    "You are an impartial judge evaluating how well a consensus statement "
                    "represents a specific agent's opinion on a given issue. "
                    "Output ONLY a JSON object containing two keys: "
                    "'reasoning' (a string with your step-by-step evaluation process) and "
                    "'score' (an integer between 1 (very poorly represented) and 5 (very well represented))."
                )
                
                user_prompt = (
                    f"Issue: {issue}\n\n"
                    f"Agent's Opinion: {opinion}\n\n"
                    f"Consensus Statement: {statement}\n\n"
                    "Evaluate how well the Consensus Statement represents the Agent's Opinion on the Issue. "
                    "Provide your response as a JSON object with 'reasoning' and 'score' keys."
                )
                
                try:
                    # Initialize variables to default values
                    message_content = "{}"
                    finish_reason = "error"
                    
                    if self.llm_judge_model == "o3":
                        # Use the O3 API with the responses endpoint and Pydantic model
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ]
                        
                        try:
                            # Skip the beta parsing API and use regular API directly to avoid schema issues
                            response = self.openai_client.chat.completions.create(
                                model="gpt-4.1",  
                                messages=messages,
                                response_format={"type": "json_object"},
                                temperature=0.0,
                                seed=42,  # Use consistent seed for reproducibility
                            )
                            finish_reason = response.choices[0].finish_reason
                            message_content = response.choices[0].message.content
                            
                            if self.verbose:
                                print(f"      Successfully obtained response using standard API")
                        
                        except Exception as e:
                            # Handle any API errors or JSON parsing issues
                            if self.verbose:
                                print(f"      Error using API: {e}")
                            
                            # Set default values for failure case
                            finish_reason = "error"
                            message_content = "{}"
                    else:
                        # Use the standard chat completions API for other models
                        try:
                            response = self.openai_client.chat.completions.create(
                                model=self.llm_judge_model,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt},
                                ],
                                response_format={"type": "json_object"},
                                temperature=0.0,  # Deterministic output for judging
                            )
                            
                            finish_reason = response.choices[0].finish_reason
                            message_content = response.choices[0].message.content
                        except Exception as e:
                            if self.verbose:
                                print(f"      Error in standard API: {e}")
                            # Leave message_content and finish_reason with default values
                    
                    score = np.nan  # Default to NaN
                    reasoning = "Error: No reasoning extracted."  # Default reasoning
                    
                    if finish_reason == "stop":
                        try:
                            # The entire message should be the JSON object
                            parsed_json = json.loads(message_content)
                            
                            if (
                                isinstance(parsed_json, dict)
                                and "score" in parsed_json
                                and "reasoning" in parsed_json
                            ):
                                potential_score = parsed_json["score"]
                                potential_reasoning = parsed_json["reasoning"]
                                
                                # Validate score
                                if (
                                    isinstance(potential_score, int)
                                    and 1 <= potential_score <= 5
                                ):
                                    score = potential_score
                                    if self.verbose:
                                        print(f"      Agent {agent_id} Score: {score}")
                                else:
                                    if self.verbose:
                                        print(f"      WARNING: Agent {agent_id} - Invalid score value: {potential_score}")
                                    reasoning = f"Error: Invalid score value ({potential_score}). Raw JSON: {message_content}"
                                
                                # Validate reasoning (basic check for string type)
                                if isinstance(potential_reasoning, str):
                                    reasoning = potential_reasoning
                                    # Optionally print a snippet if verbose
                                    # if self.verbose:
                                    #    print(f"      Agent {agent_id} Reasoning: {reasoning[:50]}...")
                                else:
                                    if self.verbose:
                                        print(f"      WARNING: Agent {agent_id} - Invalid reasoning type: {type(potential_reasoning)}")
                                    reasoning = f"Error: Invalid reasoning type ({type(potential_reasoning)}). Raw JSON: {message_content}"
                            
                            else:
                                if self.verbose:
                                    print(f"      WARNING: Agent {agent_id} - Required keys ('score', 'reasoning') missing or invalid JSON structure.")
                                reasoning = f"Error: Missing keys or invalid structure. Raw JSON: {message_content}"
                                
                        except json.JSONDecodeError as json_e:
                            if self.verbose:
                                print(f"      ERROR: Agent {agent_id} - Failed to parse JSON response: {json_e}")
                                print(f"      Raw content: {message_content}")
                            reasoning = f"Error: JSONDecodeError - {json_e}. Raw content: {message_content}"
                        except Exception as parse_e:
                            if self.verbose:
                                print(f"      ERROR: Agent {agent_id} - Error processing score/reasoning: {parse_e}")
                            reasoning = f"Error: Processing error - {parse_e}. Raw content: {message_content}"
                    
                    elif finish_reason == "length":
                        if self.verbose:
                            print(f"      ERROR: Agent {agent_id} - LLM Judge response truncated due to length.")
                        results[f"error_llm_judge_{agent_id}"] = "Response truncated (length)"
                        reasoning = f"Error: Response truncated (length). Partial content: {message_content}"
                    elif finish_reason == "content_filter":
                        if self.verbose:
                            print(f"      ERROR: Agent {agent_id} - LLM Judge response stopped due to content filter.")
                        results[f"error_llm_judge_{agent_id}"] = "Content filter"
                        reasoning = f"Error: Content filter triggered. Partial content: {message_content}"
                    else:
                        if self.verbose:
                            print(f"      WARNING: Agent {agent_id} - Unexpected finish reason: {finish_reason}")
                        results[f"error_llm_judge_{agent_id}"] = f"Unexpected finish reason: {finish_reason}"
                        reasoning = f"Error: Unexpected finish reason '{finish_reason}'. Content: {message_content}"
                    
                    # Store score regardless of success/failure for welfare calculation
                    agent_llm_scores[agent_id] = score  # Store NaN if error occurred
                    results[f"llm_judge_score_{agent_id}"] = score
                    results[f"llm_judge_reasoning_{agent_id}"] = reasoning
                
                except Exception as api_e:
                    if self.verbose:
                        print(f"    ERROR calling LLM Judge API for Agent {agent_id}: {api_e}")
                    results[f"error_llm_judge_{agent_id}"] = str(api_e)
                    agent_llm_scores[agent_id] = np.nan  # Ensure score is NaN on API error
                    results[f"llm_judge_score_{agent_id}"] = np.nan  # Also store NaN in individual result
                    results[f"llm_judge_reasoning_{agent_id}"] = f"Error: API call failed - {api_e}"
            
            results["llm_judge_time_s"] = time.time() - judge_start_time
            
            # --- Calculate Welfare Metrics for LLM Judge Scores ---
            if self.verbose:
                print("\n  Calculating LLM Judge Score Welfare Metrics...")
            
            valid_scores = [
                s for s in agent_llm_scores.values() 
                if pd.notna(s) and np.isfinite(s)
            ]
            
            if valid_scores:
                # Calculate all welfare metrics for LLM Judge scores
                results["egalitarian_welfare_llm_judge"] = min(valid_scores)
                results["llm_judge_egalitarian_welfare"] = min(valid_scores)
                
                results["utilitarian_welfare_llm_judge"] = sum(valid_scores)
                results["llm_judge_utilitarian_welfare"] = sum(valid_scores)
                
                # Scores are 1-5, so log is safe. Use max(s, epsilon) just in case of unexpected 0.
                positive_scores = [max(s, epsilon) for s in valid_scores]
                results["log_nash_welfare_llm_judge"] = sum(np.log(positive_scores))
                results["llm_judge_log_nash_welfare"] = sum(np.log(positive_scores))
                
                if self.verbose:
                    print(f"    Egalitarian (LLM Judge): {results['egalitarian_welfare_llm_judge']:.4f}")
                    print(f"    Utilitarian (LLM Judge): {results['utilitarian_welfare_llm_judge']:.4f}")
                    print(f"    Log Nash (LLM Judge):    {results['log_nash_welfare_llm_judge']:.4f}")
            else:
                # Set all LLM judge welfare metrics to NaN if no valid scores
                results["egalitarian_welfare_llm_judge"] = np.nan
                results["utilitarian_welfare_llm_judge"] = np.nan
                results["log_nash_welfare_llm_judge"] = np.nan
                results["llm_judge_egalitarian_welfare"] = np.nan
                results["llm_judge_utilitarian_welfare"] = np.nan
                results["llm_judge_log_nash_welfare"] = np.nan
                
                if self.verbose:
                    print("    Could not calculate LLM Judge Welfare (no valid scores).")
        
        elif self.include_llm_judge:
            # LLM judge was requested but not available
            if self.verbose:
                print("  Skipping LLM Judge evaluation (OpenAI client or model not available).")
            
            # Ensure LLM judge welfare keys exist even when skipped
            results["egalitarian_welfare_llm_judge"] = np.nan
            results["utilitarian_welfare_llm_judge"] = np.nan
            results["log_nash_welfare_llm_judge"] = np.nan
            results["llm_judge_egalitarian_welfare"] = np.nan
            results["llm_judge_utilitarian_welfare"] = np.nan
            results["llm_judge_log_nash_welfare"] = np.nan
        
        return results
    
    def evaluate_comparative_rankings(
        self,
        statements: Dict[str, Union[str, Dict]],
        issue: str,
        agent_opinions: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Evaluates all statements comparatively using an LLM judge to rank them for each agent.
        Calculates maximin welfare and stores complete ranking matrix.
        
        Args:
            statements: Dictionary mapping method names to statements or statement dictionaries
            issue: The issue being discussed
            agent_opinions: Dictionary mapping agent names to their opinions
            
        Returns:
            Dictionary containing ranking results and welfare metrics
        """
        if not self.openai_client or not self.llm_judge_model:
            raise ValueError("OpenAI client and model required for comparative ranking")
        
        # Process statements to ensure we have string values
        processed_statements = {}
        for method_name, statement_data in statements.items():
            if isinstance(statement_data, dict):
                statement = statement_data.get("statement", "")
            else:
                statement = statement_data
                
            if statement and statement != "ERROR" and not pd.isna(statement):
                processed_statements[method_name] = statement
        
        if not processed_statements:
            raise ValueError("No valid statements to rank")
            
        # Initialize results
        results = {
            "issue": issue,
            "ranking_matrix": {},  # Will hold all agent-method rankings
            "method_min_ranks": {},  # Will hold the minimum rank each method receives
            "method_avg_ranks": {},  # Will hold the average rank each method receives
        }
        
        if self.verbose:
            print(f"\n--- Evaluating Comparative Rankings with LLM Judge ---")
            print(f"Total statements to rank: {len(processed_statements)}")
            print(f"Total agents: {len(agent_opinions)}")
        
        # For each agent, get a ranking of all statements
        ranking_start_time = time.time()
        
        for agent_id, opinion in agent_opinions.items():
            if self.verbose:
                print(f"\nGetting rankings for Agent {agent_id}...")
                
            # Create the prompt for this agent
            system_prompt, user_prompt = self.create_ranking_prompt(
                agent_id, opinion, issue, processed_statements
            )
            
            try:
                # Initialize variables to ensure they're defined even in case of exceptions
                method_ranking = {}
                ranking_reasoning = "No reasoning provided"
                raw_ranking = []
                response_json = {}
                
                # Call the OpenAI API to get the ranking
                try:
                    if self.llm_judge_model == "o3":
                        # Use the O3 API with the responses endpoint and Pydantic model
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ]
                        
                        try:
                            # Skip the beta parsing API and use regular API directly to avoid schema issues
                            response = self.openai_client.chat.completions.create(
                                model="gpt-4.1",
                                messages=messages,
                                response_format={"type": "json_object"},
                                temperature=0.0,
                                seed=42,  # Use consistent seed for reproducibility
                            )
                            response_content = response.choices[0].message.content
                            response_json = json.loads(response_content)
                            
                            # Extract values from response_json
                            method_ranking = response_json.get("method_ranking", {})
                            ranking_reasoning = response_json.get("reasoning", "No reasoning provided")
                            raw_ranking = response_json.get("ranking", [])
                            
                            if self.verbose:
                                print(f"  Successfully parsed response using standard API")
                        
                        except Exception as e:
                            # Handle any API errors or JSON parsing issues
                            if self.verbose:
                                print(f"  Error using standard API call: {e}")
                            
                            # Initialize with empty values
                            method_ranking = {}
                            ranking_reasoning = f"Error occurred: {str(e)}"
                            raw_ranking = []
                            response_json = {}
                    else:
                        # Use the standard chat completions API for other models
                        response = self.openai_client.chat.completions.create(
                            model=self.llm_judge_model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            response_format={"type": "json_object"},
                            temperature=0.0,  # Deterministic output for judging
                        )
                        response_content = response.choices[0].message.content
                        response_json = json.loads(response_content)
                        
                        # Extract values from response_json
                        method_ranking = response_json.get("method_ranking", {})
                        ranking_reasoning = response_json.get("reasoning", "No reasoning provided")
                        raw_ranking = response_json.get("ranking", [])
                        
                except Exception as api_e:
                    # Handle any API call failures
                    error_msg = f"API call error: {api_e}"
                    if self.verbose:
                        print(f"  {error_msg}")
                    # Leave variables with default values
                
                # Validate the rankings
                if not method_ranking or len(method_ranking) != len(processed_statements):
                    if self.verbose:
                        print(f"  Warning: Invalid method ranking for agent {agent_id}. Expected {len(processed_statements)} methods, got {len(method_ranking)}")
                        
                    # Try to reconstruct method ranking from raw ranking if available
                    if raw_ranking and len(raw_ranking) == len(processed_statements):
                        if self.verbose:
                            print(f"  Attempting to reconstruct method ranking from raw ranking...")
                        
                        method_list = list(processed_statements.keys())
                        reconstructed_ranking = {}
                        
                        # Map statement numbers (1-indexed) to methods
                        for rank, stmt_num in enumerate(raw_ranking, 1):
                            try:
                                # Statement numbers in the prompt are 1-indexed
                                stmt_idx = int(stmt_num) - 1
                                if 0 <= stmt_idx < len(method_list):
                                    method = method_list[stmt_idx]
                                    reconstructed_ranking[method] = rank
                            except (ValueError, TypeError) as e:
                                if self.verbose:
                                    print(f"  Error converting statement number: {e}")
                        
                        if len(reconstructed_ranking) == len(processed_statements):
                            if self.verbose:
                                print(f"  Successfully reconstructed method ranking")
                            method_ranking = reconstructed_ranking
                        else:
                            if self.verbose:
                                print(f"  Failed to reconstruct method ranking. Using empty ranking.")
                    elif self.verbose:
                        print(f"  No valid raw ranking available to reconstruct method ranking.")
                
                # Store the ranking for this agent
                results["ranking_matrix"][agent_id] = {
                    "method_ranking": method_ranking,
                    "reasoning": ranking_reasoning,
                    "raw_ranking": raw_ranking,
                    "raw_response": response_content
                }
                
                if self.verbose:
                    print(f"  Rankings obtained for agent {agent_id}")
                    print(f"  Ranking: {method_ranking}")
                
            except Exception as e:
                error_msg = f"Error processing ranking for agent {agent_id}: {e}"
                if self.verbose:
                    print(f"  {error_msg}")
                results["ranking_matrix"][agent_id] = {
                    "error": error_msg,
                    "method_ranking": {},
                    "reasoning": f"Error: {e}"
                }
        
        results["comparative_ranking_time_s"] = time.time() - ranking_start_time
        
        # Calculate maximin welfare (minimum rank each method receives)
        if self.verbose:
            print("\nCalculating welfare metrics from rankings...")
        
        # Initialize dictionaries if they don't exist
        results["method_min_ranks"] = {}
        results["method_max_ranks"] = {}
        results["method_avg_ranks"] = {}
            
        for method_name in processed_statements.keys():
            method_ranks = []
            
            for agent_id in agent_opinions.keys():
                if agent_id in results["ranking_matrix"]:
                    agent_results = results["ranking_matrix"][agent_id]
                    if "method_ranking" in agent_results:
                        rank = agent_results["method_ranking"].get(method_name)
                        if rank is not None:
                            method_ranks.append(rank)
            
            if method_ranks:
                # Lower rank is better (1 is best)
                min_rank = min(method_ranks)  # Best rank (lowest number)
                max_rank = max(method_ranks)  # Worst rank (highest number)
                avg_rank = sum(method_ranks) / len(method_ranks)
                
                # Store metrics
                results["method_min_ranks"][method_name] = min_rank  # Best rank
                results["method_max_ranks"][method_name] = max_rank  # Worst rank
                results["method_avg_ranks"][method_name] = avg_rank  # Average rank
                
                if self.verbose:
                    print(f"  {method_name}: min={min_rank}, max={max_rank}, avg={avg_rank:.2f}")
        
        # Calculate maximin welfare (best method based on worst rank received)
        # For maximin welfare, we want to find the method with the lowest maximum rank
        # (i.e., the method whose worst rank is better than any other method's worst rank)
        if results["method_max_ranks"]:
            maximin_method = min(results["method_max_ranks"].items(), key=lambda x: x[1])[0]
            maximin_value = results["method_max_ranks"][maximin_method]
            results["maximin_welfare_method"] = maximin_method
            results["maximin_welfare_value"] = maximin_value
            
            if self.verbose:
                print(f"\nMaximin welfare method: {maximin_method} (worst rank: {maximin_value})")
        else:
            if self.verbose:
                print("Warning: No valid method ranks available for maximin welfare calculation.")
            results["maximin_welfare_method"] = None
            results["maximin_welfare_value"] = None
        
        # Calculate utilitarian welfare (best method based on average rank)
        if results["method_avg_ranks"]:
            utilitarian_method = min(results["method_avg_ranks"].items(), key=lambda x: x[1])[0]
            utilitarian_value = results["method_avg_ranks"][utilitarian_method]
            results["utilitarian_welfare_method"] = utilitarian_method
            results["utilitarian_welfare_value"] = utilitarian_value
            
            if self.verbose:
                print(f"Utilitarian welfare method: {utilitarian_method} (avg rank: {utilitarian_value:.2f})")
        else:
            if self.verbose:
                print("Warning: No valid method ranks available for utilitarian welfare calculation.")
            results["utilitarian_welfare_method"] = None
            results["utilitarian_welfare_value"] = None
        
        return results
        
    def evaluate_statements(
        self,
        statements: Dict[str, Union[str, Dict]],
        issue: str,
        agent_opinions: Dict[str, str],
    ) -> pd.DataFrame:
        """
        Evaluates multiple statements for the same issue and agent opinions.
        
        Args:
            statements: Dictionary mapping method names to either statement strings
                      or dictionaries containing statement and metadata
            issue: The issue being discussed
            agent_opinions: Dictionary mapping agent names to their opinions
        
        Returns:
            DataFrame containing evaluation results for all statements
        """
        results_list = []
        
        for method_name, statement_data in statements.items():
            if self.verbose:
                print(f"\n--- Evaluating Method: {method_name} ---")
            
            # Handle both string statements and dictionary format
            if isinstance(statement_data, dict):
                statement = statement_data.get("statement", "")
                seed_value = statement_data.get("seed")
                row_index = statement_data.get("row_index")
            else:
                statement = statement_data
                seed_value = None
                row_index = None
            
            # Extract base method name and parameters if present
            base_method = method_name
            param_dict = {}
            seed_from_key = None
            
            # Parse parameters from method name if it contains them
            if " (" in method_name:
                parts = method_name.split(" (", 1)
                base_method = parts[0]
                
                if parts[1].endswith(")"):
                    param_str = parts[1].rstrip(")")
                    
                    # Parse parameters into a dictionary
                    for param_item in param_str.split(", "):
                        if "=" in param_item:
                            param_name, param_value = param_item.split("=", 1)
                            # Try to convert to numeric if possible
                            try:
                                param_value = float(param_value)
                                # Convert to int if it's a whole number
                                if param_value.is_integer():
                                    param_value = int(param_value)
                            except ValueError:
                                # Keep as string if not numeric
                                pass
                            
                            # Store parameter in the dictionary for later addition to results
                            param_dict[f"param_{param_name}"] = param_value
            
            # Check for seed value in method name (format: [seed=X])
            if "[seed=" in method_name and "]" in method_name:
                seed_parts = method_name.split("[seed=", 1)[1].split("]", 1)
                if seed_parts:
                    try:
                        # Extract seed from the method key if present
                        seed_from_key = int(seed_parts[0])
                    except (ValueError, IndexError):
                        pass
            
            # Use seed_value from the data dictionary if available, otherwise use seed from key
            final_seed = seed_value if seed_value is not None else seed_from_key
            
            run_data = {
                "method": base_method,  # Store the base method name without parameters
                "issue": issue,
                "statement": statement,
                "method_with_params": method_name,  # Store the full method name with parameters
            }
            
            # Add all parsed parameters to the results
            for param_name, param_value in param_dict.items():
                run_data[param_name] = param_value
            
            # Add seed if available
            if final_seed is not None:
                run_data["seed"] = final_seed
            
            # Add original row index if available (for later merging)
            if row_index is not None:
                run_data["original_row_index"] = row_index
            
            try:
                start_time = time.time()
                evaluation_results = self.evaluate_statement(
                    statement,
                    issue,
                    agent_opinions,
                )
                evaluation_time = time.time() - start_time
                
                # Add evaluation time to results
                run_data["evaluation_time_s"] = evaluation_time
                
                # Add all evaluation results to the run data
                for key, value in evaluation_results.items():
                    # Skip raw embeddings to keep results clean
                    if key != "statement_embedding":
                        run_data[key] = value
            
            except Exception as e:
                if self.verbose:
                    print(f"ERROR evaluating {method_name}: {e}")
                    import traceback
                    print(traceback.format_exc())
                
                run_data["error_message"] = str(e)
            
            results_list.append(run_data)
        
        return pd.DataFrame(results_list)
    
    def create_ranking_prompt(
        self,
        agent_id: str,
        agent_opinion: str,
        issue: str,
        statements_dict: Dict[str, str]
    ) -> Tuple[str, str]:
        """
        Creates a prompt for ranking multiple statements from an agent's perspective.
        
        Args:
            agent_id: The ID of the agent whose perspective is being evaluated
            agent_opinion: The agent's stated opinion on the issue
            issue: The issue being discussed
            statements_dict: Dictionary mapping method names to statements
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = (
            f"You are evaluating consensus statements from the perspective of a specific agent. "
            f"Your task is to rank multiple statements based on how well they represent "
            f"the agent's opinion and interests on a given issue. "
            f"Use ONLY the agent's stated opinion to determine the ranking."
        )
        
        user_prompt = (
            f"## Issue\n{issue}\n\n"
            f"## Agent {agent_id}'s Opinion\n{agent_opinion}\n\n"
            f"## Statements to Rank\n"
        )
        
        # Add statements with identifiers
        for i, (method_name, statement) in enumerate(statements_dict.items(), 1):
            user_prompt += f"Statement {i} ({method_name}):\n{statement}\n\n"
        
        user_prompt += (
            f"## Task\n"
            f"From Agent {agent_id}'s perspective, rank all statements from most favorable (1) "
            f"to least favorable ({len(statements_dict)}) based on how well they represent the agent's "
            f"opinion and interests.\n\n"
            f"Provide your ranking as a JSON object with:\n"
            f"1. 'reasoning': brief explanation for your ranking decisions\n"
            f"2. 'ranking': an array of statement numbers in ranked order (best to worst)\n"
            f"3. 'method_ranking': a mapping of method names to their rank positions\n\n"
            f"For example: {{'reasoning': 'Statement 3 best represents the agent's concerns about...',\n"
            f"'ranking': [3, 1, 2], 'method_ranking': {{'Method A': 2, 'Method B': 3, 'Method C': 1}}}}"
        )
        
        return system_prompt, user_prompt
    
    def evaluate_results_file(
        self,
        results_path: str,
        config_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        is_seed_specific: bool = False,
    ) -> pd.DataFrame:
        """
        Evaluates statements from an existing results file.
        
        Args:
            results_path: Path to the results CSV file
            config_path: Path to the experiment config YAML (optional, will try to find in same dir)
            output_dir: Directory to save evaluation results (default is subdirectory of results dir)
            is_seed_specific: Whether this is a seed-specific evaluation (skip saving to parent folder)
            
        Returns:
            DataFrame containing evaluation results
        """
        # Normalize paths
        results_path = Path(results_path)
        results_dir = results_path.parent
        
        if config_path is None:
            # Try to find config in the same directory
            config_path = results_dir / "config.yaml"
        
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")
        
        # Load results
        results_df = pd.read_csv(results_path)
        
        # Extract statements and method information
        statements = {}
        for _, row in results_df.iterrows():
            method = row.get("method", "Unknown method")
            statement = row.get("statement", "No statement generated")
            
            # Skip entries with errors
            if statement == "ERROR" or pd.isna(statement):
                if self.verbose:
                    print(f"Skipping evaluation for method '{method}' with error")
                continue
            
            # Get parameter columns
            param_cols = [col for col in row.index if col.startswith("param_")]

            # Extract seed value if present
            seed_value = None
            if "seed" in row and pd.notna(row["seed"]):
                seed_value = row["seed"]

            # Create a dictionary of parameter values
            params_dict = {}
            for param in param_cols:
                if pd.notna(row[param]):
                    params_dict[param] = row[param]

            # Use the standardized method identifier function
            from .utils import create_method_identifier
            method_key = create_method_identifier(
                method_name=method,
                params_dict=params_dict,
                include_seed=(seed_value is not None),
                seed_value=seed_value
            )
            
            # Store the full row data with the statement for later reference
            statements[method_key] = {
                "statement": statement,
                "seed": seed_value if pd.notna(seed_value) else None,
                "row_index": row.name,  # Keep track of the original row index
            }
        
        # Load config to get issue and agent opinions
        config = None
        issue = None
        agent_opinions = {}
        
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                
            if config:
                issue = config.get("scenario", {}).get("issue")
                agent_opinions = config.get("scenario", {}).get("agent_opinions", {})
        
        if not issue or not agent_opinions:
            raise ValueError(
                "Could not extract issue and agent opinions from config. "
                "Please provide a valid config file."
            )
        
        # Determine output directory
        if output_dir is None:
            eval_model_name = self.evaluation_model.replace("/", "_")
            judge_model_name = (
                self.llm_judge_model.replace("/", "_")
                if self.include_llm_judge and self.llm_judge_model
                else "no_judge"
            )
            output_dir_name = f"posthoc_eval_{eval_model_name}_judge_{judge_model_name}"
            output_dir = results_dir / output_dir_name
        else:
            output_dir = Path(output_dir)
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluate all statements
        if self.verbose:
            print(f"\nEvaluating {len(statements)} statements for issue: {issue}")
            print(f"Using evaluation model: {self.evaluation_model}")
            
            if self.include_llm_judge and self.openai_client and self.llm_judge_model:
                print(f"Using LLM Judge model: {self.llm_judge_model}")
            elif self.include_llm_judge:
                print("LLM Judge evaluation requested but unavailable (client or model missing).")
            else:
                print("Skipping LLM Judge evaluation.")
                
            if self.include_comparative_ranking and self.openai_client and self.llm_judge_model:
                print(f"Using comparative statement ranking with model: {self.llm_judge_model}")
            elif self.include_comparative_ranking:
                print("Comparative ranking requested but unavailable (client or model missing).")
            else:
                print("Skipping comparative statement ranking.")
        
        # Run standard evaluation
        evaluation_results = self.evaluate_statements(
            statements,
            issue,
            agent_opinions,
        )
        
        # If comparative ranking is enabled, run and merge the results
        if self.include_comparative_ranking:
            print("\n--- Comparative Ranking Status ---")
            print(f"include_comparative_ranking: {self.include_comparative_ranking}")
            print(f"openai_client available: {self.openai_client is not None}")
            print(f"llm_judge_model: {self.llm_judge_model}")
            
            if self.openai_client and self.llm_judge_model:
                print("All requirements met - running comparative ranking evaluation...")
                
                if self.verbose:
                    print("\nRunning comparative ranking evaluation...")
                    
                try:
                    # Run comparative ranking evaluation
                    ranking_results = self.evaluate_comparative_rankings(
                        statements,
                        issue,
                        agent_opinions,
                    )
                    
                    # Create a dataframe with the comparative ranking results
                    ranking_df = pd.DataFrame()
                    
                    # Add methods as rows
                    for method_name in statements.keys():
                        row_data = {
                            "method": method_name,
                            "issue": issue,
                        }
                        
                        # Add basic method ranking metrics
                        if method_name in ranking_results.get("method_min_ranks", {}):
                            row_data["comparative_min_rank"] = ranking_results["method_min_ranks"][method_name]
                        if method_name in ranking_results.get("method_max_ranks", {}):
                            row_data["comparative_max_rank"] = ranking_results["method_max_ranks"][method_name]
                        if method_name in ranking_results.get("method_avg_ranks", {}):
                            row_data["comparative_avg_rank"] = ranking_results["method_avg_ranks"][method_name]
                        
                        # Add per-agent rankings
                        for agent_id in agent_opinions.keys():
                            if agent_id in ranking_results.get("ranking_matrix", {}):
                                agent_results = ranking_results["ranking_matrix"][agent_id]
                                if "method_ranking" in agent_results:
                                    rank = agent_results["method_ranking"].get(method_name)
                                    if rank is not None:
                                        row_data[f"comparative_rank_{agent_id}"] = rank
                        
                        # Add welfare indicators
                        if "maximin_welfare_method" in ranking_results and ranking_results["maximin_welfare_method"] == method_name:
                            row_data["is_maximin_welfare_best"] = 1
                        else:
                            row_data["is_maximin_welfare_best"] = 0
                            
                        if "utilitarian_welfare_method" in ranking_results and ranking_results["utilitarian_welfare_method"] == method_name:
                            row_data["is_utilitarian_welfare_best"] = 1
                        else:
                            row_data["is_utilitarian_welfare_best"] = 0
                        
                        # Add to dataframe
                        ranking_df = pd.concat([ranking_df, pd.DataFrame([row_data])], ignore_index=True)
                    
                    # Store the full ranking matrix in a separate file
                    if output_dir:
                        # Instead of saving to the main output_dir, create seed_0 subfolder
                        # if this is not a seed-specific evaluation
                        if is_seed_specific:
                            ranking_matrix_path = Path(output_dir) / "comparative_ranking_matrix.json"
                        else:
                            seed_dir = Path(output_dir) / "seed_0"
                            seed_dir.mkdir(parents=True, exist_ok=True)
                            ranking_matrix_path = seed_dir / "comparative_ranking_matrix.json"
                        
                        with open(ranking_matrix_path, "w") as f:
                            json.dump(ranking_results["ranking_matrix"], f, indent=2)
                        
                        if self.verbose:
                            print(f"Saved complete ranking matrix to {ranking_matrix_path}")
                    
                    # Merge with main evaluation results
                    evaluation_results = pd.merge(
                        evaluation_results, 
                        ranking_df[["method"] + [col for col in ranking_df.columns if col != "method" and col != "issue"]], 
                        on="method", 
                        how="left"
                    )
                    
                    if self.verbose:
                        print("Comparative ranking results merged with evaluation results")
                    
                except Exception as e:
                    if self.verbose:
                        print(f"ERROR in comparative ranking: {e}")
                        import traceback
                        print(traceback.format_exc())
            else:
                print("WARNING: Cannot run comparative ranking - missing OpenAI client or model.")
        
        
        # Merge the evaluation results with the original results
        if self.verbose:
            print(f"\nMerging evaluation results with original data...")
        
        # Make a copy of the original results to add evaluation metrics
        combined_results = results_df.copy()
        
        # Create a mapping from original row indices to evaluation results
        eval_by_index = {}
        for _, eval_row in evaluation_results.iterrows():
            if "original_row_index" in eval_row:
                eval_by_index[eval_row["original_row_index"]] = eval_row
        
        # Add evaluation metrics to the combined results
        eval_columns = []
        for i, row in combined_results.iterrows():
            if i in eval_by_index:
                eval_row = eval_by_index[i]
                
                # Update the evaluation status
                combined_results.at[i, "evaluation_status"] = "completed"
                
                # Add all evaluation metrics from the evaluated result
                for col, value in eval_row.items():
                    # Skip columns we don't want to duplicate
                    if col not in ["method", "statement", "issue", "original_row_index"] and col not in combined_results.columns:
                        combined_results.at[i, col] = value
                        if col not in eval_columns:
                            eval_columns.append(col)
                
                # Extract and preserve parameter values if present in the method_with_params field
                if "method_with_params" in eval_row and " (" in eval_row["method_with_params"]:
                    parts = eval_row["method_with_params"].split(" (", 1)
                    if parts[1].endswith(")"):
                        param_str = parts[1].rstrip(")")
                        for param_item in param_str.split(", "):
                            if "=" in param_item:
                                param_name, param_value = param_item.split("=", 1)
                                # Try to convert to numeric if possible
                                try:
                                    param_value = float(param_value)
                                    if param_value.is_integer():
                                        param_value = int(param_value)
                                except ValueError:
                                    pass
                                
                                param_col = f"param_{param_name}"
                                if param_col not in combined_results.columns:
                                    combined_results.at[i, param_col] = param_value
                                    if param_col not in eval_columns:
                                        eval_columns.append(param_col)
            else:
                # If the row wasn't evaluated, mark it accordingly
                combined_results.at[i, "evaluation_status"] = "skipped"
        
        if self.verbose:
            print(f"Added {len(eval_columns)} evaluation metrics to {len(eval_by_index)} rows")
        
        # Save the evaluation results to appropriate directory
        if is_seed_specific:
            # Save directly to the provided output directory since this is already seed-specific
            evaluation_csv_path = output_dir / "evaluation_results.csv"
            evaluation_results.to_csv(evaluation_csv_path, index=False)
            
            if self.verbose:
                print(f"Evaluation results saved to {evaluation_csv_path}")
                
            # Save config used for evaluation
            eval_config = {
                "original_config": config,
                "evaluation": {
                    "evaluation_model": self.evaluation_model,
                    "embedding_model": self.embedding_model,
                    "include_llm_judge": self.include_llm_judge,
                    "llm_judge_model": self.llm_judge_model if self.include_llm_judge else None,
                    "original_results_path": str(results_path),
                    "statements_evaluated": len(statements),
                    "rows_processed": len(eval_by_index),
                    "total_rows": len(results_df),
                }
            }
            
            config_output_path = output_dir / "evaluation_config.yaml"
            with open(config_output_path, "w") as f:
                yaml.dump(eval_config, f, default_flow_style=False)
            
            if self.verbose:
                print(f"Evaluation config saved to {config_output_path}")
        else:
            # Create a seed_0 directory and save there
            seed_dir = output_dir / "seed_0"
            seed_dir.mkdir(parents=True, exist_ok=True)
            
            evaluation_csv_path = seed_dir / "evaluation_results.csv"
            evaluation_results.to_csv(evaluation_csv_path, index=False)
            
            if self.verbose:
                print(f"Evaluation results saved to {evaluation_csv_path}")
                
            # Save config used for evaluation
            eval_config = {
                "original_config": config,
                "evaluation": {
                    "evaluation_model": self.evaluation_model,
                    "embedding_model": self.embedding_model,
                    "include_llm_judge": self.include_llm_judge,
                    "llm_judge_model": self.llm_judge_model if self.include_llm_judge else None,
                    "original_results_path": str(results_path),
                    "statements_evaluated": len(statements),
                    "rows_processed": len(eval_by_index),
                    "total_rows": len(results_df),
                }
            }
            
            config_output_path = seed_dir / "evaluation_config.yaml"
            with open(config_output_path, "w") as f:
                yaml.dump(eval_config, f, default_flow_style=False)
            
            if self.verbose:
                print(f"Evaluation config saved to {config_output_path}")
        
        return combined_results


# Backward compatibility functions that match the original API

def evaluate_statement(
    statement: str, 
    issue: str, 
    agent_opinions: Dict[str, str], 
    evaluation_model: str
) -> Dict[str, Any]:
    """
    Legacy function for backwards compatibility with the original API.
    Evaluates a statement using the basic metrics only (avg_logprob, cosine_similarity).
    
    Args:
        statement: The statement to evaluate
        issue: The issue being discussed
        agent_opinions: Dictionary mapping agent names to their opinions
        evaluation_model: The model to use for evaluation
    
    Returns:
        Dictionary containing evaluation results
    """
    # Create evaluator with verbose=False to match original behavior
    evaluator = StatementEvaluator(
        evaluation_model=evaluation_model,
        include_llm_judge=False,
        verbose=False,
    )
    
    # Run evaluation
    results = evaluator.evaluate_statement(statement, issue, agent_opinions)
    
    # Filter results to match original API (only include avg_logprob and cosine_similarity)
    filtered_results = {}
    for key, value in results.items():
        if key.startswith("avg_logprob_") or key.startswith("cosine_similarity_"):
            filtered_results[key] = value
        elif key in ["egalitarian_welfare_logprob", "egalitarian_welfare_cosine"]:
            filtered_results[key] = value
    
    return filtered_results


# Command-line interface for direct usage
def main():
    """Command-line interface for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate consensus statements using various metrics"
    )
    
    # Input sources
    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "--results-file",
        type=str,
        help="Path to a results CSV file to evaluate",
    )
    input_group.add_argument(
        "--config",
        type=str,
        help="Path to a config YAML file (required if not using --results-file)",
    )
    input_group.add_argument(
        "--statements-file",
        type=str,
        help="Path to a YAML or JSON file with method:statement pairs",
    )
    
    # Evaluation options
    eval_group = parser.add_argument_group("Evaluation Options")
    eval_group.add_argument(
        "--evaluation-model",
        type=str,
        default="meta-llama/Llama-3-70b-Instruct-Turbo",
        help="Model to use for logprob/embedding evaluation",
    )
    eval_group.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Model to use for embeddings",
    )
    eval_group.add_argument(
        "--include-llm-judge",
        action="store_true",
        help="Include LLM-as-judge evaluation",
    )
    eval_group.add_argument(
        "--llm-judge-model",
        type=str,
        default="gpt-4-turbo",
        help="OpenAI model to use for LLM-as-judge evaluation",
    )
    eval_group.add_argument(
        "--include-comparative-ranking",
        action="store_true",
        help="Include comparative ranking of statements for egalitarian welfare",
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save evaluation results",
    )
    output_group.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.results_file and not (args.config and args.statements_file):
        parser.error(
            "Either --results-file or both --config and --statements-file must be provided"
        )
    
    # Setup evaluator
    evaluator = StatementEvaluator(
        evaluation_model=args.evaluation_model,
        llm_judge_model=args.llm_judge_model if (args.include_llm_judge or args.include_comparative_ranking) else None,
        include_llm_judge=args.include_llm_judge,
        include_comparative_ranking=args.include_comparative_ranking,
        embedding_model=args.embedding_model,
        verbose=not args.quiet,
    )
    
    # Run evaluation based on input type
    if args.results_file:
        # Evaluate from results file
        results_df = evaluator.evaluate_results_file(
            results_path=args.results_file,
            config_path=args.config,  # Optional, will try to find in results dir
            output_dir=args.output_dir,
        )
    else:
        # Evaluate from config and statements file
        # Load config
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        
        issue = config.get("scenario", {}).get("issue")
        agent_opinions = config.get("scenario", {}).get("agent_opinions", {})
        
        if not issue or not agent_opinions:
            print("Error: Config file does not contain issue and agent opinions")
            return
        
        # Load statements
        statements = {}
        if args.statements_file.endswith((".yaml", ".yml")):
            with open(args.statements_file, "r") as f:
                statements = yaml.safe_load(f)
        elif args.statements_file.endswith(".json"):
            with open(args.statements_file, "r") as f:
                statements = json.load(f)
        else:
            print("Error: Statements file must be YAML or JSON")
            return
        
        # Run evaluation
        results_df = evaluator.evaluate_statements(
            statements=statements,
            issue=issue,
            agent_opinions=agent_opinions,
        )
        
        # Save results if output directory specified
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results
            results_path = output_dir / "evaluation_results.csv"
            results_df.to_csv(results_path, index=False)
            
            if not args.quiet:
                print(f"Results saved to {results_path}")
            
            # Save config
            eval_config = {
                "scenario": {
                    "issue": issue,
                    "agent_opinions": agent_opinions,
                },
                "evaluation": {
                    "evaluation_model": args.evaluation_model,
                    "embedding_model": args.embedding_model,
                    "include_llm_judge": args.include_llm_judge,
                    "llm_judge_model": args.llm_judge_model if args.include_llm_judge else None,
                },
                "statements": statements,
            }
            
            config_path = output_dir / "evaluation_config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(eval_config, f, default_flow_style=False)
            
            if not args.quiet:
                print(f"Evaluation config saved to {config_path}")
    
    # Print summary
    if not args.quiet:
        print("\n--- Evaluation Summary ---")
        print(f"Statements evaluated: {len(results_df)}")
        print(f"Evaluation model: {args.evaluation_model}")
        if args.include_llm_judge:
            print(f"LLM Judge model: {args.llm_judge_model}")
        print("------------------------")


if __name__ == "__main__":
    main()