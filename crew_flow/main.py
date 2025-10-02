#!/usr/bin/env python
import json
import os
import asyncio
import pandas as pd
from typing import List, Dict, Tuple, Literal, Optional, Any
from pydantic import BaseModel, Field
from crewai import LLM
from crewai.flow.flow import Flow, listen, start
from crew_flow.crews.prod_eng_crew import ProdEngCrew, AgentScores
import mlflow
from domino.aisystems.tracing import add_tracing, search_traces
from domino.aisystems.logging import DominoRun, log_evaluation
from pathlib import Path
from dotenv import load_dotenv
import yaml
from openai import OpenAI

load_dotenv()

# Load configuration from YAML
def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration settings from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary containing models, 
                       instructions, and other settings
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Define our models for structured data
class TicketRecord(BaseModel):
    ticket_id: int
    description: str
    customers_requesting: List[str]
    priority: Literal["P0", "P1", "P2", "P3"]

class FinalScore(BaseModel):
    final_score: float
    alignment_rationale: str
    effort_rationale: str

# Define our flow state
class TicketState(BaseModel):
    ticket_record: Optional[TicketRecord] = None
    final_score: Optional[FinalScore] = None

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_arr_map(path: str) -> Dict[str, float]:
    """
    Load customer Annual Recurring Revenue (ARR) mapping from CSV file.
    
    Args:
        path (str): Path to CSV file containing 'Company' and 'ARR' columns
        
    Returns:
        Dict[str, float]: Mapping of company names to their ARR values
    """
    df = pd.read_csv(path)
    arr_map = dict(zip(df['Company'], df['ARR']))
    return arr_map


def parse_customers(s: str) -> List[str]:
    """
    Parse customer list from JSON string format with fallback handling.
    
    Attempts to parse a JSON string containing customer names. If JSON parsing
    fails, falls back to manual parsing of comma-separated values.
    
    Args:
        s (str): JSON string or comma-separated string of customer names
        
    Returns:
        List[str]: List of customer names
        
    Example:
        >>> parse_customers('["Company A", "Company B"]')
        ['Company A', 'Company B']
        >>> parse_customers('Company A, Company B')
        ['Company A', 'Company B']
    """
    try:
        result = json.loads(s)
        return result
    except Exception as e:
        # Fallback parsing for malformed JSON
        result = [c.strip().strip('"') for c in s.strip('[]').split(',') if c.strip()]
        return result

PRIORITY_MAP = {"P0": 1, "P1": 2, "P2": 3, "P3": 4}

def reach_score_fn(customers: List[str]) -> int:
    """
    Calculate the reach score based on requesting customers' Annual Recurring Revenue (ARR).
    
    The reach score represents the potential business impact of implementing a feature
    based on the total ARR of customers requesting it. Higher ARR customers contribute
    to a higher reach score.
    
    Args:
        customers (List[str]): List of customer names requesting the feature
        
    Returns:
        int: Reach score from 1-5, where:
             - 1: Low reach (ARR < $1M)
             - 2-4: Medium reach (ARR $1M-$4M)
             - 5: High reach (ARR ≥ $5M)
             
    Note:
        Uses global arr_map_global for customer ARR lookup.
        Unknown customers default to $0 ARR.
    """
    total = sum(arr_map_global.get(c, 0) for c in customers)
    score = min(max(round(total / 1_000_000), 1), 5)
    return score

def impact_score_fn(priority: str) -> int:
    """
    Calculate the impact score based on the feature request priority level.
    
    Converts customer-assigned priority levels to numerical impact scores.
    Higher priority (P0) indicates more critical business impact.
    
    Args:
        priority (str): Priority level string ("P0", "P1", "P2", or "P3")
        
    Returns:
        int: Impact score where:
             - P0: 1 (highest impact - critical)
             - P1: 2 (high impact)
             - P2: 3 (medium impact) 
             - P3: 4 (lower impact)
             
    Note:
        Lower numerical scores represent higher impact priorities.
    """
    score = PRIORITY_MAP[priority]
    return score

def final_score_fn(reach: int, impact: int, align: int, effort: int) -> float:
    """
    Calculate the final prioritization score for a feature request.
    
    Uses a weighted formula that balances business value (reach × impact × alignment)
    against implementation cost (effort). Higher scores indicate higher priority
    features that should be implemented first.
    
    Args:
        reach (int): Reach score (1-5) based on requesting customers' ARR
        impact (int): Impact score (1-4) based on priority level (P0-P3)
        align (int): Strategic alignment score (1-5) from alignment agent
        effort (int): Implementation effort score (1-5) from effort agent
        
    Returns:
        float: Final prioritization score using formula:
               (reach × impact × alignment) / effort
               
    Note:
        Higher scores = higher priority. Effort in denominator means
        easier implementations get boosted scores.
    """
    score = (reach * impact * align) / effort
    return score

# ── Flow ─────────────────────────────────────────────────────────────────────

class TicketFlow(Flow[TicketState]):
    """Flow for evaluating a ticket"""

    @start()
    def parse_ticket(self):
        """Parse the ticket record and calculate the reach and impact scores"""
        print("\n=== Evaluating Ticket ===\n")

        # Get ticket record from inputs
        ticket_record = self.state.ticket_record
        reach_score = reach_score_fn(ticket_record.customers_requesting)
        impact_score = impact_score_fn(ticket_record.priority)
        return reach_score, impact_score

    @listen(parse_ticket)
    def get_final_score(self, reach_impact_scores: Tuple[int, int]):
        """Calculate the final score for a ticket"""
        print("Calculating final score...")
        reach_score, impact_score = reach_impact_scores
        
        # Get crew output and parse it
        crew_output = ProdEngCrew().crew().kickoff(inputs={
            "reach_score": reach_score,
            "impact_score": impact_score,
            "request_description": self.state.ticket_record.description
        })
        agent_scores = crew_output.pydantic
        final_score = final_score_fn(reach_score, impact_score, agent_scores.alignment_score, agent_scores.effort_score)
        
        self.state.final_score = FinalScore(
            final_score=final_score,
            alignment_rationale=agent_scores.alignment_rationale,
            effort_rationale=agent_scores.effort_rationale
        )
        return self.state

def judge_response(inputs, output):
    """
    Evaluate the accuracy of effort rationale using an AI judge for inline evaluation.
    
    This function is used as an inline evaluator that automatically assesses
    the quality of effort estimations during trace execution. It extracts
    the effort rationale and request description from the trace data.
    
    Args:
        inputs (dict): Trace input data containing ticket information
        output: Trace output data containing agent responses
        
    Returns:
        Dict[str, int]: Dictionary with evaluation metric:
                       {"eng_effort_accuracy": rating} where rating is 1-5
    """
    request_description = inputs['ticket_record']['description']
    effort_rationale = output['final_score']['effort_rationale']
    client = OpenAI()
    config = load_config(JUDGE_CONFIG)
    judge_prompt = config['judge_prompt'].format(
        effort_rationale=effort_rationale,
        request_description=request_description
    )

    completion = client.chat.completions.create(
        model=config['models']['judge'],
        messages=[
            {"role": "user", "content": judge_prompt}
        ]
    )
    rating = int(completion.choices[0].message.content)
    return {"eng_effort_accuracy": rating}

@add_tracing(name="prioritize_ticket", autolog_frameworks=["crewai"], evaluator=judge_response)
def kickoff(ticket_record: TicketRecord):
    """Run the ticket flow"""
    flow = TicketFlow()
    result = flow.kickoff(inputs={"ticket_record": ticket_record})
    
    print("\n=== Flow Complete ===")
    print("Your ticket has been evaluated.")
    print(f"Final score: {result.final_score.final_score}")
    print(f"Alignment rationale: {result.final_score.alignment_rationale}")
    print(f"Effort rationale: {result.final_score.effort_rationale}")
    
    return result

if __name__ == "__main__":
    # Define configuration paths - you'll need to set these
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    input_csv = PROJECT_ROOT / "feature_requests.csv"  # Set your input CSV path
    output_csv = PROJECT_ROOT / "scored_tickets.csv"  # Set your output CSV path
    customers_csv = PROJECT_ROOT / "customers.csv"  # Set your customers CSV path
    CONFIG_PATH = PROJECT_ROOT / "crew_flow/crews/config/agents.yaml"  
    JUDGE_CONFIG = PROJECT_ROOT / "crew_flow/crews/config/judge.yaml" 
    
    df = pd.read_csv(input_csv)
    global arr_map_global
    arr_map_global = load_arr_map(customers_csv)

    tickets = []
    for idx, r in enumerate(df.itertuples(index=False)):
        try:
            ticket = TicketRecord(
                ticket_id=r.ticket_id,
                description=r.description,
                customers_requesting=parse_customers(r.customers_requesting),
                priority=r.customer_priority.strip(),
            )
            tickets.append(ticket)
        except Exception as e:
            print(f"Error processing ticket {idx}: {e}")
            continue

    mlflow.set_experiment("feature_requests_prioritization_crew_inline")
    with DominoRun(ai_system_config_path=CONFIG_PATH) as run:
        # Run synchronously for now - you can make this async if needed
        results = [kickoff(t) for t in tickets]
    df_out = pd.DataFrame(
        [
            {
                "final_score": r.final_score.final_score,
                "alignment_rationale": r.final_score.alignment_rationale,
                "effort_rationale": r.final_score.effort_rationale,
                "ticket_id": r.ticket_record.ticket_id
            }
            for r in results
        ]
    )

    df_merged = df_out.merge(df[['description', 'ticket_id']], how="inner", on='ticket_id')
    df_merged.to_csv(output_csv, index=False)