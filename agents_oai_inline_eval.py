import os
import json
import asyncio
import pandas as pd
import logging
import yaml
from typing import List, Literal, Dict, Any
from agents import Agent, Runner, function_tool, TResponseInputItem
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from domino.agents.tracing import add_tracing, search_traces
from domino.agents.logging import DominoRun, log_evaluation
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import mlflow
import time
import random
from openai import OpenAI

# Load environment variables
dotenv_path = os.getenv('DOTENV_PATH') or None
load_dotenv(override=True)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')

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

config = load_config(CONFIG_PATH)

# ── Models ─────────────────────────────────────────────────────────────────────
class TicketRecord(BaseModel):
    ticket_id: int
    description: str
    customers_requesting: List[str]
    priority: Literal["P0", "P1", "P2", "P3"]

class EffortResult(BaseModel):
    ticket_id: int
    score: int = Field(ge=1, le=5)
    rationale: str

class AlignmentResult(BaseModel):
    ticket_id: int
    score: int = Field(ge=1, le=5)
    rationale: str

class FinalScore(BaseModel):
    final_score: float
    alignment_rationale: str
    effort_rationale: str

class TraceScore(BaseModel):
    ticket_id: int
    final_score: FinalScore
    trace_id: str

# ── Helpers ─────────────────────────────────────────────────────────────────────

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

# ── Function Tools ──────────────────────────────────────────────────────────────
@function_tool(name_override="reach_score")
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

@function_tool(name_override="impact_score")
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

@function_tool(name_override="calculate_final_score")
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

# ── AI Sub-Agents ──────────────────────────────────────────────────────────────
MODEL = os.getenv('MODEL', config['models']['default'])
ALT_MODEL = config['models']['agent']

effort_agent = Agent(
    name="EffortAgent",
    instructions=config['instructions']['effort_agent'],
    model=ALT_MODEL,
    output_type=EffortResult,
)

alignment_agent = Agent(
    name="AlignmentAgent",
    instructions=config['instructions']['alignment_agent'],
    model=ALT_MODEL,
    output_type=AlignmentResult,
)

# ── Orchestrator Agent with SDK-native Handoffs ────────────────────────────────
instructions = prompt_with_handoff_instructions(config['instructions']['ticket_agent'])
ticket_agent = Agent(
    name="TicketPrioritizationAgent",
    instructions=instructions,
    model=ALT_MODEL,
    tools=[reach_score_fn, impact_score_fn, final_score_fn,
           effort_agent.as_tool(tool_name="evaluate_effort", tool_description="Evaluate implementation effort"),
           alignment_agent.as_tool(tool_name="evaluate_alignment", tool_description="Evaluate strategic alignment")],
    output_type=FinalScore,
)

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
    request_description = inputs['ticket']['description']
    effort_rationale = output['final_score']['effort_rationale']
    client = OpenAI()
    judge_prompt = config['instructions']['judge_prompt'].format(
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

@add_tracing(name="prioritize_ticket", autolog_frameworks=["openai"], evaluator=judge_response)
async def prioritize_ticket(ticket: TicketRecord) -> TraceScore:

    try:
        inputs: list[TResponseInputItem] = [{"content": f"This is the ticket: {ticket.model_dump()}", "role": "user"}]
        run_result = await Runner.run(ticket_agent, inputs)

        # Extract the typed output from the RunResult
        final: FinalScore = run_result.final_output  # should be FinalScore instance
        trace_score: TraceScore = TraceScore(
            ticket_id=ticket.ticket_id,
            final_score=final,
            trace_id=mlflow.get_active_trace_id()
        )
        return trace_score

    except Exception as e:
        final = FinalScore(
            final_score=0.0,
            alignment_rationale=str(e),
            effort_rationale=str(e),
        )
        return TraceScore(
            ticket_id=0,
            final_score=final,
            trace_id='0'
        )

# ── Main ────────────────────────────────────────────────────────────────────────
async def prioritize_features(input_csv: str, output_csv: str, customers_csv: str):

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
            raise


    mlflow.set_experiment("feature_requests_prioritization_oai_inline")
    with DominoRun(agent_config_path=CONFIG_PATH) as run:
        results = await asyncio.gather(*[prioritize_ticket(t) for t in tickets])

    df_out = pd.DataFrame(
        [
            {
                "final_score": r.final_score.final_score,
                "alignment_rationale": r.final_score.alignment_rationale,
                "effort_rationale": r.final_score.effort_rationale,
                "ticket_id": r.ticket_id,
                "trace_id": r.trace_id,
            }
            for r in results
        ]
    )

    df_merged = df_out.merge(df[['description', 'ticket_id']], how="inner", on='ticket_id')
    df_merged.to_csv(output_csv, index=False)

if __name__ == '__main__':
    base = os.path.dirname(__file__)
    INPUT_TICKETS = os.path.join(base, 'feature_requests.csv')
    SCORED_TICKETS = os.path.join(base, 'scored_tickets.csv')
    asyncio.run(prioritize_features(
            input_csv=INPUT_TICKETS,
            output_csv=SCORED_TICKETS,
            customers_csv=os.path.join(base, 'customers.csv'),
        )
    )
