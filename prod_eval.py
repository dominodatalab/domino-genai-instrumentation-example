from domino.aisystems.tracing import search_ai_system_traces
from domino.aisystems.logging import log_evaluation
import random

def user_eval():
    return random.randint(1, 5)

def hallucination_bool(p=0.8):
    return True if random.random() < p else False

def log_dict_evals(trace_id, metrics_dict):
    for k, v in metrics_dict.items():
        log_evaluation(trace_id=trace_id, name=k, value=v)

AI_SYSTEM_ID = "68ffb11c4bac6e55f6649d18"
traces = search_ai_system_traces(ai_system_id=AI_SYSTEM_ID)
for trace in traces.data:
    effort_rationale = trace.spans[0].outputs['final_score']['effort_rationale']
    request_description = trace.spans[0].inputs['ticket']['description']
    user_score = user_eval()
    hallucination = hallucination_bool()
    log_dict_evals(trace_id=trace.id, metrics_dict={"user_score": user_score, "hallucination": hallucination})
