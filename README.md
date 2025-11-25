# Feature Request Prioritization AI System

An AI-powered system that prioritizes feature requests using multiple specialized agents to evaluate reach, impact, strategic alignment, and implementation effort.

## 🎯 Overview

This system helps product teams make data-driven decisions about feature prioritization by:
- **Quantifying business reach** based on customer Annual Recurring Revenue (ARR)
- **Assessing impact** from customer-assigned priority levels (P0-P3)
- **Evaluating strategic alignment** with enterprise goals using AI agents
- **Estimating implementation effort** through senior engineer simulation
- **Calculating final scores** using a weighted formula: `(reach × impact × alignment) / effort`

## 🏗️ Architecture

The system uses a multi-agent architecture with the following components:

### Core Agents
- **TicketPrioritizationAgent**: Main orchestrator that coordinates the entire prioritization workflow
- **EffortAgent**: Senior engineer persona that evaluates implementation complexity (1-5 scale)  
- **AlignmentAgent**: Product strategist persona that assesses strategic fit (1-5 scale)

### Function Tools
- **`reach_score`**: Calculates business reach based on requesting customers' ARR
- **`impact_score`**: Converts priority levels (P0-P3) to numerical impact scores
- **`calculate_final_score`**: Applies the prioritization formula

### Evaluation System
- **AI Judge**: Automated quality assessment of effort estimations using GPT models
- **Inline Evaluation**: Real-time assessment during agent execution
- **Adhoc Evaluation**: Post-execution batch evaluation of traces

## 📋 Requirements

```bash
pip install -r modified_requirements.txt
```

**Dependencies:**
- `pydantic`: Data validation and serialization
- `mlflow`: Experiment tracking and model management
- `pandas`: Data manipulation and analysis
- `python-dotenv`: Environment variable management
- `openai`: OpenAI API integration
- `openai-agents`: OpenAI Agents SDK
- `PyYAML`: Configuration file parsing

## ⚙️ Configuration

All system configuration is centralized in `config.yaml`:

```yaml
# Model configurations
models:
  default: "gpt-4o-mini"     # Default model for environment fallback
  agent: "gpt-4o-mini"       # Model for all AI agents
  judge: "gpt-5"             # Model for evaluation/judging

# Agent instructions/prompts  
instructions:
  effort_agent: "You are a senior engineer..."
  alignment_agent: "You are a product strategist..."
  ticket_agent: |
    You receive a ticket dict with ticket_id, description, customers_requesting, priority.
    1. Call reach_score(customers_requesting) → reach.
    # ... full workflow instructions
  judge_prompt: |
    You are an expert tech lead
    Rate the following effort rationale on a scale of 1-5...
    # ... evaluation criteria
```

## 📊 Data Requirements

### Feature Requests (`feature_requests.csv`)
```csv
ticket_id,description,customers_requesting,customer_priority
1,"Add SSO integration","[\"Company A\", \"Company B\"]","P1"
```

### Customer Data (`customers.csv`)  
```csv
Company,ARR
Company A,2500000
Company B,1200000
```

## 🚀 Usage

### Basic Execution

```bash
# Run with inline evaluation
python agents_oai_inline_eval.py

# Run with adhoc evaluation  
python agents_oai_simple_adhoc_eval.py
```

### Programmatic Usage

```python
import asyncio
from agents_oai_inline_eval import prioritize_features

# Run prioritization
await prioritize_features(
    input_csv='feature_requests.csv',
    output_csv='scored_tickets.csv', 
    customers_csv='customers.csv'
)
```

## 📈 Instrumentation

This system demonstrates comprehensive AI system instrumentation using Domino's tracing and evaluation capabilities.

### 🔍 Tracing with @add_tracing

The `@add_tracing` decorator automatically captures execution traces for AI operations:

```python
@add_tracing(name="prioritize_ticket", autolog_frameworks=["openai"])
async def prioritize_ticket(ticket: TicketRecord) -> ScoredTicket:
    # Your AI workflow here
    pass
```

**Key Features:**
- **Automatic trace capture**: Records inputs, outputs, and execution metadata
- **Framework integration**: `autolog_frameworks=["openai"]` automatically logs OpenAI API calls
- **Hierarchical tracing**: Captures nested agent calls and tool usage
- **Performance monitoring**: Tracks latency and resource usage

### 📊 Experiment Management with DominoRun

The `DominoRun` context manager provides MLflow experiment tracking:

```python
from domino.agents.logging import DominoRun

with DominoRun(agent_config_path=CONFIG_PATH) as run:
    results = await asyncio.gather(*[prioritize_ticket(t) for t in tickets])
    # All traces automatically associated with this run
```

**Benefits:**
- **Centralized logging**: All traces grouped under a single experiment run
- **Config integration**: Links traces to system configuration
- **Metadata capture**: Automatically records environment and system details
- **Run comparison**: Easy comparison between different experiment runs

### ⚡ Inline Evaluation

**File: `agents_oai_inline_eval.py`**

Inline evaluation provides real-time quality assessment during execution:

```python
def judge_response(inputs, output):
    """Evaluator function called automatically for each trace"""
    request_description = inputs['ticket']['description']
    effort_rationale = output['final_score']['effort_rationale']
    
    # AI-powered evaluation
    rating = evaluate_with_ai_judge(effort_rationale, request_description)
    return {"eng_effort_accuracy": rating}

@add_tracing(
    name="prioritize_ticket", 
    autolog_frameworks=["openai"],
    evaluator=judge_response  # Automatic evaluation
)
async def prioritize_ticket(ticket: TicketRecord) -> TraceScore:
    # Function executes normally, evaluation happens automatically
    pass
```

**Advantages:**
- ✅ **Real-time feedback**: Immediate quality assessment
- ✅ **Zero overhead**: No separate evaluation step required
- ✅ **Automatic scaling**: Evaluates every trace without manual intervention
- ✅ **Immediate insights**: Quality metrics available instantly

### 🔄 Adhoc Evaluation  

**File: `agents_oai_simple_adhoc_eval.py`**

Adhoc evaluation performs batch assessment after execution:

```python
with DominoRun(agent_config_path=CONFIG_PATH) as run:
    # Execute main workflow
    results = await asyncio.gather(*[prioritize_ticket(t) for t in tickets])
    
    # Disable autologging to avoid logging evaluation traces
    mlflow.openai.autolog(disable=True)
    
    # Batch evaluation of all traces
    traces = search_traces(run_id=run.info.run_id)
    for trace in traces.data:
        effort_rationale = trace.spans[0].outputs['final_score']['effort_rationale']
        request_description = trace.spans[0].inputs['ticket']['description']
        
        score = judge_response(effort_rationale, request_description)
        log_evaluation(trace_id=trace.id, name="eng_effort_accuracy", value=score)
```

**Use Cases:**
- 🔧 **Custom evaluation logic**: Complex multi-step evaluations
- 📊 **Batch processing**: Evaluate large datasets efficiently  
- 🎛️ **Evaluation control**: Fine-grained control over when/how evaluation occurs
- 🔍 **Trace analysis**: Access to complete trace data for complex evaluations

### 📋 Evaluation Comparison

| Aspect | Inline Evaluation | Adhoc Evaluation |
|--------|------------------|------------------|
| **Timing** | During execution | After execution |
| **Overhead** | Minimal | Separate process |
| **Flexibility** | Limited to simple functions | Full programmatic control |
| **Scalability** | Automatic | Manual batching required |
| **Use Case** | Real-time quality monitoring | Complex analysis workflows |

### 🎯 Best Practices

1. **Use inline evaluation** for:
   - Simple quality metrics
   - Real-time monitoring  
   - Production systems requiring immediate feedback

2. **Use adhoc evaluation** for:
   - Complex multi-dimensional assessments
   - Batch processing of historical data
   - Custom evaluation workflows

3. **Disable autologging** during evaluation:
   ```python
   mlflow.openai.autolog(disable=True)  # Prevents evaluation traces
   ```

4. **Trace search and analysis**:
   ```python
   traces = search_traces(run_id=run.info.run_id)
   # Analyze trace patterns, performance, quality metrics
   ```

## 📁 File Structure

```
├── README.md                           # This documentation
├── config.yaml                         # Centralized configuration
├── modified_requirements.txt           # Python dependencies
├── agents_oai_inline_eval.py          # Inline evaluation implementation
├── agents_oai_simple_adhoc_eval.py    # Adhoc evaluation implementation  
├── feature_requests.csv               # Input: feature requests to prioritize
├── customers.csv                       # Input: customer ARR data
└── scored_tickets.csv                  # Output: prioritized features with scores
```

### File Descriptions

- **`agents_oai_inline_eval.py`**: Demonstrates real-time evaluation during agent execution using the `evaluator` parameter in `@add_tracing`
- **`agents_oai_simple_adhoc_eval.py`**: Shows post-execution batch evaluation by searching traces and manually logging evaluation results
- **`config.yaml`**: Centralizes all model configurations, agent instructions, and system settings
- **Input CSVs**: Contains feature requests and customer data for prioritization
- **Output CSV**: Generated prioritized feature list with scores and rationales

## 🔍 Output Format

The system generates a CSV with the following columns:

| Column | Description |
|--------|-------------|
| `ticket_id` | Unique identifier for the feature request |
| `final_score` | Calculated priority score (higher = more important) |
| `alignment_rationale` | AI-generated explanation of strategic alignment |  
| `effort_rationale` | AI-generated explanation of implementation effort |
| `description` | Original feature request description |
| `trace_id` | (inline version) Associated trace ID for debugging |

## 🎛️ Scoring Logic

**Final Score Formula:**
```
final_score = (reach × impact × alignment) / effort
```

**Component Ranges:**
- **Reach**: 1-5 (based on customer ARR: $1M increments) 
- **Impact**: 1-4 (P0=1, P1=2, P2=3, P3=4, lower = higher priority)
- **Alignment**: 1-5 (AI-evaluated strategic fit)
- **Effort**: 1-5 (AI-evaluated implementation complexity)

## 🚀 Getting Started

1. **Install dependencies**:
   ```bash
   pip install -r modified_requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

3. **Prepare your data**:
   - Update `feature_requests.csv` with your feature requests
   - Update `customers.csv` with your customer ARR data

4. **Run prioritization**:
   ```bash
   # For inline evaluation
   python agents_oai_inline_eval.py
   
   # For adhoc evaluation  
   python agents_oai_simple_adhoc_eval.py
   ```

5. **Review results**:
   - Check `scored_tickets.csv` for prioritized features
   - Review MLflow experiments for detailed traces and evaluations

## 🤝 Contributing

This system demonstrates best practices for:
- Multi-agent AI system design
- Comprehensive instrumentation and evaluation
- Configuration-driven development
- Production-ready AI workflows

Feel free to extend the system with additional agents, evaluation metrics, or scoring algorithms!
