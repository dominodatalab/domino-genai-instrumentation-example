# Feature Request Prioritization AI System

This is an AI-powered system that prioritizes feature requests using multiple specialized agents to evaluate reach, impact, strategic alignment, and implementation effort.

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

## 📈 Instrumentation

This system demonstrates comprehensive AI system instrumentation using Domino's tracing and evaluation capabilities. See the `domino.agents` sdk [documentation](https://github.com/dominodatalab/python-domino/blob/release-2.0.0/README.md#dominoagentstracing) for more information

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

The `DominoRun` context manager provides MLflow experiment tracking. See [documentation](https://github.com/dominodatalab/python-domino/blob/release-2.0.0/README.md#dominoagentslogging) for more information:

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

Inline evaluation provides real-time quality assessment during execution. See `judge_response` in `agents_oai_inline_eval.py`.

**Advantages:**
- ✅ **Real-time feedback**: Immediate quality assessment
- ✅ **Zero overhead**: No separate evaluation step required
- ✅ **Automatic scaling**: Evaluates every trace without manual intervention
- ✅ **Immediate insights**: Quality metrics available instantly

### 🔄 Adhoc Evaluation  

**File: `agents_oai_simple_adhoc_eval.py`**

Adhoc evaluation performs batch assessment after execution. See the usage of `log_evaluation` in `agents_oai_simple_adhoc_eval.py`.

**Use Cases:**
- 🔧 **Custom evaluation logic**: Complex multi-step evaluations
- 📊 **Batch processing**: Evaluate large datasets efficiently  
- 🎛️ **Evaluation control**: Fine-grained control over when/how evaluation occurs
- 🔍 **Trace analysis**: Access to complete trace data for complex evaluations

### 📋 Evaluation Comparison

| Aspect | Inline Evaluation | Adhoc Evaluation |
|--------|------------------|------------------|
| **Timing** | During dev mode evaluation | After execution |
| **Overhead** | Uses application resources | Separate process |
| **Scalability** | Automatic | Manual batching required |
| **Use Case** | Real-time quality monitoring during development | Evaluate production system |

### 🎯 Best Practices

1. **Use inline evaluation** for:
   - Develpment mode
   - Automatic evaluation
   - Simple assessments

2. **Use adhoc evaluation** for:
   - Complex multi-dimensional assessments
   - Batch processing of historical data
   - Custom evaluation workflows
   - Evaluating production traces

3. **Disable autologging** during evaluation:
   ```python
   mlflow.openai.autolog(disable=True)  # Prevents evaluation traces
   ```

4. **Trace search and analysis**:
   ```python
   # find dev run traces
   traces = search_traces(run_id=run.info.run_id)
   # Analyze trace patterns, performance, quality metrics
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

## Input Data Format

### feature_requests.csv
```csv
ticket_id,description,customers_requesting,customer_priority
1,"Add support for multi-model deployment","[""Alpha Financial"", ""Beta Health""]",P1
```

### customers.csv
```csv
Company,Industry,ARR
Alpha Financial,Financial Services,1500000
```

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
