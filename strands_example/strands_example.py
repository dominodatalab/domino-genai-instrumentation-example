from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools import calculator
import mlflow
import os
from domino.aisystems.tracing import add_tracing
from domino.aisystems.logging import DominoRun
from dotenv import load_dotenv


mlflow.set_experiment("Strand Agent")

# Load environment variables
load_dotenv(override=True)
API_KEY=os.getenv("OPENAI_API_KEY")

model = OpenAIModel(
    client_args={"api_key": API_KEY},
    # **model_config
    model_id="gpt-4o",
    params={
        "max_tokens": 2000,
        "temperature": 0.7,
    },
)

agent = Agent(model=model, tools=[calculator])

@add_tracing(name='strands_agent', autolog_frameworks=["strands"])
def call_agent(user_prompt):
    return agent(user_prompt)

if __name__== '__main__':
    user_query = "What is 2+2"
    with DominoRun(ai_system_config_path="config.yaml") as run:
        response = call_agent(user_query)
        print(response)