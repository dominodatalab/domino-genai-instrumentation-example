# src/guide_creator_flow/crews/content_crew/content_crew.py
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from pydantic import BaseModel
from typing import List

class AgentScores(BaseModel):
    alignment_score: int
    alignment_rationale: str
    effort_score: int
    effort_rationale: str

@CrewBase
class ProdEngCrew():
    """Product Engineering crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def product_strategist(self) -> Agent:
        return Agent(
            config=self.agents_config['product_strategist'], # type: ignore[index]
            verbose=True
        )

    @agent
    def product_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['product_engineer'], # type: ignore[index]
            verbose=True
        )

    @task
    def product_strategy_task(self) -> Task:
        return Task(
            config=self.tasks_config['product_strategy_task'] # type: ignore[index]
        )

    @task
    def product_engineer_task(self) -> Task:
        return Task(
            config=self.tasks_config['product_engineer_task'] # type: ignore[index]
        )

    @task
    def score_synthesizer_task(self) -> Task:
        return Task(
            config=self.tasks_config['score_synthesizer_task'], # type: ignore[index]
            output_pydantic=AgentScores
        )

    @crew
    def crew(self) -> Crew:
        """Creates the product engineering crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )