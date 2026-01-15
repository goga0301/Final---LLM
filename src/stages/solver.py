"""
Stage 1: Independent Solution Generation
Each Solver independently generates a complete solution with step-by-step reasoning.
"""

import asyncio
from typing import List, Dict
from src.llm_clients.base_client import BaseLLMClient
from src.models.schemas import (
    Problem,
    Solution,
    ReasoningStep,
    RoleAssignmentResult
)


class Solver:
    """Handles Stage 1: Independent solution generation."""
    
    SOLVER_SYSTEM_PROMPT = """You are a problem solver in a multi-LLM debate system. Your task is to solve the given problem independently with detailed step-by-step reasoning.

Guidelines:
1. Break down the problem into clear steps
2. Show all calculations and logical deductions
3. Consider edge cases and verify your reasoning
4. State any assumptions you make
5. Provide a clear final answer with confidence level

Your solution will be reviewed by other LLMs, so be thorough and precise."""

    SOLVER_PROMPT_TEMPLATE = """Solve the following problem with detailed step-by-step reasoning.

PROBLEM:
{problem_text}

CATEGORY: {category}

Provide your solution in the following JSON format:
{{
    "solver_id": "{solver_id}",
    "model_name": "{model_name}",
    "problem_id": "{problem_id}",
    "reasoning_steps": [
        {{
            "step_number": 1,
            "description": "Description of what this step does",
            "calculation": "Any calculations (optional)",
            "result": "Result of this step (optional)"
        }}
    ],
    "final_answer": "Your final answer",
    "confidence": <float between 0 and 1>,
    "assumptions": ["List of assumptions made (optional)"],
    "alternative_approaches": ["Other approaches considered (optional)"]
}}

Be thorough in your reasoning. Show your work clearly.
Respond with ONLY the JSON object."""

    def __init__(self, clients: Dict[str, BaseLLMClient]):
        """
        Initialize the Solver stage.
        
        Args:
            clients: Dictionary mapping model names to their clients
        """
        self.clients = clients
    
    async def generate_solution(
        self,
        client: BaseLLMClient,
        solver_id: str,
        model_name: str,
        problem: Problem
    ) -> Solution:
        """
        Generate a solution from a single Solver.
        
        Args:
            client: The LLM client
            solver_id: ID for this solver (e.g., "solver_1")
            model_name: Name of the model
            problem: The problem to solve
            
        Returns:
            Solution with reasoning and answer
        """
        prompt = self.SOLVER_PROMPT_TEMPLATE.format(
            problem_text=problem.problem_text,
            category=problem.category.value,
            solver_id=solver_id,
            model_name=model_name,
            problem_id=problem.id
        )
        
        try:
            result = await client.generate_structured_with_retry(
                prompt=prompt,
                schema=Solution,
                system_prompt=self.SOLVER_SYSTEM_PROMPT,
                temperature=0.7,  # Medium-high for creative problem solving
                max_tokens=4096
            )
            # Ensure IDs are set correctly
            result.solver_id = solver_id
            result.model_name = model_name
            result.problem_id = problem.id
            return result
        except Exception as e:
            # Log the error for debugging
            import traceback
            print(f"[ERROR] {solver_id} ({model_name}) failed: {str(e)}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            # Return error solution
            return Solution(
                solver_id=solver_id,
                model_name=model_name,
                problem_id=problem.id,
                reasoning_steps=[
                    ReasoningStep(
                        step_number=1,
                        description=f"Error generating solution: {str(e)}",
                        calculation=None,
                        result=None
                    )
                ],
                final_answer="ERROR",
                confidence=0.0,
                assumptions=None,
                alternative_approaches=None
            )
    
    async def stage_1_generate_solutions(
        self,
        problem: Problem,
        assignment: RoleAssignmentResult
    ) -> List[Solution]:
        """
        Stage 1: Generate independent solutions from all Solvers in parallel.
        
        Args:
            problem: The problem to solve
            assignment: Role assignment with solver assignments
            
        Returns:
            List of Solutions from each Solver
        """
        # Create tasks for parallel execution
        tasks = [
            self.generate_solution(
                client=self.clients[assignment.solver_1],
                solver_id="solver_1",
                model_name=assignment.solver_1,
                problem=problem
            ),
            self.generate_solution(
                client=self.clients[assignment.solver_2],
                solver_id="solver_2",
                model_name=assignment.solver_2,
                problem=problem
            ),
            self.generate_solution(
                client=self.clients[assignment.solver_3],
                solver_id="solver_3",
                model_name=assignment.solver_3,
                problem=problem
            )
        ]
        
        solutions = await asyncio.gather(*tasks)
        return list(solutions)
    
    def format_solution_for_review(self, solution: Solution) -> str:
        """
        Format a solution for peer review.
        
        Args:
            solution: The solution to format
            
        Returns:
            Formatted string representation
        """
        steps_text = "\n".join([
            f"Step {step.step_number}: {step.description}"
            + (f"\n  Calculation: {step.calculation}" if step.calculation else "")
            + (f"\n  Result: {step.result}" if step.result else "")
            for step in solution.reasoning_steps
        ])
        
        return f"""SOLVER: {solution.solver_id} ({solution.model_name})
CONFIDENCE: {solution.confidence:.2f}

REASONING:
{steps_text}

FINAL ANSWER: {solution.final_answer}

ASSUMPTIONS: {', '.join(solution.assumptions) if solution.assumptions else 'None stated'}
"""
