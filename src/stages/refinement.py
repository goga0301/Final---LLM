"""
Stage 3: Refinement Based on Feedback
Each Solver receives peer reviews and refines their solution.
"""

import asyncio
from typing import List, Dict
from src.llm_clients.base_client import BaseLLMClient
from src.models.schemas import (
    Problem,
    Solution,
    PeerReviewResult,
    RefinedSolution,
    CritiqueResponse,
    ReasoningStep,
    RoleAssignmentResult
)
from src.stages.solver import Solver
from src.stages.peer_review import PeerReview


class Refinement:
    """Handles Stage 3: Solution refinement based on peer feedback."""
    
    REFINEMENT_SYSTEM_PROMPT = """You are refining your solution based on peer feedback. Your task is to:

1. Address each critique explicitly - either accept and fix, or defend your reasoning
2. Be open to valid corrections but defend sound reasoning
3. Produce an improved solution incorporating valid feedback
4. Clearly explain what changes you made and why

Be thorough and maintain intellectual honesty."""

    REFINEMENT_PROMPT_TEMPLATE = """You previously submitted a solution that has been peer-reviewed. Refine your solution based on the feedback.

PROBLEM:
{problem_text}

YOUR ORIGINAL SOLUTION:
{original_solution}

PEER REVIEWS:
{peer_reviews}

Refine your solution and respond in the following JSON format:
{{
    "solver_id": "{solver_id}",
    "model_name": "{model_name}",
    "problem_id": "{problem_id}",
    "original_answer": "{original_answer}",
    "critique_responses": [
        {{
            "critique": "The specific critique being addressed",
            "response": "How you addressed or defended against this critique",
            "accepted": true or false,
            "changes_made": "Specific changes if accepted (null if rejected)"
        }}
    ],
    "refined_reasoning": [
        {{
            "step_number": 1,
            "description": "Updated step description",
            "calculation": "Any calculations (optional)",
            "result": "Result (optional)"
        }}
    ],
    "refined_answer": "Your refined final answer",
    "confidence": <float between 0 and 1>,
    "answer_changed": true or false
}}

Address the most significant critiques first. Be honest about whether critiques are valid.
Respond with ONLY the JSON object."""

    def __init__(self, clients: Dict[str, BaseLLMClient]):
        """
        Initialize the Refinement stage.
        
        Args:
            clients: Dictionary mapping model names to their clients
        """
        self.clients = clients
        self.solver_formatter = Solver(clients)
        self.review_formatter = PeerReview(clients)
    
    async def refine_solution(
        self,
        client: BaseLLMClient,
        solver_id: str,
        model_name: str,
        original_solution: Solution,
        reviews: List[PeerReviewResult],
        problem: Problem
    ) -> RefinedSolution:
        """
        Refine a solution based on peer feedback.
        
        Args:
            client: The Solver's LLM client
            solver_id: ID of the solver
            model_name: Name of the model
            original_solution: The original solution
            reviews: Peer reviews of the solution
            problem: The original problem
            
        Returns:
            RefinedSolution with responses to critiques
        """
        original_text = self.solver_formatter.format_solution_for_review(original_solution)
        reviews_text = self.review_formatter.format_reviews_for_refinement(reviews)
        
        prompt = self.REFINEMENT_PROMPT_TEMPLATE.format(
            problem_text=problem.problem_text,
            original_solution=original_text,
            peer_reviews=reviews_text,
            solver_id=solver_id,
            model_name=model_name,
            problem_id=problem.id,
            original_answer=original_solution.final_answer
        )
        
        try:
            result = await client.generate_structured_with_retry(
                prompt=prompt,
                schema=RefinedSolution,
                system_prompt=self.REFINEMENT_SYSTEM_PROMPT,
                temperature=0.5,  # Medium for balanced refinement
                max_tokens=4096
            )
            # Ensure IDs are set correctly
            result.solver_id = solver_id
            result.model_name = model_name
            result.problem_id = problem.id
            result.original_answer = original_solution.final_answer
            return result
        except Exception as e:
            # Log the error for debugging
            import traceback
            print(f"[ERROR] Refinement failed for {solver_id} ({model_name}): {str(e)}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            # Return unchanged solution on error
            return RefinedSolution(
                solver_id=solver_id,
                model_name=model_name,
                problem_id=problem.id,
                original_answer=original_solution.final_answer,
                critique_responses=[
                    CritiqueResponse(
                        critique="Unable to process reviews",
                        response=f"Error during refinement: {str(e)}",
                        accepted=False,
                        changes_made=None
                    )
                ],
                refined_reasoning=original_solution.reasoning_steps,
                refined_answer=original_solution.final_answer,
                confidence=original_solution.confidence * 0.8,
                answer_changed=False
            )
    
    async def stage_3_refine_solutions(
        self,
        problem: Problem,
        solutions: List[Solution],
        reviews: List[PeerReviewResult],
        assignment: RoleAssignmentResult
    ) -> List[RefinedSolution]:
        """
        Stage 3: All Solvers refine their solutions based on peer feedback.
        
        Args:
            problem: The original problem
            solutions: Original solutions from Stage 1
            reviews: All peer reviews from Stage 2
            assignment: Role assignment info
            
        Returns:
            List of RefinedSolutions
        """
        # Map solver_id to solution and reviews
        solver_data = {
            "solver_1": {
                "solution": solutions[0],
                "client": self.clients[assignment.solver_1],
                "model": assignment.solver_1,
                "reviews": [r for r in reviews if r.solution_id == "solver_1"]
            },
            "solver_2": {
                "solution": solutions[1],
                "client": self.clients[assignment.solver_2],
                "model": assignment.solver_2,
                "reviews": [r for r in reviews if r.solution_id == "solver_2"]
            },
            "solver_3": {
                "solution": solutions[2],
                "client": self.clients[assignment.solver_3],
                "model": assignment.solver_3,
                "reviews": [r for r in reviews if r.solution_id == "solver_3"]
            }
        }
        
        # Create refinement tasks
        tasks = [
            self.refine_solution(
                client=data["client"],
                solver_id=solver_id,
                model_name=data["model"],
                original_solution=data["solution"],
                reviews=data["reviews"],
                problem=problem
            )
            for solver_id, data in solver_data.items()
        ]
        
        # Execute all refinements in parallel
        refined_solutions = await asyncio.gather(*tasks)
        return list(refined_solutions)
    
    def format_refined_solution_for_judgment(
        self,
        original: Solution,
        refined: RefinedSolution
    ) -> str:
        """
        Format a refined solution for the final judgment.
        
        Args:
            original: The original solution
            refined: The refined solution
            
        Returns:
            Formatted string for judge review
        """
        # Format reasoning steps
        steps_text = "\n".join([
            f"Step {step.step_number}: {step.description}"
            + (f"\n  Calculation: {step.calculation}" if step.calculation else "")
            + (f"\n  Result: {step.result}" if step.result else "")
            for step in refined.refined_reasoning
        ])
        
        # Format critique responses
        responses_text = "\n".join([
            f"- Critique: {cr.critique}\n  Response: {cr.response}\n  Accepted: {cr.accepted}"
            for cr in refined.critique_responses[:5]  # Limit to top 5
        ])
        
        return f"""SOLVER: {refined.solver_id} ({refined.model_name})
ORIGINAL ANSWER: {refined.original_answer}
REFINED ANSWER: {refined.refined_answer}
ANSWER CHANGED: {"Yes" if refined.answer_changed else "No"}
CONFIDENCE: {refined.confidence:.2f}

REFINED REASONING:
{steps_text}

KEY CRITIQUE RESPONSES:
{responses_text}
"""
