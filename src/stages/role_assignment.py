"""
Stage 0 & 0.5: Role Assignment
Each LLM self-assesses their suitability for roles, then an algorithm assigns final roles.
"""

import asyncio
from typing import List, Dict, Tuple
from src.llm_clients.base_client import BaseLLMClient
from src.models.schemas import (
    Problem,
    RolePreference,
    RoleConfidence,
    RoleAssignmentResult
)


class RoleAssignment:
    """Handles role assignment for the debate system."""
    
    ROLE_ASSESSMENT_SYSTEM_PROMPT = """You are participating in a multi-LLM debate system where you will be assigned one of two roles:
1. SOLVER: Independently solve the problem with detailed step-by-step reasoning
2. JUDGE: Evaluate all solutions and select the best one

Assess your own capabilities for each role based on the problem type and your strengths.
Be honest about your confidence levels - overconfidence leads to poor assignments."""

    ROLE_ASSESSMENT_PROMPT_TEMPLATE = """Given the following problem, assess which role you would be best suited for.

PROBLEM:
{problem_text}

PROBLEM CATEGORY: {category}

Please provide your self-assessment in the following JSON format:
{{
    "model_name": "{model_name}",
    "role_preferences": ["Solver", "Judge"] or ["Judge", "Solver"],
    "confidence_by_role": {{
        "solver": <float between 0 and 1>,
        "judge": <float between 0 and 1>
    }},
    "reasoning": "<Your reasoning for the role preferences>",
    "problem_analysis": "<Brief analysis of what makes this problem challenging>"
}}

Consider:
- For Solver role: Your ability to perform step-by-step reasoning, mathematical calculations, logical deductions
- For Judge role: Your ability to critically evaluate multiple solutions, identify errors, and compare approaches

Respond with ONLY the JSON object."""

    def __init__(self, clients: Dict[str, BaseLLMClient]):
        """
        Initialize role assignment.
        
        Args:
            clients: Dictionary mapping model names to their clients
        """
        self.clients = clients
        self.model_names = list(clients.keys())
    
    async def get_role_preference(
        self,
        client: BaseLLMClient,
        model_name: str,
        problem: Problem
    ) -> RolePreference:
        """
        Get role preference from a single LLM.
        
        Args:
            client: The LLM client
            model_name: Name of the model
            problem: The problem to assess
            
        Returns:
            RolePreference with the model's self-assessment
        """
        prompt = self.ROLE_ASSESSMENT_PROMPT_TEMPLATE.format(
            problem_text=problem.problem_text,
            category=problem.category.value,
            model_name=model_name
        )
        
        try:
            result = await client.generate_structured_with_retry(
                prompt=prompt,
                schema=RolePreference,
                system_prompt=self.ROLE_ASSESSMENT_SYSTEM_PROMPT,
                temperature=0.3  # Low temperature for consistent role selection
            )
            # Ensure model_name is set correctly
            result.model_name = model_name
            return result
        except Exception as e:
            # Log the error for debugging
            import traceback
            print(f"[ERROR] Role preference failed for {model_name}: {str(e)}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            # Return default preference on failure
            return RolePreference(
                model_name=model_name,
                role_preferences=["Solver", "Judge"],
                confidence_by_role=RoleConfidence(solver=0.5, judge=0.5),
                reasoning=f"Default assignment due to error: {str(e)}",
                problem_analysis="Unable to analyze"
            )
    
    async def stage_0_get_all_preferences(
        self,
        problem: Problem
    ) -> List[RolePreference]:
        """
        Stage 0: Get role preferences from all LLMs in parallel.
        
        Args:
            problem: The problem to assess
            
        Returns:
            List of RolePreference from each model
        """
        tasks = [
            self.get_role_preference(client, name, problem)
            for name, client in self.clients.items()
        ]
        
        preferences = await asyncio.gather(*tasks)
        return list(preferences)
    
    def stage_05_assign_roles(
        self,
        preferences: List[RolePreference]
    ) -> RoleAssignmentResult:
        """
        Stage 0.5: Algorithmic role assignment based on preferences.
        
        Algorithm:
        1. Assign Judge to the model with highest judge confidence
        2. Assign remaining 3 models as Solvers based on solver confidence
        
        Args:
            preferences: List of role preferences from all models
            
        Returns:
            RoleAssignmentResult with final assignments
        """
        if len(preferences) != 4:
            raise ValueError(f"Expected 4 preferences, got {len(preferences)}")
        
        # Sort by judge confidence to find the best judge
        judge_sorted = sorted(
            preferences,
            key=lambda p: p.confidence_by_role.judge,
            reverse=True
        )
        
        # Assign highest judge confidence as Judge
        judge_model = judge_sorted[0].model_name
        
        # Remaining models are Solvers, sorted by solver confidence
        solver_candidates = [p for p in preferences if p.model_name != judge_model]
        solver_sorted = sorted(
            solver_candidates,
            key=lambda p: p.confidence_by_role.solver,
            reverse=True
        )
        
        # Assign Solvers
        solver_1 = solver_sorted[0].model_name
        solver_2 = solver_sorted[1].model_name
        solver_3 = solver_sorted[2].model_name
        
        # Build reasoning
        reasoning_parts = [
            f"Judge assigned to {judge_model} (highest judge confidence: {judge_sorted[0].confidence_by_role.judge:.2f})",
            f"Solver 1: {solver_1} (solver confidence: {solver_sorted[0].confidence_by_role.solver:.2f})",
            f"Solver 2: {solver_2} (solver confidence: {solver_sorted[1].confidence_by_role.solver:.2f})",
            f"Solver 3: {solver_3} (solver confidence: {solver_sorted[2].confidence_by_role.solver:.2f})"
        ]
        
        return RoleAssignmentResult(
            solver_1=solver_1,
            solver_2=solver_2,
            solver_3=solver_3,
            judge=judge_model,
            assignment_reasoning="; ".join(reasoning_parts)
        )
    
    async def assign_roles(
        self,
        problem: Problem
    ) -> Tuple[List[RolePreference], RoleAssignmentResult]:
        """
        Complete role assignment workflow (Stage 0 + 0.5).
        
        Args:
            problem: The problem to solve
            
        Returns:
            Tuple of (preferences, assignment)
        """
        # Stage 0: Get preferences
        preferences = await self.stage_0_get_all_preferences(problem)
        
        # Stage 0.5: Assign roles
        assignment = self.stage_05_assign_roles(preferences)
        
        return preferences, assignment
    
    def get_solver_clients(
        self,
        assignment: RoleAssignmentResult
    ) -> Dict[str, BaseLLMClient]:
        """
        Get the clients assigned as Solvers.
        
        Args:
            assignment: Role assignment result
            
        Returns:
            Dictionary mapping solver_id to client
        """
        return {
            "solver_1": self.clients[assignment.solver_1],
            "solver_2": self.clients[assignment.solver_2],
            "solver_3": self.clients[assignment.solver_3]
        }
    
    def get_judge_client(
        self,
        assignment: RoleAssignmentResult
    ) -> BaseLLMClient:
        """
        Get the client assigned as Judge.
        
        Args:
            assignment: Role assignment result
            
        Returns:
            The judge client
        """
        return self.clients[assignment.judge]
