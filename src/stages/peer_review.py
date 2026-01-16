"""
Stage 2: Peer Review
Each Solver evaluates the other two solutions with structured feedback.
"""

import asyncio
from typing import List, Dict, Tuple
from src.llm_clients.base_client import BaseLLMClient
from src.models.schemas import (
    Problem,
    Solution,
    PeerReviewResult,
    RoleAssignmentResult
)
from src.stages.solver import Solver


class PeerReview:
    """Handles Stage 2: Peer review of solutions."""
    
    REVIEW_SYSTEM_PROMPT = """You are a critical reviewer in a multi-LLM debate system. Your task is to thoroughly evaluate another solver's solution.

Be constructive but rigorous:
1. Identify both strengths and weaknesses
2. Point out any errors with specific locations
3. Suggest concrete improvements
4. Assess whether you agree with the final answer
5. If you disagree, provide your alternative answer

Your review will help the solver improve their solution."""

    REVIEW_PROMPT_TEMPLATE = """Review the following solution to this problem.

PROBLEM:
{problem_text}

CATEGORY: {category}

SOLUTION TO REVIEW:
{solution_text}

Provide your review in the following JSON format:
{{
    "reviewer_id": "{reviewer_id}",
    "reviewer_model": "{reviewer_model}",
    "solution_id": "{solution_id}",
    "evaluation": {{
        "strengths": ["List of positive aspects"],
        "weaknesses": ["List of areas needing improvement"],
        "errors": [
            {{
                "location": "Where the error occurs (e.g., 'Step 3')",
                "error_type": "Type (logical_error, calculation_error, assumption_error, etc.)",
                "description": "What the error is",
                "severity": "minor|moderate|critical"
            }}
        ],
        "suggested_changes": ["List of recommended improvements"]
    }},
    "overall_assessment": "excellent|good|promising_but_flawed|needs_major_revision|fundamentally_wrong",
    "agreement_with_answer": true or false,
    "alternative_answer": "Your answer if you disagree (null if you agree)"
}}

Be thorough and specific in your critique.
Respond with ONLY the JSON object."""

    def __init__(self, clients: Dict[str, BaseLLMClient]):
        """
        Initialize the Peer Review stage.
        
        Args:
            clients: Dictionary mapping model names to their clients
        """
        self.clients = clients
        self.solver_formatter = Solver(clients)
    
    async def review_solution(
        self,
        reviewer_client: BaseLLMClient,
        reviewer_id: str,
        reviewer_model: str,
        solution: Solution,
        problem: Problem
    ) -> PeerReviewResult:
        """
        Generate a peer review for a single solution.
        
        Args:
            reviewer_client: The reviewing LLM client
            reviewer_id: ID of the reviewer
            reviewer_model: Name of the reviewer model
            solution: The solution to review
            problem: The original problem
            
        Returns:
            PeerReviewResult with detailed feedback
        """
        solution_text = self.solver_formatter.format_solution_for_review(solution)
        
        prompt = self.REVIEW_PROMPT_TEMPLATE.format(
            problem_text=problem.problem_text,
            category=problem.category.value,
            solution_text=solution_text,
            reviewer_id=reviewer_id,
            reviewer_model=reviewer_model,
            solution_id=solution.solver_id
        )
        
        try:
            result = await reviewer_client.generate_structured_with_retry(
                prompt=prompt,
                schema=PeerReviewResult,
                system_prompt=self.REVIEW_SYSTEM_PROMPT,
                temperature=0.5
            )
            # Ensure IDs are set correctly
            result.reviewer_id = reviewer_id
            result.reviewer_model = reviewer_model
            result.solution_id = solution.solver_id
            return result
        except Exception as e:
            # Log the error for debugging
            import traceback
            print(f"[ERROR] Peer review failed for {reviewer_id} ({reviewer_model}) reviewing {solution.solver_id}: {str(e)}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            # Return minimal review on error
            from src.models.schemas import Evaluation, OverallAssessment
            return PeerReviewResult(
                reviewer_id=reviewer_id,
                reviewer_model=reviewer_model,
                solution_id=solution.solver_id,
                evaluation=Evaluation(
                    strengths=["Unable to fully evaluate"],
                    weaknesses=[f"Review error: {str(e)}"],
                    errors=[],
                    suggested_changes=[]
                ),
                overall_assessment=OverallAssessment.PROMISING_BUT_FLAWED,
                agreement_with_answer=True,
                alternative_answer=None
            )
    
    async def stage_2_peer_review(
        self,
        problem: Problem,
        solutions: List[Solution],
        assignment: RoleAssignmentResult
    ) -> List[PeerReviewResult]:
        """
        Stage 2: Each Solver reviews the other two solutions.
        
        Args:
            problem: The original problem
            solutions: List of solutions from Stage 1
            assignment: Role assignment with solver info
            
        Returns:
            List of PeerReviewResults (6 total: 3 solvers × 2 reviews each)
        """
        # Build mapping of solver_id to solution and client
        solver_map = {
            "solver_1": {
                "solution": solutions[0],
                "client": self.clients[assignment.solver_1],
                "model": assignment.solver_1
            },
            "solver_2": {
                "solution": solutions[1],
                "client": self.clients[assignment.solver_2],
                "model": assignment.solver_2
            },
            "solver_3": {
                "solution": solutions[2],
                "client": self.clients[assignment.solver_3],
                "model": assignment.solver_3
            }
        }
        
        # Create review tasks: each solver reviews the other two
        review_tasks = []
        
        for reviewer_id, reviewer_info in solver_map.items():
            for target_id, target_info in solver_map.items():
                if reviewer_id != target_id:
                    review_tasks.append(
                        self.review_solution(
                            reviewer_client=reviewer_info["client"],
                            reviewer_id=reviewer_id,
                            reviewer_model=reviewer_info["model"],
                            solution=target_info["solution"],
                            problem=problem
                        )
                    )
        
        # Execute all reviews in parallel
        reviews = await asyncio.gather(*review_tasks)
        return list(reviews)
    
    def get_reviews_for_solver(
        self,
        solver_id: str,
        all_reviews: List[PeerReviewResult]
    ) -> List[PeerReviewResult]:
        """
        Get all reviews received by a specific solver.
        
        Args:
            solver_id: The solver's ID
            all_reviews: All peer reviews
            
        Returns:
            List of reviews for that solver's solution
        """
        return [r for r in all_reviews if r.solution_id == solver_id]
    
    def format_reviews_for_refinement(
        self,
        reviews: List[PeerReviewResult]
    ) -> str:
        """
        Format reviews for the refinement stage.
        
        Args:
            reviews: List of reviews for a solution
            
        Returns:
            Formatted string of all reviews
        """
        review_texts = []
        
        for i, review in enumerate(reviews, 1):
            errors_text = "\n".join([
                f"  - [{e.severity.value}] {e.location}: {e.description}"
                for e in review.evaluation.errors
            ]) if review.evaluation.errors else "  None identified"
            
            review_text = f"""
REVIEW {i} from {review.reviewer_id} ({review.reviewer_model}):
Overall Assessment: {review.overall_assessment.value}
Agreement with Answer: {"Yes" if review.agreement_with_answer else "No"}
{f"Alternative Answer: {review.alternative_answer}" if review.alternative_answer else ""}

Strengths:
{chr(10).join(f"  - {s}" for s in review.evaluation.strengths)}

Weaknesses:
{chr(10).join(f"  - {w}" for w in review.evaluation.weaknesses)}

Errors:
{errors_text}

Suggested Changes:
{chr(10).join(f"  - {c}" for c in review.evaluation.suggested_changes)}
"""
            review_texts.append(review_text)
        
        return "\n".join(review_texts)
