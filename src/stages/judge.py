"""
Stage 4: Final Judgment
The Judge LLM evaluates all solutions and selects the best answer.
"""

from typing import List, Dict
from src.llm_clients.base_client import BaseLLMClient
from src.models.schemas import (
    Problem,
    Solution,
    PeerReviewResult,
    RefinedSolution,
    JudgmentResult,
    SolutionComparison,
    RoleAssignmentResult
)
from src.stages.solver import Solver
from src.stages.refinement import Refinement


class Judge:
    """Handles Stage 4: Final judgment of all solutions."""
    
    JUDGE_SYSTEM_PROMPT = """You are the final judge in a multi-LLM debate system. Your task is to:

1. Carefully evaluate all three refined solutions
2. Compare their reasoning quality and correctness
3. Consider how well each solver addressed peer feedback
4. Select the best solution with clear justification

Be objective and thorough. The winning answer will be the final output of the system."""

    JUDGE_PROMPT_TEMPLATE = """You are the judge. Evaluate all solutions and select the winner.

PROBLEM:
{problem_text}

CATEGORY: {category}

=== SOLVER 1 ===
{solver_1_solution}

=== SOLVER 2 ===
{solver_2_solution}

=== SOLVER 3 ===
{solver_3_solution}

SUMMARY OF PEER REVIEWS:
{peer_review_summary}

Provide your judgment in the following JSON format:
{{
    "judge_model": "{judge_model}",
    "problem_id": "{problem_id}",
    "solution_comparisons": [
        {{
            "solver_id": "solver_1",
            "strengths": ["Key strengths"],
            "weaknesses": ["Key weaknesses"],
            "correctness_assessment": "Assessment of whether the answer is correct",
            "quality_score": <float between 0 and 1>
        }},
        {{
            "solver_id": "solver_2",
            "strengths": ["Key strengths"],
            "weaknesses": ["Key weaknesses"],
            "correctness_assessment": "Assessment",
            "quality_score": <float between 0 and 1>
        }},
        {{
            "solver_id": "solver_3",
            "strengths": ["Key strengths"],
            "weaknesses": ["Key weaknesses"],
            "correctness_assessment": "Assessment",
            "quality_score": <float between 0 and 1>
        }}
    ],
    "winner": "solver_1|solver_2|solver_3",
    "winning_answer": "The exact answer from the winning solution",
    "confidence": <float between 0 and 1>,
    "reasoning": "Detailed explanation of why this solution was selected",
    "consensus_exists": true or false (whether all solvers agreed on the answer)
}}

Evaluate carefully and select the best answer.
Respond with ONLY the JSON object."""

    def __init__(self, clients: Dict[str, BaseLLMClient]):
        """
        Initialize the Judge stage.
        
        Args:
            clients: Dictionary mapping model names to their clients
        """
        self.clients = clients
        self.solver_formatter = Solver(clients)
        self.refinement_formatter = Refinement(clients)
    
    def _summarize_peer_reviews(
        self,
        reviews: List[PeerReviewResult]
    ) -> str:
        """
        Create a summary of peer reviews for the judge.
        
        Args:
            reviews: All peer reviews
            
        Returns:
            Summary string
        """
        # Group reviews by target solution
        by_target = {}
        for review in reviews:
            if review.solution_id not in by_target:
                by_target[review.solution_id] = []
            by_target[review.solution_id].append(review)
        
        summaries = []
        for solver_id in sorted(by_target.keys()):
            solver_reviews = by_target[solver_id]
            
            # Count assessments
            assessments = [r.overall_assessment.value for r in solver_reviews]
            agreements = sum(1 for r in solver_reviews if r.agreement_with_answer)
            
            # Collect key critiques
            all_errors = []
            for r in solver_reviews:
                all_errors.extend([e.description for e in r.evaluation.errors[:2]])
            
            summary = f"{solver_id}: Assessments={assessments}, Agreements={agreements}/2"
            if all_errors:
                summary += f", Key issues: {all_errors[:3]}"
            summaries.append(summary)
        
        return "\n".join(summaries)
    
    async def stage_4_judge(
        self,
        problem: Problem,
        original_solutions: List[Solution],
        reviews: List[PeerReviewResult],
        refined_solutions: List[RefinedSolution],
        assignment: RoleAssignmentResult
    ) -> JudgmentResult:
        """
        Stage 4: Judge evaluates all solutions and selects the winner.
        
        Args:
            problem: The original problem
            original_solutions: Solutions from Stage 1
            reviews: Peer reviews from Stage 2
            refined_solutions: Refined solutions from Stage 3
            assignment: Role assignment info
            
        Returns:
            JudgmentResult with winner selection
        """
        # Get judge client
        judge_client = self.clients[assignment.judge]
        
        # Format solutions for judgment
        solution_texts = []
        for orig, refined in zip(original_solutions, refined_solutions):
            text = self.refinement_formatter.format_refined_solution_for_judgment(orig, refined)
            solution_texts.append(text)
        
        # Create peer review summary
        review_summary = self._summarize_peer_reviews(reviews)
        
        prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            problem_text=problem.problem_text,
            category=problem.category.value,
            solver_1_solution=solution_texts[0],
            solver_2_solution=solution_texts[1],
            solver_3_solution=solution_texts[2],
            peer_review_summary=review_summary,
            judge_model=assignment.judge,
            problem_id=problem.id
        )
        
        try:
            result = await judge_client.generate_structured_with_retry(
                prompt=prompt,
                schema=JudgmentResult,
                system_prompt=self.JUDGE_SYSTEM_PROMPT,
                temperature=0.3,  # Low for deterministic judgment
                max_tokens=4096
            )
            # Ensure values are set correctly
            result.judge_model = assignment.judge
            result.problem_id = problem.id
            return result
        except Exception as e:
            # Log the error for debugging
            import traceback
            print(f"[ERROR] Judgment failed for judge ({assignment.judge}): {str(e)}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            # Fallback: pick highest confidence solution
            best_solution = max(refined_solutions, key=lambda s: s.confidence)
            
            return JudgmentResult(
                judge_model=assignment.judge,
                problem_id=problem.id,
                solution_comparisons=[
                    SolutionComparison(
                        solver_id=s.solver_id,
                        strengths=["Unable to fully evaluate"],
                        weaknesses=[f"Judgment error: {str(e)}"],
                        correctness_assessment="Unknown",
                        quality_score=s.confidence
                    )
                    for s in refined_solutions
                ],
                winner=best_solution.solver_id,
                winning_answer=best_solution.refined_answer,
                confidence=0.5,
                reasoning=f"Fallback selection based on confidence due to error: {str(e)}",
                consensus_exists=len(set(s.refined_answer for s in refined_solutions)) == 1
            )
    
    def check_consensus(
        self,
        refined_solutions: List[RefinedSolution]
    ) -> bool:
        """
        Check if all solvers reached the same answer.
        
        Args:
            refined_solutions: All refined solutions
            
        Returns:
            True if all answers match
        """
        answers = [s.refined_answer.strip().lower() for s in refined_solutions]
        return len(set(answers)) == 1
    
    def get_majority_answer(
        self,
        refined_solutions: List[RefinedSolution]
    ) -> tuple:
        """
        Get the majority answer if it exists.
        
        Args:
            refined_solutions: All refined solutions
            
        Returns:
            Tuple of (majority_answer, count) or (None, 0) if no majority
        """
        from collections import Counter
        
        answers = [s.refined_answer.strip() for s in refined_solutions]
        counts = Counter(answers)
        most_common = counts.most_common(1)[0]
        
        if most_common[1] >= 2:  # At least 2 out of 3
            return most_common
        return (None, 0)
