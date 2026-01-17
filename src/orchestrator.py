"""
Main Orchestrator for the Multi-LLM Debate System.
Coordinates all stages of the debate workflow.
"""

import asyncio
import json
import time
from typing import List, Dict, Optional
from pathlib import Path

from config.config import SYSTEM_CONFIG, ALL_MODELS
from src.llm_clients.base_client import BaseLLMClient
from src.llm_clients.openai_client import OpenAIClient
from src.llm_clients.anthropic_client import AnthropicClient
from src.llm_clients.google_client import GoogleClient
from src.llm_clients.xai_client import XAIClient
from src.stages.role_assignment import RoleAssignment
from src.stages.solver import Solver
from src.stages.peer_review import PeerReview
from src.stages.refinement import Refinement
from src.stages.judge import Judge
from src.models.schemas import (
    Problem,
    DebateResult,
    StageResults,
    RolePreference,
    RoleAssignmentResult,
    Solution,
    PeerReviewResult,
    RefinedSolution,
    JudgmentResult
)


class DebateOrchestrator:
    """
    Orchestrates the complete Multi-LLM Debate workflow.
    
    Workflow:
    1. Stage 0: Role self-assessment
    2. Stage 0.5: Algorithmic role assignment
    3. Stage 1: Independent solution generation
    4. Stage 2: Peer review
    5. Stage 3: Refinement based on feedback
    6. Stage 4: Final judgment
    """
    
    def __init__(
        self,
        clients: Optional[Dict[str, BaseLLMClient]] = None,
        verbose: bool = True
    ):
        """
        Initialize the orchestrator.
        
        Args:
            clients: Optional dictionary of LLM clients. If None, will initialize from config.
            verbose: Whether to print progress messages
        """
        self.verbose = verbose
        
        if clients:
            self.clients = clients
        else:
            self.clients = self._initialize_clients()
        
        # Initialize stage handlers
        self.role_assignment = RoleAssignment(self.clients)
        self.solver = Solver(self.clients)
        self.peer_review = PeerReview(self.clients)
        self.refinement = Refinement(self.clients)
        self.judge = Judge(self.clients)
    
    def _initialize_clients(self) -> Dict[str, BaseLLMClient]:
        """
        Initialize LLM clients from configuration.
        
        Returns:
            Dictionary mapping model names to clients
        """
        clients = {}
        
        # Try to initialize each client
        try:
            clients["gpt4"] = OpenAIClient()
            self._log("Initialized GPT client")
        except Exception as e:
            self._log(f"Warning: Could not initialize GPT: {e}")
        
        try:
            clients["claude"] = AnthropicClient()
            self._log("Initialized Claude client")
        except Exception as e:
            self._log(f"Warning: Could not initialize Claude: {e}")
        
        try:
            clients["gemini"] = GoogleClient()
            self._log("Initialized Gemini client")
        except Exception as e:
            self._log(f"Warning: Could not initialize Gemini: {e}")
        
        try:
            clients["grok"] = XAIClient()
            self._log("Initialized Grok client")
        except Exception as e:
            self._log(f"Warning: Could not initialize Grok: {e}")
        
        if len(clients) < 4:
            self._log(f"Warning: Only {len(clients)} clients available. Need 4 for full debate.")
        
        return clients
    
    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[Orchestrator] {message}")
    
    async def run_debate(
        self,
        problem: Problem
    ) -> DebateResult:
        """
        Run the complete debate workflow for a single problem.
        
        Args:
            problem: The problem to solve
            
        Returns:
            DebateResult with all stage outputs and final answer
        """
        start_time = time.time()
        
        self._log(f"\n{'='*60}")
        self._log(f"Starting debate for problem: {problem.id}")
        self._log(f"Category: {problem.category.value}")
        self._log(f"{'='*60}")
        
        # Stage 0 & 0.5: Role Assignment
        self._log("\n[Stage 0] Getting role preferences from all models...")
        preferences, assignment = await self.role_assignment.assign_roles(problem)
        self._log(f"[Stage 0.5] Roles assigned:")
        self._log(f"  - Solver 1: {assignment.solver_1}")
        self._log(f"  - Solver 2: {assignment.solver_2}")
        self._log(f"  - Solver 3: {assignment.solver_3}")
        self._log(f"  - Judge: {assignment.judge}")
        
        # Stage 1: Independent Solutions
        self._log("\n[Stage 1] Generating independent solutions...")
        solutions = await self.solver.stage_1_generate_solutions(problem, assignment)
        self._log(f"  Solutions generated. Answers: {[s.final_answer for s in solutions]}")
        
        # Log any errors in detail
        for sol in solutions:
            if sol.final_answer == "ERROR":
                self._log(f"  [ERROR DETAIL] {sol.solver_id} ({sol.model_name}): {sol.reasoning_steps[0].description if sol.reasoning_steps else 'Unknown error'}")
        
        # Stage 2: Peer Review
        self._log("\n[Stage 2] Conducting peer reviews...")
        reviews = await self.peer_review.stage_2_peer_review(problem, solutions, assignment)
        self._log(f"  {len(reviews)} reviews completed")
        
        # Stage 3: Refinement
        self._log("\n[Stage 3] Refining solutions based on feedback...")
        refined = await self.refinement.stage_3_refine_solutions(
            problem, solutions, reviews, assignment
        )
        changes = sum(1 for r in refined if r.answer_changed)
        self._log(f"  {changes} solutions changed their answer")
        self._log(f"  Refined answers: {[r.refined_answer for r in refined]}")
        
        # Stage 4: Final Judgment
        self._log("\n[Stage 4] Judge making final decision...")
        judgment = await self.judge.stage_4_judge(
            problem, solutions, reviews, refined, assignment
        )
        self._log(f"  Winner: {judgment.winner}")
        self._log(f"  Final Answer: {judgment.winning_answer}")
        self._log(f"  Confidence: {judgment.confidence:.2f}")
        
        execution_time = time.time() - start_time
        
        # Check correctness
        is_correct = self._check_answer(judgment.winning_answer, problem.correct_answer)
        
        self._log(f"\n[Result] Correct: {is_correct}")
        self._log(f"  Expected: {problem.correct_answer}")
        self._log(f"  Got: {judgment.winning_answer}")
        self._log(f"  Time: {execution_time:.2f}s")
        
        # Build result
        result = DebateResult(
            problem=problem,
            stage_results=StageResults(
                role_preferences=preferences,
                role_assignment=assignment,
                initial_solutions=solutions,
                peer_reviews=reviews,
                refined_solutions=refined,
                judgment=judgment
            ),
            final_answer=judgment.winning_answer,
            is_correct=is_correct,
            execution_time_seconds=execution_time
        )
        
        return result
    
    def _check_answer(self, predicted: str, correct: str) -> bool:
        """
        Check if the predicted answer matches the correct answer.
        
        Handles various formats and normalizations.
        
        Args:
            predicted: The predicted answer
            correct: The correct answer
            
        Returns:
            True if answers match
        """
        # Normalize both answers
        pred_clean = self._normalize_answer(predicted)
        correct_clean = self._normalize_answer(correct)
        
        # Direct match
        if pred_clean == correct_clean:
            return True
        
        # Check if correct answer is contained (for multi-part answers)
        if correct_clean in pred_clean or pred_clean in correct_clean:
            return True
        
        # Try numerical comparison for numeric answers
        try:
            pred_num = float(pred_clean.replace(',', ''))
            # Handle multiple possible correct values (e.g., "27/216 or 1/8")
            for part in correct.split(' or '):
                try:
                    if '/' in part:
                        nums = part.strip().split('/')
                        correct_num = float(nums[0]) / float(nums[1])
                    else:
                        correct_num = float(part.strip().replace(',', ''))
                    
                    if abs(pred_num - correct_num) < 0.01:
                        return True
                except:
                    continue
        except:
            pass
        
        return False
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize an answer for comparison."""
        # Convert to lowercase and strip
        normalized = answer.lower().strip()
        
        # Remove common prefixes
        prefixes = ["the answer is", "answer:", "final answer:", "="]
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
        
        # Remove units and extra text after numbers
        import re
        # Try to extract main numeric value or key term
        numeric_match = re.search(r'^[\d.,/]+', normalized)
        if numeric_match:
            normalized = numeric_match.group()
        
        return normalized
    
    async def run_all_problems(
        self,
        problems: List[Problem],
        save_results: bool = True,
        results_path: Optional[str] = None
    ) -> List[DebateResult]:
        """
        Run the debate for all problems.
        
        Args:
            problems: List of problems to solve
            save_results: Whether to save results to file
            results_path: Path for saving results
            
        Returns:
            List of DebateResults
        """
        results = []
        
        for i, problem in enumerate(problems):
            self._log(f"\n{'#'*60}")
            self._log(f"Problem {i+1}/{len(problems)}")
            self._log(f"{'#'*60}")
            
            try:
                result = await self.run_debate(problem)
                results.append(result)
            except Exception as e:
                self._log(f"Error solving problem {problem.id}: {e}")
                # Create error result
                results.append(self._create_error_result(problem, str(e)))
        
        # Save results
        if save_results:
            self._save_results(results, results_path)
        
        return results
    
    def _create_error_result(self, problem: Problem, error: str) -> DebateResult:
        """Create a placeholder result for failed problems."""
        from src.models.schemas import (
            RoleConfidence, OverallAssessment, Evaluation
        )
        
        # Create minimal placeholder structures
        empty_preference = RolePreference(
            model_name="error",
            role_preferences=["Solver"],
            confidence_by_role=RoleConfidence(solver=0, judge=0),
            reasoning=f"Error: {error}"
        )
        
        return DebateResult(
            problem=problem,
            stage_results=StageResults(
                role_preferences=[empty_preference] * 4,
                role_assignment=RoleAssignmentResult(
                    solver_1="error", solver_2="error", solver_3="error",
                    judge="error", assignment_reasoning=f"Error: {error}"
                ),
                initial_solutions=[],
                peer_reviews=[],
                refined_solutions=[],
                judgment=JudgmentResult(
                    judge_model="error",
                    problem_id=problem.id,
                    solution_comparisons=[],
                    winner="none",
                    winning_answer="ERROR",
                    confidence=0,
                    reasoning=f"Error during debate: {error}",
                    consensus_exists=False
                )
            ),
            final_answer="ERROR",
            is_correct=False,
            execution_time_seconds=0
        )
    
    def _save_results(
        self,
        results: List[DebateResult],
        path: Optional[str] = None
    ):
        """Save debate results to JSON file."""
        if path is None:
            path = Path(SYSTEM_CONFIG.results_dir) / "debate_results.json"
        else:
            path = Path(path)
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        results_data = [r.model_dump() for r in results]
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, default=str, ensure_ascii=False)
        
        self._log(f"Results saved to {path}")


def load_problems(path: str = "data/problems.json") -> List[Problem]:
    """
    Load problems from JSON file.
    
    Args:
        path: Path to problems JSON file
        
    Returns:
        List of Problem objects
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    problems = []
    for p in data["problems"]:
        problems.append(Problem(
            id=p["id"],
            category=p["category"],
            difficulty=p["difficulty"],
            problem_text=p["problem_text"],
            correct_answer=p["correct_answer"],
            verification_method=p["verification_method"],
            hints=p.get("hints")
        ))
    
    return problems
