"""
Pydantic models for structured LLM outputs in the debate system.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum


class ProblemCategory(str, Enum):
    """Categories of problems in the dataset."""
    MATHEMATICAL = "mathematical_logical"
    PHYSICS = "physics_scientific"
    LOGIC_PUZZLE = "logic_puzzle"
    GAME_THEORY = "game_theory"


class Difficulty(str, Enum):
    """Problem difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Role(str, Enum):
    """Roles in the debate system."""
    SOLVER_1 = "solver_1"
    SOLVER_2 = "solver_2"
    SOLVER_3 = "solver_3"
    JUDGE = "judge"


class ErrorSeverity(str, Enum):
    """Severity levels for errors identified in peer review."""
    MINOR = "minor"
    MODERATE = "moderate"
    CRITICAL = "critical"


class OverallAssessment(str, Enum):
    """Overall assessment categories for peer review."""
    EXCELLENT = "excellent"
    GOOD = "good"
    PROMISING_BUT_FLAWED = "promising_but_flawed"
    NEEDS_MAJOR_REVISION = "needs_major_revision"
    FUNDAMENTALLY_WRONG = "fundamentally_wrong"


# ============== Problem Dataset Models ==============

class Problem(BaseModel):
    """A problem in the dataset."""
    id: str = Field(..., description="Unique identifier for the problem")
    category: ProblemCategory = Field(..., description="Problem category")
    difficulty: Difficulty = Field(..., description="Problem difficulty")
    problem_text: str = Field(..., description="The full problem statement")
    correct_answer: str = Field(..., description="The verified correct answer")
    verification_method: str = Field(..., description="How the answer was verified")
    hints: Optional[List[str]] = Field(default=None, description="Optional hints")


# ============== Stage 0: Role Assignment Models ==============

class RoleConfidence(BaseModel):
    """Confidence scores for each role."""
    solver: float = Field(..., ge=0.0, le=1.0, description="Confidence for Solver role")
    judge: float = Field(..., ge=0.0, le=1.0, description="Confidence for Judge role")


class RolePreference(BaseModel):
    """LLM's self-assessment for role assignment."""
    model_name: str = Field(..., description="Name of the LLM model")
    role_preferences: List[str] = Field(..., description="Ordered list of preferred roles")
    confidence_by_role: RoleConfidence = Field(..., description="Confidence scores by role")
    reasoning: str = Field(..., description="Explanation for role preferences")
    problem_analysis: Optional[str] = Field(default=None, description="Brief analysis of the problem")


class RoleAssignmentResult(BaseModel):
    """Result of the role assignment algorithm."""
    solver_1: str = Field(..., description="Model assigned to Solver 1")
    solver_2: str = Field(..., description="Model assigned to Solver 2")
    solver_3: str = Field(..., description="Model assigned to Solver 3")
    judge: str = Field(..., description="Model assigned to Judge")
    assignment_reasoning: str = Field(..., description="Explanation of assignment decisions")


# ============== Stage 1: Solution Models ==============

class ReasoningStep(BaseModel):
    """A single step in the solution reasoning."""
    step_number: int = Field(..., description="Step number in the solution")
    description: str = Field(..., description="Description of this step")
    calculation: Optional[str] = Field(default=None, description="Any calculations performed")
    result: Optional[str] = Field(default=None, description="Result of this step")


class Solution(BaseModel):
    """A complete solution from a Solver."""
    solver_id: str = Field(..., description="Identifier of the solver")
    model_name: str = Field(..., description="Name of the LLM model")
    problem_id: str = Field(..., description="ID of the problem being solved")
    reasoning_steps: List[ReasoningStep] = Field(..., description="Step-by-step reasoning")
    final_answer: str = Field(..., description="The final answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the solution")
    assumptions: Optional[List[str]] = Field(default=None, description="Any assumptions made")
    alternative_approaches: Optional[List[str]] = Field(default=None, description="Other approaches considered")


# ============== Stage 2: Peer Review Models ==============

class Error(BaseModel):
    """An error identified in a solution."""
    location: str = Field(..., description="Where the error occurs (e.g., 'Step 5')")
    error_type: str = Field(..., description="Type of error (logical, calculation, etc.)")
    description: str = Field(..., description="Description of the error")
    severity: ErrorSeverity = Field(..., description="How severe the error is")


class Evaluation(BaseModel):
    """Detailed evaluation of a solution."""
    strengths: List[str] = Field(..., description="Positive aspects of the solution")
    weaknesses: List[str] = Field(..., description="Areas needing improvement")
    errors: List[Error] = Field(default_factory=list, description="Specific errors found")
    suggested_changes: List[str] = Field(..., description="Recommended improvements")


class PeerReviewResult(BaseModel):
    """Complete peer review from one Solver about another's solution."""
    reviewer_id: str = Field(..., description="ID of the reviewing solver")
    reviewer_model: str = Field(..., description="Model name of the reviewer")
    solution_id: str = Field(..., description="ID of the solution being reviewed")
    evaluation: Evaluation = Field(..., description="Detailed evaluation")
    overall_assessment: OverallAssessment = Field(..., description="Overall quality assessment")
    agreement_with_answer: bool = Field(..., description="Whether reviewer agrees with the answer")
    alternative_answer: Optional[str] = Field(default=None, description="Reviewer's alternative answer if disagreeing")


# ============== Stage 3: Refinement Models ==============

class CritiqueResponse(BaseModel):
    """Response to a specific critique from peer review."""
    critique: str = Field(..., description="The original critique")
    response: str = Field(..., description="How the critique was addressed")
    accepted: bool = Field(..., description="Whether the critique was accepted")
    changes_made: Optional[str] = Field(default=None, description="Specific changes made if accepted")


class RefinedSolution(BaseModel):
    """A refined solution after incorporating peer feedback."""
    solver_id: str = Field(..., description="Identifier of the solver")
    model_name: str = Field(..., description="Name of the LLM model")
    problem_id: str = Field(..., description="ID of the problem")
    original_answer: str = Field(..., description="The original answer before refinement")
    critique_responses: List[CritiqueResponse] = Field(..., description="Responses to each critique")
    refined_reasoning: List[ReasoningStep] = Field(..., description="Updated reasoning steps")
    refined_answer: str = Field(..., description="The refined final answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in refined solution")
    answer_changed: bool = Field(..., description="Whether the answer changed from original")


# ============== Stage 4: Judgment Models ==============

class SolutionComparison(BaseModel):
    """Comparison of a single solution by the judge."""
    solver_id: str = Field(..., description="ID of the solver")
    strengths: List[str] = Field(..., description="Key strengths identified")
    weaknesses: List[str] = Field(..., description="Key weaknesses identified")
    correctness_assessment: str = Field(..., description="Assessment of answer correctness")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Quality score 0-1")


class JudgmentResult(BaseModel):
    """Final judgment from the Judge LLM."""
    judge_model: str = Field(..., description="Name of the judge model")
    problem_id: str = Field(..., description="ID of the problem")
    solution_comparisons: List[SolutionComparison] = Field(..., description="Analysis of each solution")
    winner: str = Field(..., description="ID of the winning solver")
    winning_answer: str = Field(..., description="The selected final answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the judgment")
    reasoning: str = Field(..., description="Detailed reasoning for the decision")
    consensus_exists: bool = Field(..., description="Whether all solvers agreed")


# ============== Complete Debate Result Models ==============

class StageResults(BaseModel):
    """Results from all stages for a single problem."""
    role_preferences: List[RolePreference] = Field(..., description="Stage 0 results")
    role_assignment: RoleAssignmentResult = Field(..., description="Stage 0.5 results")
    initial_solutions: List[Solution] = Field(..., description="Stage 1 results")
    peer_reviews: List[PeerReviewResult] = Field(..., description="Stage 2 results")
    refined_solutions: List[RefinedSolution] = Field(..., description="Stage 3 results")
    judgment: JudgmentResult = Field(..., description="Stage 4 results")


class DebateResult(BaseModel):
    """Complete result from the debate system for one problem."""
    problem: Problem = Field(..., description="The problem that was solved")
    stage_results: StageResults = Field(..., description="Results from all stages")
    final_answer: str = Field(..., description="The final selected answer")
    is_correct: Optional[bool] = Field(default=None, description="Whether answer matches ground truth")
    execution_time_seconds: float = Field(..., description="Total execution time")


# ============== Evaluation Metrics Models ==============

class CategoryMetrics(BaseModel):
    """Metrics for a specific problem category."""
    category: ProblemCategory = Field(..., description="The problem category")
    total_problems: int = Field(..., description="Total problems in category")
    correct_count: int = Field(..., description="Number solved correctly")
    accuracy: float = Field(..., description="Accuracy percentage")


class ModelPerformance(BaseModel):
    """Performance metrics for a specific model."""
    model_name: str = Field(..., description="Name of the model")
    times_as_solver: int = Field(..., description="Number of times assigned as solver")
    times_as_judge: int = Field(..., description="Number of times assigned as judge")
    solver_accuracy: float = Field(..., description="Accuracy when acting as solver")
    solutions_selected_by_judge: int = Field(..., description="Times solution was selected")


class EvaluationMetrics(BaseModel):
    """Complete evaluation metrics for the system."""
    total_problems: int = Field(..., description="Total problems evaluated")
    overall_accuracy: float = Field(..., description="System-wide accuracy")
    improvement_rate: float = Field(..., description="Rate of improvement after refinement")
    consensus_rate: float = Field(..., description="Rate of solver consensus")
    judge_accuracy: float = Field(..., description="Judge accuracy on disputed problems")
    category_metrics: List[CategoryMetrics] = Field(..., description="Per-category metrics")
    model_performances: List[ModelPerformance] = Field(..., description="Per-model metrics")
    
    # Baseline comparisons
    single_llm_accuracy: Dict[str, float] = Field(..., description="Single LLM baseline accuracies")
    voting_baseline_accuracy: float = Field(..., description="Simple voting baseline accuracy")
    system_vs_best_single: float = Field(..., description="Improvement over best single LLM")
