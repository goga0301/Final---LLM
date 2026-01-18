"""
Evaluation metrics calculator for the debate system.
"""

from typing import List, Dict
from collections import defaultdict

from src.models.schemas import (
    DebateResult,
    EvaluationMetrics,
    CategoryMetrics,
    ModelPerformance,
    ProblemCategory
)


class MetricsCalculator:
    """Calculates evaluation metrics for the debate system."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        pass
    
    def calculate_metrics(
        self,
        debate_results: List[DebateResult],
        baseline_results: Dict
    ) -> EvaluationMetrics:
        """
        Calculate all evaluation metrics.
        
        Args:
            debate_results: Results from the debate system
            baseline_results: Results from baseline methods
            
        Returns:
            EvaluationMetrics with all calculated values
        """
        total = len(debate_results)
        if total == 0:
            raise ValueError("No debate results to evaluate")
        
        # Overall accuracy
        correct = sum(1 for r in debate_results if r.is_correct)
        overall_accuracy = correct / total
        
        # Improvement rate (answers that changed and became correct)
        improvement_count = 0
        for result in debate_results:
            # Check if any solver changed their answer and the final answer is correct
            if result.is_correct:
                for refined in result.stage_results.refined_solutions:
                    if refined.answer_changed:
                        # Check if original was wrong
                        original = next(
                            (s for s in result.stage_results.initial_solutions 
                             if s.solver_id == refined.solver_id),
                            None
                        )
                        if original and original.final_answer != refined.refined_answer:
                            improvement_count += 1
                            break
        
        improvement_rate = improvement_count / total if total > 0 else 0
        
        # Consensus rate
        consensus_count = sum(
            1 for r in debate_results 
            if r.stage_results.judgment.consensus_exists
        )
        consensus_rate = consensus_count / total
        
        # Judge accuracy (when solvers disagreed)
        disputed = [r for r in debate_results if not r.stage_results.judgment.consensus_exists]
        judge_correct = sum(1 for r in disputed if r.is_correct)
        judge_accuracy = judge_correct / len(disputed) if disputed else 1.0
        
        # Category metrics
        category_metrics = self._calculate_category_metrics(debate_results)
        
        # Model performance
        model_performances = self._calculate_model_performance(debate_results)
        
        # Baseline accuracies
        single_llm_accuracy = {}
        if "single_llm" in baseline_results:
            for model, data in baseline_results["single_llm"].items():
                single_llm_accuracy[model] = data.get("accuracy", 0)
        
        voting_accuracy = 0
        if "voting" in baseline_results:
            voting_accuracy = baseline_results["voting"].get("accuracy", 0)
        
        # System vs best single LLM
        best_single = max(single_llm_accuracy.values()) if single_llm_accuracy else 0
        system_vs_best = overall_accuracy - best_single
        
        return EvaluationMetrics(
            total_problems=total,
            overall_accuracy=overall_accuracy,
            improvement_rate=improvement_rate,
            consensus_rate=consensus_rate,
            judge_accuracy=judge_accuracy,
            category_metrics=category_metrics,
            model_performances=model_performances,
            single_llm_accuracy=single_llm_accuracy,
            voting_baseline_accuracy=voting_accuracy,
            system_vs_best_single=system_vs_best
        )
    
    def _calculate_category_metrics(
        self,
        results: List[DebateResult]
    ) -> List[CategoryMetrics]:
        """Calculate metrics per problem category."""
        by_category = defaultdict(list)
        
        for result in results:
            category = result.problem.category
            by_category[category].append(result)
        
        metrics = []
        for category, cat_results in by_category.items():
            total = len(cat_results)
            correct = sum(1 for r in cat_results if r.is_correct)
            
            metrics.append(CategoryMetrics(
                category=category,
                total_problems=total,
                correct_count=correct,
                accuracy=correct / total if total > 0 else 0
            ))
        
        return metrics
    
    def _calculate_model_performance(
        self,
        results: List[DebateResult]
    ) -> List[ModelPerformance]:
        """Calculate performance metrics per model."""
        model_stats = defaultdict(lambda: {
            "solver_count": 0,
            "judge_count": 0,
            "solver_correct": 0,
            "selected_count": 0
        })
        
        for result in results:
            assignment = result.stage_results.role_assignment
            judgment = result.stage_results.judgment
            
            # Track solver assignments
            for solver_id in ["solver_1", "solver_2", "solver_3"]:
                model = getattr(assignment, solver_id)
                model_stats[model]["solver_count"] += 1
                
                # Check if this solver's answer was correct
                refined = next(
                    (s for s in result.stage_results.refined_solutions 
                     if s.solver_id == solver_id),
                    None
                )
                if refined:
                    is_solver_correct = self._check_answer(
                        refined.refined_answer, 
                        result.problem.correct_answer
                    )
                    if is_solver_correct:
                        model_stats[model]["solver_correct"] += 1
                    
                    # Check if selected by judge
                    if judgment.winner == solver_id:
                        model_stats[model]["selected_count"] += 1
            
            # Track judge assignments
            model_stats[assignment.judge]["judge_count"] += 1
        
        performances = []
        for model, stats in model_stats.items():
            solver_accuracy = (
                stats["solver_correct"] / stats["solver_count"]
                if stats["solver_count"] > 0 else 0
            )
            
            performances.append(ModelPerformance(
                model_name=model,
                times_as_solver=stats["solver_count"],
                times_as_judge=stats["judge_count"],
                solver_accuracy=solver_accuracy,
                solutions_selected_by_judge=stats["selected_count"]
            ))
        
        return performances
    
    def _check_answer(self, predicted: str, correct: str) -> bool:
        """Check if predicted matches correct answer."""
        pred_clean = predicted.lower().strip()
        correct_clean = correct.lower().strip()
        
        if pred_clean == correct_clean:
            return True
        
        if correct_clean in pred_clean or pred_clean in correct_clean:
            return True
        
        try:
            pred_num = float(pred_clean.replace(',', ''))
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
    
    def generate_summary_report(
        self,
        metrics: EvaluationMetrics
    ) -> str:
        """
        Generate a text summary report of metrics.
        
        Args:
            metrics: Calculated evaluation metrics
            
        Returns:
            Formatted summary string
        """
        report = []
        report.append("=" * 60)
        report.append("MULTI-LLM DEBATE SYSTEM - EVALUATION REPORT")
        report.append("=" * 60)
        
        report.append(f"\nOVERALL PERFORMANCE:")
        report.append(f"  Total Problems: {metrics.total_problems}")
        report.append(f"  Overall Accuracy: {metrics.overall_accuracy:.1%}")
        report.append(f"  Improvement Rate: {metrics.improvement_rate:.1%}")
        report.append(f"  Consensus Rate: {metrics.consensus_rate:.1%}")
        report.append(f"  Judge Accuracy (disputed): {metrics.judge_accuracy:.1%}")
        
        report.append(f"\nCATEGORY BREAKDOWN:")
        for cat_metric in metrics.category_metrics:
            report.append(
                f"  {cat_metric.category.value}: "
                f"{cat_metric.correct_count}/{cat_metric.total_problems} "
                f"({cat_metric.accuracy:.1%})"
            )
        
        report.append(f"\nMODEL PERFORMANCE:")
        for model_perf in metrics.model_performances:
            report.append(f"  {model_perf.model_name}:")
            report.append(f"    As Solver: {model_perf.times_as_solver} times, "
                         f"{model_perf.solver_accuracy:.1%} accuracy")
            report.append(f"    As Judge: {model_perf.times_as_judge} times")
            report.append(f"    Selected by Judge: {model_perf.solutions_selected_by_judge} times")
        
        report.append(f"\nBASELINE COMPARISONS:")
        for model, acc in metrics.single_llm_accuracy.items():
            report.append(f"  Single {model}: {acc:.1%}")
        report.append(f"  Voting Baseline: {metrics.voting_baseline_accuracy:.1%}")
        report.append(f"  Our System: {metrics.overall_accuracy:.1%}")
        report.append(f"  Improvement over best single: {metrics.system_vs_best_single:+.1%}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
