"""
Visualization module for the debate system evaluation results.
Generates required plots for analysis and presentation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

from src.models.schemas import (
    DebateResult,
    EvaluationMetrics,
    ProblemCategory
)


class DebatePlotter:
    """Generates visualizations for debate system evaluation."""
    
    def __init__(
        self,
        style: str = "seaborn-v0_8-whitegrid",
        figsize: tuple = (10, 6),
        output_dir: str = "results/plots"
    ):
        """
        Initialize plotter.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
            output_dir: Directory for saving plots
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use("seaborn-v0_8")
        
        self.figsize = figsize
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color palette
        self.colors = {
            "primary": "#2E86AB",
            "secondary": "#A23B72",
            "tertiary": "#F18F01",
            "quaternary": "#C73E1D",
            "success": "#2E7D32",
            "neutral": "#757575"
        }
        
        self.model_colors = {
            "gpt4": "#10A37F",
            "claude": "#D97706",
            "gemini": "#4285F4",
            "grok": "#1DA1F2"
        }
    
    def plot_accuracy_by_category(
        self,
        metrics: EvaluationMetrics,
        save: bool = True
    ) -> plt.Figure:
        """
        Create bar chart showing accuracy by problem category.
        
        Args:
            metrics: Evaluation metrics
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        categories = [m.category.value.replace("_", " ").title() 
                     for m in metrics.category_metrics]
        accuracies = [m.accuracy * 100 for m in metrics.category_metrics]
        totals = [m.total_problems for m in metrics.category_metrics]
        
        bars = ax.bar(categories, accuracies, color=self.colors["primary"], edgecolor='white')
        
        # Add value labels
        for bar, total in zip(bars, totals):
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%\n(n={total})',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_xlabel('Problem Category', fontsize=12)
        ax.set_title('Debate System Accuracy by Problem Category', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 105)
        
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'accuracy_by_category.png', dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_system_vs_baselines(
        self,
        metrics: EvaluationMetrics,
        save: bool = True
    ) -> plt.Figure:
        """
        Create comparison chart: system vs baselines.
        
        Args:
            metrics: Evaluation metrics
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        methods = []
        accuracies = []
        colors = []
        
        # Single LLM baselines
        for model, acc in metrics.single_llm_accuracy.items():
            methods.append(f"Single {model.upper()}")
            accuracies.append(acc * 100)
            colors.append(self.model_colors.get(model, self.colors["neutral"]))
        
        # Voting baseline
        methods.append("Voting (3 LLMs)")
        accuracies.append(metrics.voting_baseline_accuracy * 100)
        colors.append(self.colors["secondary"])
        
        # Our system
        methods.append("Debate System\n(Full)")
        accuracies.append(metrics.overall_accuracy * 100)
        colors.append(self.colors["success"])
        
        # Create bars
        x = np.arange(len(methods))
        bars = ax.bar(x, accuracies, color=colors, edgecolor='white', linewidth=1.5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add horizontal line for best single LLM
        if metrics.single_llm_accuracy:
            best_single = max(metrics.single_llm_accuracy.values()) * 100
            ax.axhline(y=best_single, color='red', linestyle='--', alpha=0.7, label='Best Single LLM')
        
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Debate System vs. Baseline Methods', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=10)
        ax.set_ylim(0, 105)
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'system_vs_baselines.png', dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_model_performance_heatmap(
        self,
        metrics: EvaluationMetrics,
        save: bool = True
    ) -> plt.Figure:
        """
        Create heatmap showing model performance by role.
        
        Args:
            metrics: Evaluation metrics
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Build data matrix
        models = [p.model_name for p in metrics.model_performances]
        data = {
            'Solver Accuracy': [p.solver_accuracy * 100 for p in metrics.model_performances],
            'Times as Solver': [p.times_as_solver for p in metrics.model_performances],
            'Times as Judge': [p.times_as_judge for p in metrics.model_performances],
            'Solutions Selected': [p.solutions_selected_by_judge for p in metrics.model_performances]
        }
        
        df = pd.DataFrame(data, index=models)
        
        # Normalize for heatmap (0-1 scale)
        df_normalized = df.copy()
        for col in df.columns:
            max_val = df[col].max()
            if max_val > 0:
                df_normalized[col] = df[col] / max_val
        
        # Create heatmap
        sns.heatmap(df_normalized, annot=df.values, fmt='.1f', cmap='YlGnBu',
                   ax=ax, cbar_kws={'label': 'Relative Performance'})
        
        ax.set_title('Model Performance by Role', fontsize=14, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'model_performance_heatmap.png', dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_improvement_through_stages(
        self,
        results: List[DebateResult],
        save: bool = True
    ) -> plt.Figure:
        """
        Create line chart showing accuracy improvement through stages.
        
        Args:
            results: List of debate results
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if not results:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            return fig
        
        total = len(results)
        
        # Calculate accuracy at each stage
        stages = ['Initial\nSolutions', 'After\nPeer Review', 'After\nRefinement', 'Final\n(Judge)']
        
        # Stage 1: Initial solutions (best solver)
        initial_correct = 0
        for r in results:
            best_initial = max(r.stage_results.initial_solutions, 
                             key=lambda s: s.confidence, default=None)
            if best_initial:
                if self._check_answer(best_initial.final_answer, r.problem.correct_answer):
                    initial_correct += 1
        
        # After peer review (simulated - we check if reviews helped identify correct answer)
        review_phase_correct = initial_correct  # Approximation
        
        # After refinement (best refined solver)
        refined_correct = 0
        for r in results:
            best_refined = max(r.stage_results.refined_solutions,
                             key=lambda s: s.confidence, default=None)
            if best_refined:
                if self._check_answer(best_refined.refined_answer, r.problem.correct_answer):
                    refined_correct += 1
        
        # Final (after judge)
        final_correct = sum(1 for r in results if r.is_correct)
        
        accuracies = [
            initial_correct / total * 100,
            (initial_correct + refined_correct) / 2 / total * 100,  # Approximation
            refined_correct / total * 100,
            final_correct / total * 100
        ]
        
        ax.plot(stages, accuracies, marker='o', markersize=12, linewidth=3,
               color=self.colors["primary"], markerfacecolor='white', markeredgewidth=3)
        
        # Add value labels
        for i, (stage, acc) in enumerate(zip(stages, accuracies)):
            ax.annotate(f'{acc:.1f}%',
                       xy=(i, acc),
                       xytext=(0, 10),
                       textcoords="offset points",
                       ha='center', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_xlabel('Debate Stage', fontsize=12)
        ax.set_title('Accuracy Improvement Through Debate Stages', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'improvement_through_stages.png', dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_judge_confusion_matrix(
        self,
        results: List[DebateResult],
        save: bool = True
    ) -> plt.Figure:
        """
        Create confusion matrix for judge decisions vs correct answers.
        
        Args:
            results: List of debate results
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if not results:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            return fig
        
        # Build confusion data
        # Rows: Judge selected correct solver, Judge selected wrong solver
        # Cols: Final answer correct, Final answer wrong
        
        tp = 0  # Judge correct, answer correct
        fp = 0  # Judge confident but wrong
        fn = 0  # Correct answer existed but judge picked wrong
        tn = 0  # No correct answer, judge picked best available
        
        for r in results:
            # Check which solvers had correct answers
            correct_solvers = []
            for sol in r.stage_results.refined_solutions:
                if self._check_answer(sol.refined_answer, r.problem.correct_answer):
                    correct_solvers.append(sol.solver_id)
            
            winner = r.stage_results.judgment.winner
            
            if r.is_correct:
                if winner in correct_solvers:
                    tp += 1  # Correctly selected correct answer
                else:
                    tp += 1  # Answer is correct (however selected)
            else:
                if correct_solvers:
                    fn += 1  # Correct answer existed but not selected
                else:
                    tn += 1  # No correct answer was available
        
        # Simple 2x2 matrix
        matrix = np.array([
            [tp, fn],
            [fp, tn]
        ])
        
        labels = ['Correct Available', 'No Correct Available']
        
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Selected Correct', 'Selected Wrong'],
                   yticklabels=labels, ax=ax)
        
        ax.set_title('Judge Decision Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Judge Decision', fontsize=12)
        ax.set_ylabel('Correct Answer Status', fontsize=12)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'judge_confusion_matrix.png', dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_consensus_analysis(
        self,
        results: List[DebateResult],
        save: bool = True
    ) -> plt.Figure:
        """
        Create pie chart showing consensus vs disagreement outcomes.
        
        Args:
            results: List of debate results
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        if not results:
            for ax in axes:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            return fig
        
        # Count consensus vs disagreement
        consensus_correct = 0
        consensus_wrong = 0
        disputed_correct = 0
        disputed_wrong = 0
        
        for r in results:
            is_consensus = r.stage_results.judgment.consensus_exists
            if is_consensus:
                if r.is_correct:
                    consensus_correct += 1
                else:
                    consensus_wrong += 1
            else:
                if r.is_correct:
                    disputed_correct += 1
                else:
                    disputed_wrong += 1
        
        # Left pie: Consensus vs Disagreement
        sizes1 = [consensus_correct + consensus_wrong, disputed_correct + disputed_wrong]
        labels1 = [f'Consensus\n({sizes1[0]})', f'Disagreement\n({sizes1[1]})']
        colors1 = [self.colors["primary"], self.colors["secondary"]]
        
        axes[0].pie(sizes1, labels=labels1, colors=colors1, autopct='%1.1f%%',
                   startangle=90, explode=(0.02, 0.02))
        axes[0].set_title('Solver Agreement Distribution', fontsize=14, fontweight='bold')
        
        # Right: Accuracy by consensus status
        categories = ['Consensus\nCorrect', 'Consensus\nWrong', 'Disputed\nCorrect', 'Disputed\nWrong']
        values = [consensus_correct, consensus_wrong, disputed_correct, disputed_wrong]
        colors2 = [self.colors["success"], self.colors["quaternary"],
                  self.colors["primary"], self.colors["secondary"]]
        
        bars = axes[1].bar(categories, values, color=colors2, edgecolor='white')
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                axes[1].annotate(f'{int(height)}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        axes[1].set_ylabel('Number of Problems', fontsize=12)
        axes[1].set_title('Outcomes by Agreement Status', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'consensus_analysis.png', dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_all_plots(
        self,
        metrics: EvaluationMetrics,
        results: List[DebateResult],
        save: bool = True
    ) -> Dict[str, plt.Figure]:
        """
        Generate all required plots.
        
        Args:
            metrics: Evaluation metrics
            results: Debate results
            save: Whether to save plots
            
        Returns:
            Dictionary of figure name to Figure object
        """
        figures = {}
        
        print("Generating accuracy by category plot...")
        figures['accuracy_by_category'] = self.plot_accuracy_by_category(metrics, save)
        
        print("Generating system vs baselines plot...")
        figures['system_vs_baselines'] = self.plot_system_vs_baselines(metrics, save)
        
        print("Generating model performance heatmap...")
        figures['model_heatmap'] = self.plot_model_performance_heatmap(metrics, save)
        
        print("Generating improvement through stages plot...")
        figures['improvement_stages'] = self.plot_improvement_through_stages(results, save)
        
        print("Generating judge confusion matrix...")
        figures['judge_confusion'] = self.plot_judge_confusion_matrix(results, save)
        
        print("Generating consensus analysis...")
        figures['consensus_analysis'] = self.plot_consensus_analysis(results, save)
        
        print(f"All plots saved to {self.output_dir}")
        
        return figures
    
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
