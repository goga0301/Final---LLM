"""
Multi-LLM Collaborative Debate System
Main entry point for running the debate system and evaluation.

Usage:
    python main.py --run-debate          # Run full debate on all problems
    python main.py --run-baselines       # Run baseline comparisons only
    python main.py --evaluate            # Evaluate existing results
    python main.py --generate-plots      # Generate visualization plots
    python main.py --full                # Run everything (debate + baselines + evaluation + plots)
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import List, Optional

from config.config import validate_api_keys, SYSTEM_CONFIG
from src.orchestrator import DebateOrchestrator, load_problems
from src.evaluation.baselines import BaselineRunner
from src.evaluation.metrics import MetricsCalculator
from src.models.schemas import DebateResult, Problem
from visualization.plots import DebatePlotter


def check_api_keys():
    """Check and report API key status."""
    print("\n" + "=" * 60)
    print("API Key Status")
    print("=" * 60)
    
    status = validate_api_keys()
    all_configured = True
    
    for model, configured in status.items():
        status_str = "[OK] Configured" if configured else "[X] Missing"
        print(f"  {model.upper()}: {status_str}")
        if not configured:
            all_configured = False
    
    if not all_configured:
        print("\nWarning: Some API keys are missing.")
        print("Create a .env file with your API keys:")
        print("  OPENAI_API_KEY=your_key")
        print("  ANTHROPIC_API_KEY=your_key")
        print("  GOOGLE_API_KEY=your_key")
        print("  XAI_API_KEY=your_key")
    
    return all_configured


async def test_api_connections(test_message: str, temperature: float = 0.5):
    """
    Test API connections by sending a test message to each LLM.
    
    Args:
        test_message: Message to send to each LLM
        temperature: Temperature setting for generation (0.0 to 1.0)
    """
    from config.config import ALL_MODELS
    from src.llm_clients.openai_client import OpenAIClient
    from src.llm_clients.anthropic_client import AnthropicClient
    from src.llm_clients.google_client import GoogleClient
    from src.llm_clients.xai_client import XAIClient
    
    print("\n" + "=" * 60)
    print("API Connection Test")
    print("=" * 60)
    print(f"Test message: \"{test_message}\"")
    print(f"Temperature: {temperature}")
    print("-" * 60)
    
    # Map model names to client classes
    client_classes = {
        "gpt4": ("OpenAI GPT", OpenAIClient),
        "claude": ("Anthropic Claude", AnthropicClient),
        "gemini": ("Google Gemini", GoogleClient),
        "grok": ("xAI Grok", XAIClient)
    }
    
    results = {}
    
    for model_key, (model_name, client_class) in client_classes.items():
        print(f"\n  Testing {model_name}...", end=" ", flush=True)
        
        try:
            # Initialize client
            client = client_class()
            
            # Send test message
            import time
            start_time = time.time()
            response = await client.generate(
                prompt=test_message,
                temperature=temperature,
                max_tokens=100
            )
            elapsed = time.time() - start_time
            
            # Success
            print(f"[OK] ({elapsed:.2f}s)")
            print(f"    Response: {response[:100]}{'...' if len(response) > 100 else ''}")
            results[model_key] = {"status": "success", "time": elapsed, "response": response}
            
        except ValueError as e:
            # API key not configured
            print(f"[SKIP] API key not configured")
            results[model_key] = {"status": "skipped", "error": str(e)}
            
        except Exception as e:
            # Connection or API error
            print(f"[FAIL]")
            print(f"    Error: {str(e)}")
            results[model_key] = {"status": "failed", "error": str(e)}
    
    # Summary
    print("\n" + "-" * 60)
    print("Summary:")
    success_count = sum(1 for r in results.values() if r["status"] == "success")
    skip_count = sum(1 for r in results.values() if r["status"] == "skipped")
    fail_count = sum(1 for r in results.values() if r["status"] == "failed")
    
    print(f"  Successful: {success_count}")
    print(f"  Skipped (no API key): {skip_count}")
    print(f"  Failed: {fail_count}")
    
    if fail_count > 0:
        print("\n[WARNING] Some API connections failed. Check your API keys and network.")
    elif success_count == 0:
        print("\n[WARNING] No API connections succeeded. Configure at least one API key.")
    else:
        print(f"\n[OK] {success_count} API connection(s) working!")
    
    return results


async def run_debate(problems: List[Problem], save_path: Optional[str] = None) -> List[DebateResult]:
    """
    Run the full debate system on all problems.
    
    Args:
        problems: List of problems to solve
        save_path: Path to save results
        
    Returns:
        List of debate results
    """
    print("\n" + "=" * 60)
    print("Running Multi-LLM Debate System")
    print("=" * 60)
    
    orchestrator = DebateOrchestrator(verbose=True)
    
    results = await orchestrator.run_all_problems(
        problems=problems,
        save_results=True,
        results_path=save_path
    )
    
    # Print summary
    correct = sum(1 for r in results if r.is_correct)
    print(f"\n{'=' * 60}")
    print(f"DEBATE COMPLETE")
    print(f"  Total Problems: {len(results)}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {correct/len(results)*100:.1f}%")
    print(f"{'=' * 60}")
    
    return results


async def run_baselines(problems: List[Problem]) -> dict:
    """
    Run baseline comparisons.
    
    Args:
        problems: List of problems
        
    Returns:
        Baseline results dictionary
    """
    print("\n" + "=" * 60)
    print("Running Baseline Comparisons")
    print("=" * 60)
    
    orchestrator = DebateOrchestrator(verbose=False)
    baseline_runner = BaselineRunner(orchestrator.clients)
    
    results = await baseline_runner.run_all_baselines(problems)
    
    # Save results
    results_path = Path(SYSTEM_CONFIG.results_dir) / "baseline_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"\nBaseline results saved to {results_path}")
    
    # Print summary
    print("\nBaseline Accuracies:")
    for model, data in results.get("single_llm", {}).items():
        print(f"  Single {model}: {data['accuracy']*100:.1f}%")
    if "voting" in results:
        print(f"  Voting: {results['voting']['accuracy']*100:.1f}%")
    
    return results


def load_results(debate_path: str = None, baseline_path: str = None) -> tuple:
    """
    Load existing results from files.
    
    Args:
        debate_path: Path to debate results
        baseline_path: Path to baseline results
        
    Returns:
        Tuple of (debate_results, baseline_results)
    """
    debate_results = []
    baseline_results = {}
    
    if debate_path is None:
        debate_path = Path(SYSTEM_CONFIG.results_dir) / "debate_results.json"
    
    if baseline_path is None:
        baseline_path = Path(SYSTEM_CONFIG.results_dir) / "baseline_results.json"
    
    # Load debate results
    if Path(debate_path).exists():
        with open(debate_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        debate_results = [DebateResult.model_validate(d) for d in data]
        print(f"Loaded {len(debate_results)} debate results from {debate_path}")
    else:
        print(f"No debate results found at {debate_path}")
    
    # Load baseline results
    if Path(baseline_path).exists():
        with open(baseline_path, 'r', encoding='utf-8') as f:
            baseline_results = json.load(f)
        print(f"Loaded baseline results from {baseline_path}")
    else:
        print(f"No baseline results found at {baseline_path}")
    
    return debate_results, baseline_results


def evaluate_results(debate_results: List[DebateResult], baseline_results: dict):
    """
    Calculate and display evaluation metrics.
    
    Args:
        debate_results: Results from debate system
        baseline_results: Results from baselines
    """
    print("\n" + "=" * 60)
    print("Calculating Evaluation Metrics")
    print("=" * 60)
    
    calculator = MetricsCalculator()
    metrics = calculator.calculate_metrics(debate_results, baseline_results)
    
    # Generate and print report
    report = calculator.generate_summary_report(metrics)
    print(report)
    
    # Save metrics
    metrics_path = Path(SYSTEM_CONFIG.results_dir) / "evaluation_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics.model_dump(), f, indent=2, default=str, ensure_ascii=False)
    print(f"\nMetrics saved to {metrics_path}")
    
    return metrics


def generate_plots(debate_results: List[DebateResult], metrics):
    """
    Generate all visualization plots.
    
    Args:
        debate_results: Results from debate system
        metrics: Evaluation metrics
    """
    print("\n" + "=" * 60)
    print("Generating Visualization Plots")
    print("=" * 60)
    
    plotter = DebatePlotter(output_dir=f"{SYSTEM_CONFIG.results_dir}/plots")
    figures = plotter.generate_all_plots(metrics, debate_results, save=True)
    
    print(f"\nGenerated {len(figures)} plots")


async def run_full_pipeline(problems: List[Problem]):
    """
    Run the complete pipeline: debate + baselines + evaluation + plots.
    
    Args:
        problems: List of problems
    """
    print("\n" + "#" * 60)
    print("MULTI-LLM DEBATE SYSTEM - FULL EVALUATION PIPELINE")
    print("#" * 60)
    
    # Run debate
    debate_results = await run_debate(problems)
    
    # Run baselines
    baseline_results = await run_baselines(problems)
    
    # Evaluate
    metrics = evaluate_results(debate_results, baseline_results)
    
    # Generate plots
    generate_plots(debate_results, metrics)
    
    print("\n" + "#" * 60)
    print("PIPELINE COMPLETE")
    print("#" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-LLM Collaborative Debate System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --full                    # Run everything
    python main.py --run-debate              # Run debate only
    python main.py --run-baselines           # Run baselines only
    python main.py --evaluate                # Evaluate existing results
    python main.py --generate-plots          # Generate plots from existing results
    python main.py --check-keys              # Check API key configuration
    python main.py --check-keys --test-message "Hello" --temperature 0.7  # Test API connections
        """
    )
    
    parser.add_argument('--full', action='store_true',
                       help='Run full pipeline (debate + baselines + evaluation + plots)')
    parser.add_argument('--run-debate', action='store_true',
                       help='Run the debate system on all problems')
    parser.add_argument('--run-baselines', action='store_true',
                       help='Run baseline comparisons')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate existing results')
    parser.add_argument('--generate-plots', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--check-keys', action='store_true',
                       help='Check API key configuration')
    parser.add_argument('--test-message', type=str, default=None,
                       help='Test message to send to LLMs for connection testing (use with --check-keys)')
    parser.add_argument('--temperature', type=float, default=0.5,
                       help='Temperature for test message generation (0.0-1.0, default: 0.5)')
    parser.add_argument('--problems-file', type=str, default='data/problems.json',
                       help='Path to problems JSON file')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of problems to process')
    
    args = parser.parse_args()
    
    # Check API keys first
    if args.check_keys:
        check_api_keys()
        if args.test_message:
            asyncio.run(test_api_connections(args.test_message, args.temperature))
        return
    
    # Default to full pipeline if no specific action
    if not any([args.full, args.run_debate, args.run_baselines, 
                args.evaluate, args.generate_plots]):
        args.full = True
    
    # Load problems
    print(f"Loading problems from {args.problems_file}...")
    problems = load_problems(args.problems_file)
    
    if args.limit:
        problems = problems[:args.limit]
        print(f"Limited to {len(problems)} problems")
    else:
        print(f"Loaded {len(problems)} problems")
    
    # Check API keys for operations that need them
    if args.full or args.run_debate or args.run_baselines:
        if not check_api_keys():
            print("\nPlease configure API keys before running the system.")
            return
    
    # Run requested operations
    if args.full:
        asyncio.run(run_full_pipeline(problems))
    else:
        if args.run_debate:
            asyncio.run(run_debate(problems))
        
        if args.run_baselines:
            asyncio.run(run_baselines(problems))
        
        if args.evaluate or args.generate_plots:
            debate_results, baseline_results = load_results()
            
            if args.evaluate:
                metrics = evaluate_results(debate_results, baseline_results)
            else:
                # Need metrics for plots
                calculator = MetricsCalculator()
                metrics = calculator.calculate_metrics(debate_results, baseline_results)
            
            if args.generate_plots:
                generate_plots(debate_results, metrics)


if __name__ == "__main__":
    main()
