"""
Baseline implementations for comparison.
- Single-LLM baseline: Ask each model once
- Simple voting baseline: 3 models vote, majority wins
"""

import asyncio
from typing import List, Dict, Optional
from collections import Counter

from src.llm_clients.base_client import BaseLLMClient
from src.models.schemas import Problem


class BaselineRunner:
    """Runs baseline comparisons for the debate system."""
    
    SINGLE_PROMPT_TEMPLATE = """Solve the following problem. Think step by step, then provide your final answer.

PROBLEM:
{problem_text}

CATEGORY: {category}

Provide your answer in this format:
REASONING: [Your step-by-step reasoning]
FINAL ANSWER: [Your answer]"""

    def __init__(self, clients: Dict[str, BaseLLMClient]):
        """
        Initialize baseline runner.
        
        Args:
            clients: Dictionary mapping model names to clients
        """
        self.clients = clients
    
    async def run_single_llm(
        self,
        client: BaseLLMClient,
        model_name: str,
        problem: Problem
    ) -> tuple:
        """
        Run a single LLM on a problem (baseline).
        
        Args:
            client: The LLM client
            model_name: Name of the model
            problem: The problem to solve
            
        Returns:
            Tuple of (answer, is_correct)
        """
        prompt = self.SINGLE_PROMPT_TEMPLATE.format(
            problem_text=problem.problem_text,
            category=problem.category.value
        )
        
        try:
            response = await client.generate_with_retry(
                prompt=prompt,
                temperature=0.7  # Same as solver for fair comparison
            )
            
            # Extract answer
            answer = self._extract_answer(response)
            is_correct = self._check_answer(answer, problem.correct_answer)
            
            return answer, is_correct
        except Exception as e:
            return f"ERROR: {str(e)}", False
    
    async def run_single_llm_baseline(
        self,
        problems: List[Problem]
    ) -> Dict[str, Dict]:
        """
        Run single-LLM baseline for all problems and all models.
        
        Args:
            problems: List of problems
            
        Returns:
            Dictionary with results per model
        """
        results = {}
        
        for model_name, client in self.clients.items():
            model_results = {
                "correct": 0,
                "total": len(problems),
                "answers": []
            }
            
            tasks = [
                self.run_single_llm(client, model_name, problem)
                for problem in problems
            ]
            
            answers = await asyncio.gather(*tasks)
            
            for problem, (answer, is_correct) in zip(problems, answers):
                model_results["answers"].append({
                    "problem_id": problem.id,
                    "answer": answer,
                    "correct": is_correct
                })
                if is_correct:
                    model_results["correct"] += 1
            
            model_results["accuracy"] = model_results["correct"] / model_results["total"]
            results[model_name] = model_results
        
        return results
    
    async def run_voting_baseline(
        self,
        problems: List[Problem],
        voters: Optional[List[str]] = None
    ) -> Dict:
        """
        Run simple majority voting baseline.
        
        Uses 3 models to vote, majority wins.
        
        Args:
            problems: List of problems
            voters: Optional list of 3 model names to use as voters
            
        Returns:
            Dictionary with voting results
        """
        if voters is None:
            voters = list(self.clients.keys())[:3]
        
        if len(voters) < 3:
            raise ValueError(f"Need at least 3 voters, got {len(voters)}")
        
        voters = voters[:3]  # Use first 3
        
        results = {
            "correct": 0,
            "total": len(problems),
            "answers": [],
            "voters": voters
        }
        
        for problem in problems:
            # Get answers from all voters
            tasks = [
                self.run_single_llm(self.clients[voter], voter, problem)
                for voter in voters
            ]
            
            voter_answers = await asyncio.gather(*tasks)
            
            # Extract just the answers for voting
            answers_only = [a[0] for a in voter_answers]
            
            # Determine majority
            majority_answer, majority_correct = self._majority_vote(
                answers_only, problem.correct_answer
            )
            
            results["answers"].append({
                "problem_id": problem.id,
                "individual_answers": answers_only,
                "majority_answer": majority_answer,
                "correct": majority_correct
            })
            
            if majority_correct:
                results["correct"] += 1
        
        results["accuracy"] = results["correct"] / results["total"]
        return results
    
    def _majority_vote(
        self,
        answers: List[str],
        correct_answer: str
    ) -> tuple:
        """
        Determine majority answer and correctness.
        
        Args:
            answers: List of answers from voters
            correct_answer: The correct answer
            
        Returns:
            Tuple of (majority_answer, is_correct)
        """
        # Normalize answers for comparison
        normalized = [self._normalize_answer(a) for a in answers]
        
        # Count votes
        counts = Counter(normalized)
        majority_normalized = counts.most_common(1)[0][0]
        
        # Find original answer (unnormalized)
        for a, n in zip(answers, normalized):
            if n == majority_normalized:
                majority_answer = a
                break
        
        is_correct = self._check_answer(majority_answer, correct_answer)
        return majority_answer, is_correct
    
    def _extract_answer(self, response: str) -> str:
        """Extract the final answer from a response."""
        response_lower = response.lower()
        
        # Look for "FINAL ANSWER:" marker
        if "final answer:" in response_lower:
            idx = response_lower.rfind("final answer:")
            answer = response[idx + len("final answer:"):].strip()
            # Take first line
            answer = answer.split('\n')[0].strip()
            return answer
        
        # Look for "Answer:" marker
        if "answer:" in response_lower:
            idx = response_lower.rfind("answer:")
            answer = response[idx + len("answer:"):].strip()
            answer = answer.split('\n')[0].strip()
            return answer
        
        # Return last non-empty line as fallback
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        if lines:
            return lines[-1]
        
        return response.strip()
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        normalized = answer.lower().strip()
        
        # Remove common prefixes
        prefixes = ["the answer is", "answer:", "=", "therefore"]
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
        
        return normalized
    
    def _check_answer(self, predicted: str, correct: str) -> bool:
        """Check if predicted answer matches correct answer."""
        pred_clean = self._normalize_answer(predicted)
        correct_clean = self._normalize_answer(correct)
        
        # Direct match
        if pred_clean == correct_clean:
            return True
        
        # Containment check
        if correct_clean in pred_clean or pred_clean in correct_clean:
            return True
        
        # Numeric comparison
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
    
    async def run_all_baselines(
        self,
        problems: List[Problem]
    ) -> Dict:
        """
        Run all baseline methods.
        
        Args:
            problems: List of problems
            
        Returns:
            Dictionary with all baseline results
        """
        print("Running single-LLM baselines...")
        single_results = await self.run_single_llm_baseline(problems)
        
        print("Running voting baseline...")
        voting_results = await self.run_voting_baseline(problems)
        
        return {
            "single_llm": single_results,
            "voting": voting_results
        }
