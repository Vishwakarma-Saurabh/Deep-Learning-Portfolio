"""
A/B Testing Framework - Compare two prompt versions.
Serves different prompts to different users and tracks performance.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hashlib
import json
import time
from typing import Dict, List
from datetime import datetime
from collections import defaultdict


class Experiment:
    """A single A/B test experiment."""
    
    def __init__(self, name: str, variants: Dict[str, str], traffic_split: Dict[str, float]):
        """
        Args:
            name: Experiment name
            variants: Dict of variant_name → prompt_text
            traffic_split: Dict of variant_name → percentage (e.g., {"A": 0.8, "B": 0.2})
        """
        self.name = name
        self.variants = variants
        self.traffic_split = traffic_split
        self.results: Dict[str, List[Dict]] = defaultdict(list)
        self.start_time = time.time()
    
    def assign_variant(self, user_id: str) -> str:
        """Assign a user to a variant consistently."""
        # Hash user_id to get consistent assignment
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        bucket = hash_val % 100 / 100.0  # 0.0 to 1.0
        
        cumulative = 0.0
        for variant, percentage in self.traffic_split.items():
            cumulative += percentage
            if bucket <= cumulative:
                return variant
        
        return list(self.traffic_split.keys())[0]
    
    def get_prompt(self, user_id: str) -> str:
        """Get the prompt variant for a user."""
        variant = self.assign_variant(user_id)
        return self.variants[variant]
    
    def record_result(self, user_id: str, variant: str, latency: float, tokens: int, success: bool, answer_length: int):
        """Record a result for analysis."""
        self.results[variant].append({
            "user_id": user_id,
            "variant": variant,
            "latency": latency,
            "tokens": tokens,
            "success": success,
            "answer_length": answer_length,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_stats(self) -> Dict:
        """Get experiment statistics."""
        stats = {
            "experiment": self.name,
            "duration_seconds": time.time() - self.start_time,
            "variants": {}
        }
        
        for variant, results in self.results.items():
            if not results:
                stats["variants"][variant] = {"requests": 0, "message": "No data yet"}
                continue
            
            successes = [r for r in results if r["success"]]
            latencies = [r["latency"] for r in results]
            tokens_list = [r["tokens"] for r in results]
            
            stats["variants"][variant] = {
                "requests": len(results),
                "success_rate": round(len(successes) / len(results) * 100, 1),
                "avg_latency": round(sum(latencies) / len(latencies), 2),
                "avg_tokens": round(sum(tokens_list) / len(tokens_list), 0),
                "avg_answer_length": round(sum(r["answer_length"] for r in results) / len(results), 0)
            }
        
        # Determine winner
        if len(stats["variants"]) >= 2:
            variants_list = list(stats["variants"].keys())
            v1, v2 = variants_list[0], variants_list[1]
            
            s1 = stats["variants"][v1]
            s2 = stats["variants"][v2]
            
            if s1.get("requests", 0) >= 10 and s2.get("requests", 0) >= 10:
                if s1["success_rate"] > s2["success_rate"]:
                    stats["winner"] = f"{v1} (higher success rate)"
                elif s2["success_rate"] > s1["success_rate"]:
                    stats["winner"] = f"{v2} (higher success rate)"
                elif s1["avg_latency"] < s2["avg_latency"]:
                    stats["winner"] = f"{v1} (lower latency)"
                else:
                    stats["winner"] = f"{v2} (lower latency)"
        
        return stats


class ExperimentRunner:
    """Manages multiple A/B experiments."""
    
    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
    
    def create_experiment(self, name: str, variants: Dict[str, str], traffic_split: Dict[str, float]) -> Experiment:
        """Create a new experiment."""
        experiment = Experiment(name, variants, traffic_split)
        self.experiments[name] = experiment
        return experiment
    
    def get_prompt(self, experiment_name: str, user_id: str, default_prompt: str) -> str:
        """Get prompt variant for a user in an experiment."""
        if experiment_name in self.experiments:
            return self.experiments[experiment_name].get_prompt(user_id)
        return default_prompt
    
    def record_result(self, experiment_name: str, user_id: str, variant: str, latency: float, tokens: int, success: bool, answer_length: int):
        """Record a result."""
        if experiment_name in self.experiments:
            self.experiments[experiment_name].record_result(user_id, variant, latency, tokens, success, answer_length)
    
    def get_experiment_stats(self, experiment_name: str) -> Dict:
        """Get stats for an experiment."""
        if experiment_name in self.experiments:
            return self.experiments[experiment_name].get_stats()
        return {"error": "Experiment not found"}
    
    def list_experiments(self) -> List[str]:
        """List all running experiments."""
        return list(self.experiments.keys())


# Global experiment runner
experiment_runner = ExperimentRunner()

# Create a default experiment comparing two prompt styles
default_experiment = experiment_runner.create_experiment(
    name="prompt_style_comparison",
    variants={
        "concise": "You are a legal assistant. Answer in 2-3 concise sentences. Be direct and factual.",
        "detailed": "You are an expert legal analyst. Provide thorough explanations with context and reasoning. Be comprehensive."
    },
    traffic_split={
        "concise": 0.7,   # 70% of users get concise
        "detailed": 0.3   # 30% get detailed
    }
)


if __name__ == "__main__":
    # Simulate users
    users = ["user_a", "user_b", "user_c", "user_d", "user_e"]
    
    for user in users:
        variant = default_experiment.assign_variant(user)
        prompt = default_experiment.get_prompt(user)
        print(f"{user} → {variant}: {prompt[:50]}...")
        
        # Simulate results
        experiment_runner.record_result(
            "prompt_style_comparison",
            user, variant,
            latency=1.5,
            tokens=200,
            success=True,
            answer_length=120
        )
    
    print("\nExperiment Stats:")
    print(json.dumps(experiment_runner.get_experiment_stats("prompt_style_comparison"), indent=2))