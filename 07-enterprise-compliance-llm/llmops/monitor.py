"""
Monitoring System - Shared singleton that tracks performance metrics.
All files share the SAME monitor instance.
"""

import time
import json
import os
from datetime import datetime
from collections import defaultdict
from pathlib import Path


class LLMMonitor:
    """Singleton monitor - only ONE instance exists."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        
        # Load persisted data if exists
        self._load_from_disk()
    
    def record_request(self, endpoint: str, latency: float, tokens: int, success: bool):
        """Record a single request's metrics."""
        self.metrics["requests"].append({
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "latency": latency,
            "tokens": tokens,
            "success": success
        })
        
        # Persist periodically (every 10 requests)
        if len(self.metrics["requests"]) % 10 == 0:
            self._save_to_disk()
    
    def get_stats(self) -> dict:
        """Calculate current statistics."""
        requests = self.metrics["requests"]
        
        if not requests:
            return {
                "message": "No requests recorded yet. Make API calls to see metrics.",
                "uptime_seconds": time.time() - self.start_time
            }
        
        recent = requests[-100:]
        latencies = [r["latency"] for r in recent]
        tokens = [r["tokens"] for r in recent]
        successes = [r["success"] for r in recent]
        
        sorted_latencies = sorted(latencies)
        p50 = sorted_latencies[len(sorted_latencies)//2] if sorted_latencies else 0
        p95 = sorted_latencies[int(len(sorted_latencies)*0.95)] if len(sorted_latencies) >= 20 else sorted_latencies[-1] if sorted_latencies else 0
        p99 = sorted_latencies[int(len(sorted_latencies)*0.99)] if len(sorted_latencies) >= 100 else sorted_latencies[-1] if sorted_latencies else 0
        
        return {
            "uptime_seconds": time.time() - self.start_time,
            "total_requests": len(requests),
            "requests_last_hour": sum(1 for r in recent if 
                (datetime.now() - datetime.fromisoformat(r["timestamp"])).seconds < 3600),
            "success_rate": round(sum(successes) / len(successes) * 100, 1) if successes else 0,
            "avg_latency": round(sum(latencies) / len(latencies), 2) if latencies else 0,
            "latency_p50": round(p50, 2),
            "latency_p95": round(p95, 2),
            "latency_p99": round(p99, 2),
            "total_tokens": sum(tokens),
            "avg_tokens_per_request": round(sum(tokens) / len(tokens), 1) if tokens else 0,
            "endpoints": self._endpoint_stats(recent)
        }
    
    def _endpoint_stats(self, requests: list) -> dict:
        """Breakdown by endpoint."""
        stats = defaultdict(lambda: {"count": 0, "total_latency": 0, "errors": 0, "total_tokens": 0})
        
        for r in requests:
            endpoint = r["endpoint"]
            stats[endpoint]["count"] += 1
            stats[endpoint]["total_latency"] += r["latency"]
            stats[endpoint]["total_tokens"] += r.get("tokens", 0)
            if not r["success"]:
                stats[endpoint]["errors"] += 1
        
        result = {}
        for endpoint, data in stats.items():
            result[endpoint] = {
                "requests": data["count"],
                "avg_latency": round(data["total_latency"] / data["count"], 2) if data["count"] else 0,
                "error_rate": round(data["errors"] / data["count"] * 100, 1) if data["count"] else 0,
                "total_tokens": data["total_tokens"]
            }
        
        return result
    
    def should_alert(self) -> list:
        """Check if any metrics exceed thresholds."""
        stats = self.get_stats()
        alerts = []
        
        if "message" in stats:
            return alerts
        
        if stats.get("success_rate", 100) < 95:
            alerts.append(f"⚠️ Error rate above 5%: {100 - stats['success_rate']:.1f}%")
        
        if stats.get("latency_p95", 0) > 5.0:
            alerts.append(f"⚠️ P95 latency above 5s: {stats['latency_p95']:.2f}s")
        
        if stats.get("requests_last_hour", 0) > 500:
            alerts.append(f"⚠️ High traffic: {stats['requests_last_hour']} requests in last hour")
        
        return alerts
    
    def reset(self):
        """Reset all metrics (for testing)."""
        self.metrics = defaultdict(list)
        self.start_time = time.time()
    
    def _save_to_disk(self):
        """Persist metrics to disk."""
        try:
            save_path = Path("audit_logs/metrics.json")
            save_path.parent.mkdir(exist_ok=True)
            
            # Only save recent data
            recent = self.metrics["requests"][-1000:]
            with open(save_path, "w") as f:
                json.dump(recent, f, indent=2)
        except:
            pass
    
    def _load_from_disk(self):
        """Load persisted metrics from disk."""
        try:
            load_path = Path("audit_logs/metrics.json")
            if load_path.exists():
                with open(load_path) as f:
                    data = json.load(f)
                    self.metrics["requests"] = data
        except:
            pass


# Global singleton - EVERY file imports this SAME instance
monitor = LLMMonitor()


if __name__ == "__main__":
    # If run directly, show current stats (from disk)
    stats = monitor.get_stats()
    print(json.dumps(stats, indent=2))
    
    alerts = monitor.should_alert()
    if alerts:
        print("\n🚨 Alerts:")
        for alert in alerts:
            print(f"  {alert}")
    else:
        print("\n✅ All systems normal")