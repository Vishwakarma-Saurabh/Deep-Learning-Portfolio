"""
Async Task Queue - Background processing for large documents.
Prevents API blocking on long operations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uuid
import time
import threading
from typing import Dict, Callable
from collections import defaultdict
from datetime import datetime


class TaskStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskQueue:
    """Simple background task queue for async processing."""
    
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.queue = []
        self.worker_thread = None
        self.running = False
    
    def submit(self, task_name: str, func: Callable, *args, **kwargs) -> str:
        """
        Submit a task for background processing.
        
        Args:
            task_name: Human-readable task name
            func: Function to execute
            *args, **kwargs: Arguments for the function
            
        Returns:
            task_id for status checking
        """
        task_id = str(uuid.uuid4())[:8]
        
        self.tasks[task_id] = {
            "task_id": task_id,
            "name": task_name,
            "status": TaskStatus.PENDING,
            "submitted_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
            "func": func,
            "args": args,
            "kwargs": kwargs
        }
        
        self.queue.append(task_id)
        
        # Start worker if not running
        if not self.running:
            self.start_worker()
        
        return task_id
    
    def get_status(self, task_id: str) -> Dict:
        """Check status of a task."""
        if task_id not in self.tasks:
            return {"error": "Task not found", "task_id": task_id}
        
        task = self.tasks[task_id]
        return {
            "task_id": task["task_id"],
            "name": task["name"],
            "status": task["status"],
            "submitted_at": task["submitted_at"],
            "started_at": task["started_at"],
            "completed_at": task["completed_at"],
            "error": task["error"]
        }
    
    def get_result(self, task_id: str) -> Dict:
        """Get result of a completed task."""
        if task_id not in self.tasks:
            return {"error": "Task not found"}
        
        task = self.tasks[task_id]
        
        if task["status"] == TaskStatus.COMPLETED:
            return {
                "task_id": task_id,
                "status": "completed",
                "result": task["result"]
            }
        elif task["status"] == TaskStatus.FAILED:
            return {
                "task_id": task_id,
                "status": "failed",
                "error": task["error"]
            }
        else:
            return {
                "task_id": task_id,
                "status": task["status"],
                "message": "Task still processing"
            }
    
    def start_worker(self):
        """Start background worker thread."""
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
    
    def _process_queue(self):
        """Worker loop - processes tasks one at a time."""
        while self.running:
            if self.queue:
                task_id = self.queue.pop(0)
                task = self.tasks[task_id]
                
                task["status"] = TaskStatus.PROCESSING
                task["started_at"] = datetime.now().isoformat()
                
                try:
                    result = task["func"](*task["args"], **task["kwargs"])
                    task["result"] = result
                    task["status"] = TaskStatus.COMPLETED
                    task["completed_at"] = datetime.now().isoformat()
                except Exception as e:
                    task["error"] = str(e)
                    task["status"] = TaskStatus.FAILED
                    task["completed_at"] = datetime.now().isoformat()
            else:
                time.sleep(0.5)  # Wait for new tasks
    
    def get_stats(self) -> Dict:
        """Get queue statistics."""
        status_counts = defaultdict(int)
        for task in self.tasks.values():
            status_counts[task["status"]] += 1
        
        return {
            "total_tasks": len(self.tasks),
            "pending": status_counts.get(TaskStatus.PENDING, 0),
            "processing": status_counts.get(TaskStatus.PROCESSING, 0),
            "completed": status_counts.get(TaskStatus.COMPLETED, 0),
            "failed": status_counts.get(TaskStatus.FAILED, 0),
            "queued": len(self.queue)
        }


# Global queue instance
task_queue = TaskQueue()


if __name__ == "__main__":
    # Test with a slow task
    def slow_task(text: str, delay: int = 2):
        """Simulate slow processing."""
        time.sleep(delay)
        return {"text": text.upper(), "length": len(text), "processed": True}
    
    # Submit task
    task_id = task_queue.submit("uppercase_text", slow_task, "hello world", delay=1)
    print(f"Task submitted: {task_id}")
    print(f"Status: {task_queue.get_status(task_id)['status']}")
    
    # Wait and check
    time.sleep(2)
    print(f"Result: {task_queue.get_result(task_id)}")
    print(f"Stats: {task_queue.get_stats()}")