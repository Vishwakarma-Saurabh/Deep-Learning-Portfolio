"""
Circuit Breaker Pattern - Prevents cascading failures.
When a service fails repeatedly, stop calling it temporarily.
"""

import time
import functools
from enum import Enum
from typing import Callable, Any


class CircuitState(Enum):
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Failing, stop calls
    HALF_OPEN = "half_open"    # Testing if recovered


class CircuitBreaker:
    """Protects external service calls from cascading failures."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.success_count = 0
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                print("🔄 Circuit: HALF_OPEN - Testing if service recovered")
                self.state = CircuitState.HALF_OPEN
            else:
                remaining = int(self.recovery_timeout - (time.time() - self.last_failure_time))
                raise Exception(f"Circuit OPEN. Service unavailable. Retry in {remaining}s")
        
        try:
            result = func(*args, **kwargs)
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= 2:
                    print("✅ Circuit: CLOSED - Service recovered")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                print(f"🚨 Circuit: OPEN - {self.failure_count} failures. Blocking calls for {self.recovery_timeout}s")
                self.state = CircuitState.OPEN
            
            raise e
    
    def get_state(self) -> str:
        return self.state.value


# Circuit breakers for different services
groq_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)
qdrant_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=15)


if __name__ == "__main__":
    # Test with a failing function
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=5)
    
    def failing_func():
        raise Exception("Service down!")
    
    for i in range(5):
        try:
            breaker.call(failing_func)
        except Exception as e:
            print(f"Attempt {i+1}: {e}")
        time.sleep(1)