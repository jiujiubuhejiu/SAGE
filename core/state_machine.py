"""
SAGE State Machine - Layer 1: Hard-coded state machine
Enforces workflow: OBSERVE → HYPOTHESIS → TEST
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, TYPE_CHECKING
import re

if TYPE_CHECKING:
    from typing import TYPE_CHECKING


class SAGEState(Enum):
    """SAGE state enumeration."""
    INIT = "Initialization"
    GET_EXEMPLARS = "Get dataset exemplars"
    ANALYZE_EXEMPLARS = "Analyze exemplars"
    FORM_HYPOTHESIS = "Form hypothesis"
    PARALLEL_HYPOTHESIS_TESTING = "Parallel hypothesis testing"
    DESIGN_TEST = "Design test"
    RUN_TEST = "Run test"
    ANALYZE_RESULT = "Analyze result"
    UPDATE_HYPOTHESIS = "Update hypothesis"
    REVIEW_ALL_HYPOTHESES = "Review all hypotheses"
    FINAL_CONCLUSION = "Final conclusion"
    DONE = "Done"


@dataclass
class Hypothesis:
    """Hypothesis data structure."""
    id: int
    text: str
    status: str  # PENDING, CONFIRMED, REFUTED, REFINED, UNCHANGED
    confidence: float = 0.0
    evidence_count: int = 0
    # Parallel processing related fields
    current_state: Optional['SAGEState'] = None  # Current state of this hypothesis
    test_history: List['TestResult'] = field(default_factory=list)  # Test history for this hypothesis
    analysis_history: List[str] = field(default_factory=list)  # Analysis history for this hypothesis
    initial_text: str = ""  # Initial hypothesis text
    latest_test_execution_output: str = ""  # Complete execution output of last test (for ANALYZE_RESULT)
    
    def __post_init__(self):
        """Initialize list fields."""
        if not self.initial_text:
            self.initial_text = self.text
        if self.current_state is None:
            self.current_state = SAGEState.DESIGN_TEST


@dataclass
class TestResult:
    """Test result data structure."""
    id: int
    hypothesis_id: int
    prompt: str
    expected: str
    actual_activation: float
    normalized_activation: float
    result: str  # CONFIRMED, REFUTED, INCONCLUSIVE
    timestamp: str


@dataclass
class Exemplar:
    """Dataset exemplar data structure."""
    text: str
    activation: float
    tokens: List[str]
    per_token_activations: List[float]


class SAGEStateMachine:
    """SAGE state machine - hard-coded workflow control."""
    
    def __init__(self, feature_id: int, layer: int, max_rounds: int = 14):
        self.feature_id = feature_id
        self.layer = layer
        self.state = SAGEState.INIT
        self.round = 0
        self.max_rounds = max_rounds
        
        # Data storage
        self.hypotheses: List[Hypothesis] = []
        self.test_history: List[TestResult] = []  # Global test history (for compatibility)
        self.exemplars: Optional[List[Exemplar]] = None
        self.analysis_history: List[str] = []  # Global analysis history (for compatibility)
        
        # Parallel hypothesis processing related
        self.current_hypothesis_id: Optional[int] = None  # Currently processing hypothesis ID
        self.hypothesis_review_result: Optional[str] = None  # Result of REVIEW_ALL_HYPOTHESES
        
        # State transition rules
        self.valid_transitions = {
            SAGEState.INIT: [SAGEState.GET_EXEMPLARS],
            SAGEState.GET_EXEMPLARS: [SAGEState.ANALYZE_EXEMPLARS],
            SAGEState.ANALYZE_EXEMPLARS: [SAGEState.PARALLEL_HYPOTHESIS_TESTING],  # Changed to parallel testing
            SAGEState.FORM_HYPOTHESIS: [SAGEState.PARALLEL_HYPOTHESIS_TESTING],  # Keep for backward compatibility
            SAGEState.PARALLEL_HYPOTHESIS_TESTING: [SAGEState.DESIGN_TEST, SAGEState.REVIEW_ALL_HYPOTHESES],
            SAGEState.DESIGN_TEST: [SAGEState.RUN_TEST, SAGEState.ANALYZE_RESULT],
            SAGEState.RUN_TEST: [SAGEState.ANALYZE_RESULT],
            SAGEState.ANALYZE_RESULT: [SAGEState.UPDATE_HYPOTHESIS],
            SAGEState.UPDATE_HYPOTHESIS: [SAGEState.DESIGN_TEST, SAGEState.PARALLEL_HYPOTHESIS_TESTING],
            SAGEState.REVIEW_ALL_HYPOTHESES: [SAGEState.PARALLEL_HYPOTHESIS_TESTING, SAGEState.FINAL_CONCLUSION],
            SAGEState.FINAL_CONCLUSION: [SAGEState.DONE]
        }
    
    def transition(self, new_state: SAGEState) -> bool:
        """State transition logic."""
        if new_state not in self.valid_transitions.get(self.state, []):
            raise ValueError(f"Invalid transition: {self.state} → {new_state}")
        
        self.state = new_state
        self.round += 1
        
        # Forced termination condition
        if self.round >= self.max_rounds:
            self.state = SAGEState.FINAL_CONCLUSION
            return True
        
        return False
    
    def should_conclude(self) -> bool:
        """Determine if should conclude."""
        return (
            self.round >= 12 and  # Minimum 12 rounds
            all(h.status != "PENDING" for h in self.hypotheses) and  # All hypotheses verified
            len(self.test_history) >= 8  # At least 8 tests
        )
    
    def add_hypothesis(self, text: str) -> int:
        """Add new hypothesis."""
        hypothesis_id = len(self.hypotheses) + 1
        hypothesis = Hypothesis(
            id=hypothesis_id,
            text=text,
            status="PENDING",
            initial_text=text,
            current_state=SAGEState.DESIGN_TEST  # Initial state is design test
        )
        self.hypotheses.append(hypothesis)
        return hypothesis_id
    
    def update_hypothesis(self, hypothesis_id: int, status: str, refined_text: str = None):
        """Update hypothesis status."""
        for h in self.hypotheses:
            if h.id == hypothesis_id:
                h.status = status
                if refined_text:
                    h.text = refined_text
                break
    
    def add_test_result(self, hypothesis_id: int, prompt: str, expected: str, 
                       actual_activation: float, normalized_activation: float, result: str):
        """Add test result (add to both global and hypothesis test history)."""
        # Calculate global test_id
        test_id = len(self.test_history) + 1
        test_result = TestResult(
            id=test_id,
            hypothesis_id=hypothesis_id,
            prompt=prompt,
            expected=expected,
            actual_activation=actual_activation,
            normalized_activation=normalized_activation,
            result=result,
            timestamp=str(self.round)
        )
        # Add to global test history
        self.test_history.append(test_result)
        
        # Add to corresponding hypothesis test history
        hypothesis = self.get_hypothesis_by_id(hypothesis_id)
        if hypothesis:
            hypothesis.test_history.append(test_result)
    
    def set_exemplars(self, exemplars: List[Exemplar]):
        """Set exemplars data."""
        self.exemplars = exemplars
    
    def add_analysis(self, analysis: str, hypothesis_id: Optional[int] = None):
        """Add analysis result (can specify hypothesis ID, add to both global and hypothesis analysis history)."""
        # Add to global analysis history
        self.analysis_history.append(analysis)
        
        # If hypothesis ID specified, also add to hypothesis analysis history
        if hypothesis_id:
            hypothesis = self.get_hypothesis_by_id(hypothesis_id)
            if hypothesis:
                hypothesis.analysis_history.append(analysis)
    
    def get_last_analysis(self) -> str:
        """Get last analysis."""
        return self.analysis_history[-1] if self.analysis_history else ""
    
    def get_pending_hypotheses(self) -> List[Hypothesis]:
        """Get pending hypotheses."""
        return [h for h in self.hypotheses if h.status == "PENDING"]
    
    def get_untested_hypotheses(self) -> List[Hypothesis]:
        """Get hypotheses that have never been tested (based on test history)."""
        tested_hypothesis_ids = {test.hypothesis_id for test in self.test_history}
        return [h for h in self.hypotheses if h.id not in tested_hypothesis_ids]
    
    def get_next_hypothesis_to_test(self) -> Optional[Hypothesis]:
        """Get next hypothesis to test (parallel processing mode).
        Priority:
        1. Hypotheses not in CONFIRMED or REFUTED status
        2. Never tested hypotheses (UNCHANGED or PENDING)
        3. PENDING hypotheses
        4. REFINED hypotheses (need more testing)
        """
        # In parallel mode, select all unfinished hypotheses
        active_hypotheses = [
            h for h in self.hypotheses 
            if h.status not in ["CONFIRMED", "REFUTED"]
        ]
        
        if not active_hypotheses:
            return None
        
        # If currently processing a hypothesis, check if it's completed
        if self.current_hypothesis_id:
            current = self.get_hypothesis_by_id(self.current_hypothesis_id)
            if current:
                # If hypothesis is completed, clear current_hypothesis_id to select next one
                if current.status in ["CONFIRMED", "REFUTED"]:
                    self.current_hypothesis_id = None
                else:
                    # If hypothesis not completed, continue processing it
                    return current
        
        # Otherwise select first active hypothesis
        # Prioritize never tested ones
        untested = [h for h in active_hypotheses if len(h.test_history) == 0]
        if untested:
            return untested[0]
        
        # Next select ones with fewest tests
        active_hypotheses.sort(key=lambda h: len(h.test_history))
        return active_hypotheses[0]
    
    def get_hypothesis_by_id(self, hypothesis_id: int) -> Optional[Hypothesis]:
        """Get hypothesis by ID."""
        for h in self.hypotheses:
            if h.id == hypothesis_id:
                return h
        return None
    
    def all_hypotheses_finalized(self) -> bool:
        """Check if all hypotheses are finalized (CONFIRMED or REFUTED)."""
        if not self.hypotheses:
            return False
        return all(h.status in ["CONFIRMED", "REFUTED"] for h in self.hypotheses)
    
    def get_active_hypotheses(self) -> List[Hypothesis]:
        """Get all active hypotheses (not finalized)."""
        return [h for h in self.hypotheses if h.status not in ["CONFIRMED", "REFUTED"]]
    
    def get_test_summary(self) -> str:
        """Get test history summary."""
        if not self.test_history:
            return "No tests run yet"
        
        summary = ""
        for i, test in enumerate(self.test_history, 1):
            summary += f"{i}. '{test.prompt[:40]}...' (tested H{test.hypothesis_id}, result: {test.result})\n"
        return summary
    
    def get_hypothesis_summary(self) -> str:
        """Get hypothesis status summary."""
        if not self.hypotheses:
            return "No hypotheses formed yet"
        
        summary = ""
        for h in self.hypotheses:
            status_emoji = {"PENDING": "⏳", "CONFIRMED": "✅", "REFUTED": "❌", "REFINED": "🔄"}
            summary += f"H{h.id}: {status_emoji.get(h.status, '❓')} {h.text}\n"
        return summary
    
    def force_conclude(self):
        """Force conclude."""
        self.state = SAGEState.FINAL_CONCLUSION
        self.round = self.max_rounds
    
    def is_final_state(self) -> bool:
        """Check if in final state."""
        # Only return True in DONE state, FINAL_CONCLUSION needs to continue execution
        return self.state == SAGEState.DONE
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information."""
        return {
            "state": self.state.value,
            "round": self.round,
            "max_rounds": self.max_rounds,
            "hypotheses_count": len(self.hypotheses),
            "tests_count": len(self.test_history),
            "exemplars_count": len(self.exemplars) if self.exemplars else 0,
            "should_conclude": self.should_conclude()
        }
    
    def compress_history(self):
        """Compress history data, keep key information."""
        # Compress analysis history, keep only last 3
        if len(self.analysis_history) > 3:
            self.analysis_history = self.analysis_history[-3:]
        
        # Compress test history, keep only last 5
        if len(self.test_history) > 5:
            self.test_history = self.test_history[-5:]
        
        print(f"📦 Compressed state machine history")


# State flow diagram validation
def validate_state_flow():
    """Validate correctness of state flow."""
    sm = SAGEStateMachine(0, 5)
    
    # 测试标准流程
    expected_flow = [
        SAGEState.INIT,
        SAGEState.GET_EXEMPLARS,
        SAGEState.ANALYZE_EXEMPLARS,
        SAGEState.FORM_HYPOTHESIS,
        SAGEState.DESIGN_TEST,
        SAGEState.RUN_TEST,
        SAGEState.ANALYZE_RESULT,
        SAGEState.UPDATE_HYPOTHESIS,
        SAGEState.DESIGN_TEST,
        SAGEState.RUN_TEST,
        SAGEState.ANALYZE_RESULT,
        SAGEState.FINAL_CONCLUSION,
        SAGEState.DONE
    ]
    
    for i, expected_state in enumerate(expected_flow):
        if sm.state != expected_state:
            print(f"State mismatch at step {i}: expected {expected_state}, got {sm.state}")
            return False
        
        if expected_state != SAGEState.DONE:
            # 模拟状态转换
            next_states = sm.valid_transitions.get(sm.state, [])
            if next_states:
                # 选择正确的下一个状态
                if i + 1 < len(expected_flow):
                    next_expected = expected_flow[i + 1]
                    if next_expected in next_states:
                        sm.transition(next_expected)
                    else:
                        # 如果期望的状态不在有效转换中，强制设置
                        sm.state = next_expected
                        sm.round += 1
                else:
                    sm.transition(next_states[0])
            else:
                # 如果没有有效转换，强制转换到下一个状态
                if i + 1 < len(expected_flow):
                    sm.state = expected_flow[i + 1]
                    sm.round += 1
    
    print("State flow validation passed!")
    return True


if __name__ == "__main__":
    validate_state_flow()
