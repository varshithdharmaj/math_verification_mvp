"""Error location tracking and root cause analysis."""

from typing import Dict, List, Optional, Tuple
from src.utils.error_taxonomy import get_error_info, ERROR_TAXONOMY


class ErrorLocationTracker:
    """Tracks error locations across multiple steps."""
    
    def __init__(self):
        """Initialize tracker."""
        self.step_errors = []
        self.error_chains = []
    
    def track_step_error(
        self,
        step_idx: int,
        step_text: str,
        error_type: str,
        confidence: float,
        verifier_name: str
    ):
        """Track an error in a specific step.
        
        Args:
            step_idx: Index of the step (0-based)
            step_text: Text of the step
            error_type: Type of error detected
            confidence: Confidence in error detection
            verifier_name: Which verifier detected it
        """
        self.step_errors.append({
            'step_idx': step_idx,
            'step_text': step_text,
            'error_type': error_type,
            'confidence': confidence,
            'verifier': verifier_name,
            'severity': get_error_info(error_type).get('severity')
        })
    
    def find_error_ranges(self) -> List[Dict]:
        """Find ranges of consecutive steps with errors.
        
        Returns:
            List of dicts with 'start', 'end', 'errors', 'severity'
        """
        if not self.step_errors:
            return []
        
        # Sort by step index
        sorted_errors = sorted(self.step_errors, key=lambda x: x['step_idx'])
        
        ranges = []
        current_range = None
        
        for error in sorted_errors:
            step_idx = error['step_idx']
            
            if current_range is None:
                current_range = {
                    'start': step_idx,
                    'end': step_idx,
                    'errors': [error],
                    'severity': error.get('severity')
                }
            elif step_idx == current_range['end'] + 1:
                # Consecutive step
                current_range['end'] = step_idx
                current_range['errors'].append(error)
                # Update severity to highest
                if error.get('severity'):
                    current_range['severity'] = error.get('severity')
            else:
                # Gap found, save current range
                ranges.append(current_range)
                current_range = {
                    'start': step_idx,
                    'end': step_idx,
                    'errors': [error],
                    'severity': error.get('severity')
                }
        
        if current_range:
            ranges.append(current_range)
        
        return ranges
    
    def trace_root_cause(self, steps: List[str]) -> Optional[Dict]:
        """Trace back to root cause of errors.
        
        Args:
            steps: All steps in the solution
            
        Returns:
            Dict with root cause analysis
        """
        if not self.step_errors:
            return None
        
        # Find first error
        first_error = min(self.step_errors, key=lambda x: x['step_idx'])
        
        # Check if earlier steps might have caused it
        root_cause_step = first_error['step_idx']
        root_cause_type = first_error['error_type']
        
        # Analyze if this is a cascading error
        cascading = False
        if first_error['step_idx'] > 0:
            # Check if previous step had an error that could cascade
            prev_errors = [e for e in self.step_errors if e['step_idx'] < first_error['step_idx']]
            if prev_errors:
                cascading = True
                root_cause_step = min(prev_errors, key=lambda x: x['step_idx'])['step_idx']
        
        return {
            'root_cause_step': root_cause_step,
            'root_cause_type': root_cause_type,
            'first_detected_step': first_error['step_idx'],
            'is_cascading': cascading,
            'total_affected_steps': len(self.step_errors),
            'explanation': self._generate_root_cause_explanation(
                root_cause_step, root_cause_type, cascading, steps
            )
        }
    
    def _generate_root_cause_explanation(
        self,
        root_step: int,
        error_type: str,
        cascading: bool,
        steps: List[str]
    ) -> str:
        """Generate explanation of root cause."""
        if cascading:
            return (
                f"The error originated in Step {root_step + 1} and cascaded to later steps. "
                f"The root cause is a {error_type} in: '{steps[root_step][:50]}...'"
            )
        else:
            return (
                f"The error first appears in Step {root_step + 1}. "
                f"Error type: {error_type}. "
                f"Step: '{steps[root_step][:50]}...'"
            )
    
    def get_error_summary(self) -> Dict:
        """Get summary of all errors.
        
        Returns:
            Dict with error summary
        """
        if not self.step_errors:
            return {
                'total_errors': 0,
                'error_types': {},
                'affected_steps': [],
                'severity_distribution': {}
            }
        
        error_types = {}
        affected_steps = set()
        severity_dist = {}
        
        for error in self.step_errors:
            error_type = error['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
            affected_steps.add(error['step_idx'])
            severity = str(error.get('severity', 'unknown'))
            severity_dist[severity] = severity_dist.get(severity, 0) + 1
        
        return {
            'total_errors': len(self.step_errors),
            'error_types': error_types,
            'affected_steps': sorted(list(affected_steps)),
            'severity_distribution': severity_dist,
            'error_ranges': self.find_error_ranges()
        }

