"""
Core Verification Module - Wrapper for Math-Verify
Provides interface for mathematical expression verification
"""

import sys
import os

# Add Math-Verify to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Math-Verify', 'src'))

try:
    from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig
except ImportError:
    # Fallback if math-verify is installed as package
    try:
        from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig
    except ImportError:
        raise ImportError(
            "Math-Verify not found. Please install it: pip install math-verify[antlr4_13_2]"
        )


class MathVerifier:
    """
    Core verification module for mathematical expressions.
    Wraps Math-Verify functionality for use in the integrated system.
    """
    
    def __init__(self, 
                 gold_extraction_config=None,
                 pred_extraction_config=None,
                 float_rounding=6,
                 numeric_precision=15,
                 strict=True):
        """
        Initialize the Math Verifier.
        
        Args:
            gold_extraction_config: Configuration for extracting gold answers
            pred_extraction_config: Configuration for extracting predictions
            float_rounding: Number of decimal places for float rounding
            numeric_precision: Precision for numeric comparisons
            strict: Whether to use strict comparison mode
        """
        self.gold_extraction_config = gold_extraction_config or [ExprExtractionConfig()]
        self.pred_extraction_config = pred_extraction_config or [
            LatexExtractionConfig(),
            ExprExtractionConfig()
        ]
        self.float_rounding = float_rounding
        self.numeric_precision = numeric_precision
        self.strict = strict
    
    def parse_expression(self, expression: str, is_gold: bool = False):
        """
        Parse a mathematical expression from text.
        
        Args:
            expression: String containing mathematical expression
            is_gold: Whether this is a gold answer (uses gold extraction config)
            
        Returns:
            Parsed expression(s) as SymPy objects or strings
        """
        config = self.gold_extraction_config if is_gold else self.pred_extraction_config
        try:
            parsed = parse(expression, extraction_config=config)
            return parsed if parsed else []
        except Exception as e:
            print(f"Error parsing expression '{expression}': {e}")
            return []
    
    def verify_answer(self, gold: str, prediction: str, return_details: bool = False) -> bool | dict:
        """
        Verify if a prediction matches the gold answer.
        
        Args:
            gold: Gold/correct answer string
            prediction: Model prediction string
            return_details: If True, returns detailed dict instead of just boolean
            
        Returns:
            If return_details=False: bool (True if prediction matches gold, False otherwise)
            If return_details=True: dict with keys:
                - 'valid': bool - verification result
                - 'gold': str - original gold answer
                - 'prediction': str - original prediction
                - 'gold_parsed': parsed gold expression or None
                - 'pred_parsed': parsed prediction expression or None
                - 'error_type': str or None - error classification if incorrect
                - 'details': str - additional details
        """
        try:
            # Parse both expressions
            gold_parsed = self.parse_expression(gold, is_gold=True)
            pred_parsed = self.parse_expression(prediction, is_gold=False)
            
            # Handle empty parse results
            gold_parse_success = gold_parsed and len(gold_parsed) > 0
            pred_parse_success = pred_parsed and len(pred_parsed) > 0
            
            if not gold_parse_success:
                if not return_details:
                    print(f"Warning: Could not parse gold answer: {gold}")
                error_type = "Parse Error: Could not parse gold answer"
            elif not pred_parse_success:
                if not return_details:
                    print(f"Warning: Could not parse prediction: {prediction}")
                error_type = "Parse Error: Could not parse prediction"
            else:
                error_type = None
            
            # If parsing failed, return early
            if not gold_parse_success or not pred_parse_success:
                if return_details:
                    return {
                        'valid': False,
                        'gold': gold,
                        'prediction': prediction,
                        'gold_parsed': None,
                        'pred_parsed': None,
                        'error_type': error_type,
                        'details': f"Gold parsed: {gold_parsed[0] if gold_parsed else 'N/A'}, "
                                  f"Prediction parsed: {pred_parsed[0] if pred_parsed else 'N/A'}"
                    }
                return False
            
            # Use first parsed result for comparison
            gold_expr = gold_parsed[0] if isinstance(gold_parsed, list) else gold_parsed
            pred_expr = pred_parsed[0] if isinstance(pred_parsed, list) else pred_parsed
            
            # Verify using Math-Verify
            is_valid = verify(
                gold_expr,
                pred_expr,
                float_rounding=self.float_rounding,
                numeric_precision=self.numeric_precision,
                strict=self.strict
            )
            
            # Classify error if incorrect
            error_type = None
            if not is_valid:
                error_type = self._classify_error(gold_parsed, pred_parsed)
            
            if return_details:
                return {
                    'valid': is_valid,
                    'gold': gold,
                    'prediction': prediction,
                    'gold_parsed': gold_expr,
                    'pred_parsed': pred_expr,
                    'error_type': error_type,
                    'details': f"Gold parsed as: {gold_expr}, Prediction parsed as: {pred_expr}"
                }
            
            return is_valid
        except Exception as e:
            error_msg = f"Error verifying answer: {e}"
            if not return_details:
                print(error_msg)
                import traceback
                traceback.print_exc()
            
            if return_details:
                return {
                    'valid': False,
                    'gold': gold,
                    'prediction': prediction,
                    'error_type': f"Error: {str(e)}",
                    'details': error_msg
                }
            return False
    
    def _classify_error(self, gold_parsed: list, pred_parsed: list) -> str:
        """
        Classify the type of error in the prediction.
        
        Args:
            gold_parsed: Parsed gold answer
            pred_parsed: Parsed prediction
            
        Returns:
            Error classification string
        """
        if not gold_parsed:
            return "Parse Error: Could not parse gold answer"
        if not pred_parsed:
            return "Parse Error: Could not parse prediction"
        return "Format Mismatch: Answers may be equivalent but in different formats"
    
    def verify_batch(self, gold_answers: list, predictions: list, return_details: bool = False) -> list:
        """
        Verify a batch of answers.
        
        Args:
            gold_answers: List of gold answer strings
            predictions: List of prediction strings
            return_details: If True, returns list of detailed dicts instead of booleans
            
        Returns:
            If return_details=False: List of boolean verification results
            If return_details=True: List of dicts with detailed results (same format as verify_answer)
        """
        if len(gold_answers) != len(predictions):
            raise ValueError("Gold answers and predictions must have same length")
        
        results = []
        for gold, pred in zip(gold_answers, predictions):
            result = self.verify_answer(gold, pred, return_details=return_details)
            results.append(result)
        
        return results

