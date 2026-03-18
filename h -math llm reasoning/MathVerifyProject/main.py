"""
Main entry point for MathVerifyProject
Integrates all modules: verification, evaluation, and OCR
"""

import sys
import os
import argparse
from typing import Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from core_verification import MathVerifier
from benchmark_evaluation import MathVEvaluator, MathVerseEvaluator
from ocr_input import HandwritingTranscriber
from main_interface import MathVerifyCLI
try:
    from main_interface import create_gradio_app
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False


class MathVerifyPipeline:
    """
    Main pipeline integrating all modules.
    """
    
    def __init__(self, 
                 verifier_config: dict = None,
                 ocr_model_path: str = None,
                 api_key: str = None):
        """
        Initialize the integrated pipeline.
        
        Args:
            verifier_config: Configuration for MathVerifier
            ocr_model_path: Path to OCR model checkpoint
            api_key: API key for benchmark evaluation (optional)
        """
        # Initialize core verification
        self.verifier = MathVerifier(**(verifier_config or {}))
        
        # Initialize benchmark evaluators
        self.mathv_evaluator = MathVEvaluator(api_key=api_key) if api_key else None
        self.mathverse_evaluator = MathVerseEvaluator(api_key=api_key) if api_key else None
        
        # Initialize OCR transcriber
        self.transcriber = None
        if ocr_model_path and os.path.exists(ocr_model_path):
            self.transcriber = HandwritingTranscriber(model_path=ocr_model_path)
    
    def process_math_problem(self, 
                            problem_text: str,
                            model_answer: str,
                            gold_answer: str,
                            use_ocr: bool = False,
                            ocr_input_path: str = None) -> dict:
        """
        Process a mathematical problem through the full pipeline.
        
        Args:
            problem_text: The math problem text
            model_answer: Model's answer
            gold_answer: Correct answer
            use_ocr: Whether to use OCR for input processing
            ocr_input_path: Path to OCR input (InkML or image)
            
        Returns:
            Dictionary with processing results
        """
        results = {
            'problem': problem_text,
            'model_answer': model_answer,
            'gold_answer': gold_answer,
            'verification': None,
            'ocr_transcription': None
        }
        
        # Step 1: OCR processing (if enabled)
        if use_ocr and ocr_input_path and self.transcriber:
            try:
                if ocr_input_path.endswith('.inkml'):
                    latex, gt = self.transcriber.transcribe_inkml(ocr_input_path)
                    results['ocr_transcription'] = latex
                else:
                    latex = self.transcriber.transcribe_image(ocr_input_path)
                    results['ocr_transcription'] = latex
            except Exception as e:
                results['ocr_error'] = str(e)
        
        # Step 2: Verification
        try:
            is_correct = self.verifier.verify_answer(gold_answer, model_answer)
            results['verification'] = is_correct
        except Exception as e:
            results['verification_error'] = str(e)
        
        return results
    
    def evaluate_benchmark(self, 
                         benchmark: str,
                         output_file: str,
                         model_outputs: list = None) -> dict:
        """
        Evaluate on a benchmark dataset.
        
        Args:
            benchmark: Benchmark name ('mathv' or 'mathverse')
            output_file: Path to save evaluation results
            model_outputs: List of model outputs (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if benchmark.lower() == 'mathv':
            if not self.mathv_evaluator:
                return {'error': 'MATH-V evaluator not initialized (API key required)'}
            return self.mathv_evaluator.evaluate_model_outputs(output_file)
        
        elif benchmark.lower() == 'mathverse':
            if not self.mathverse_evaluator:
                return {'error': 'MathVerse evaluator not initialized (API key required)'}
            # MathVerse evaluation requires two steps
            return {'error': 'MathVerse evaluation requires two-step process (extract + score)'}
        
        else:
            return {'error': f'Unknown benchmark: {benchmark}'}


def main():
    """Main entry point."""
    # Check if CLI mode is requested - if so, handle it separately
    if '--mode' in sys.argv:
        mode_index = sys.argv.index('--mode')
        if mode_index + 1 < len(sys.argv) and sys.argv[mode_index + 1] == 'cli':
            # Extract CLI arguments (everything after 'cli')
            cli_args = ['main.py'] + sys.argv[mode_index + 2:]
            sys.argv = cli_args
            MathVerifyCLI.main()
            return
    
    parser = argparse.ArgumentParser(
        description='Mathematical Reasoning Verification System - Integrated Pipeline'
    )
    parser.add_argument('--mode', 
                       choices=['cli', 'gradio', 'pipeline'],
                       default='gradio',
                       help='Interface mode')
    parser.add_argument('--gold', help='Gold answer for verification')
    parser.add_argument('--pred', help='Prediction to verify')
    parser.add_argument('--ocr-model', help='Path to OCR model')
    parser.add_argument('--api-key', help='API key for benchmark evaluation')
    parser.add_argument('--port', type=int, default=7860, help='Port for Gradio (default: 7860)')
    
    args = parser.parse_args()
    
    if args.mode == 'gradio':
        # Launch Gradio interface
        if not GRADIO_AVAILABLE:
            print("Error: Gradio is not installed. Please install it with: pip install gradio")
            print("Alternatively, use --mode cli for command-line interface")
            sys.exit(1)
        app = create_gradio_app()
        app.launch(server_port=args.port, share=False)
    
    elif args.mode == 'cli':
        # This should not be reached due to early check above, but keep as fallback
        cli_args = ['main.py'] + [a for a in sys.argv[1:] if a not in ['--mode', 'cli']]
        sys.argv = cli_args
        MathVerifyCLI.main()
    
    elif args.mode == 'pipeline':
        # Direct pipeline usage
        pipeline = MathVerifyPipeline(
            ocr_model_path=args.ocr_model,
            api_key=args.api_key
        )
        
        if args.gold and args.pred:
            result = pipeline.process_math_problem(
                problem_text="",
                model_answer=args.pred,
                gold_answer=args.gold
            )
            print(f"Verification: {result['verification']}")
        else:
            print("Please provide --gold and --pred for pipeline mode")


if __name__ == '__main__':
    main()

