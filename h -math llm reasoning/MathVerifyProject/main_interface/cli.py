"""
Enhanced Command Line Interface for MathVerifyProject
Features: Colored output, rich formatting, error classification, progress bars
"""

import argparse
import sys
import os
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Try to import rich for colored output, fallback to basic if not available
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

from core_verification import MathVerifier
from benchmark_evaluation import MathVEvaluator, MathVerseEvaluator
from ocr_input import HandwritingTranscriber


class MathVerifyCLI:
    """
    Enhanced command line interface with colored output and rich formatting.
    """
    
    def __init__(self):
        """Initialize CLI with console."""
        self.verifier = MathVerifier()
        self.mathv_evaluator = None
        self.mathverse_evaluator = None
        self.transcriber = None
        self.console = Console() if RICH_AVAILABLE else None
    
    def classify_error(self, gold_parsed: list, pred_parsed: list) -> str:
        """Classify the type of error in the prediction."""
        if not gold_parsed:
            return "Parse Error: Could not parse gold answer"
        if not pred_parsed:
            return "Parse Error: Could not parse prediction"
        return "Format Mismatch: Answers may be equivalent but in different formats"
    
    def verify_answer_detailed(self, gold: str, prediction: str) -> Dict[str, Any]:
        """
        Verify answer with detailed information.
        
        Args:
            gold: Gold answer string
            prediction: Prediction string
            
        Returns:
            Dictionary with verification results
        """
        try:
            # Parse both expressions
            gold_parsed = self.verifier.parse_expression(gold, is_gold=True)
            pred_parsed = self.verifier.parse_expression(prediction, is_gold=False)
            
            # Verify
            is_correct = self.verifier.verify_answer(gold, prediction)
            
            # Classify error if incorrect
            error_type = None
            if not is_correct:
                error_type = self.classify_error(gold_parsed, pred_parsed)
            
            return {
                'valid': is_correct,
                'gold': gold,
                'prediction': prediction,
                'gold_parsed': gold_parsed[0] if gold_parsed else None,
                'pred_parsed': pred_parsed[0] if pred_parsed else None,
                'error_type': error_type,
                'details': f"Gold parsed as: {gold_parsed[0] if gold_parsed else 'N/A'}, "
                          f"Prediction parsed as: {pred_parsed[0] if pred_parsed else 'N/A'}"
            }
        except Exception as e:
            return {
                'valid': False,
                'gold': gold,
                'prediction': prediction,
                'error_type': f"Error: {str(e)}",
                'details': f"An error occurred: {str(e)}"
            }
    
    def verify_command(self, gold: str, prediction: str):
        """
        Verify a single answer with colored output.
        
        Args:
            gold: Gold answer string
            prediction: Prediction string
        """
        result = self.verify_answer_detailed(gold, prediction)
        
        if RICH_AVAILABLE:
            self._display_verification_rich(result)
        else:
            self._display_verification_basic(result)
        
        return result
    
    def _display_verification_rich(self, result: Dict[str, Any]):
        """Display verification result with rich formatting."""
        # Status color and icon
        status_color = "green" if result['valid'] else "red"
        status_icon = "✓" if result['valid'] else "✗"
        status_text = "CORRECT" if result['valid'] else "INCORRECT"
        
        # Create main panel
        status_text_rich = Text(f"{status_icon} {status_text}", style=f"bold {status_color}")
        
        # Create info table
        info_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        info_table.add_row("[bold]Gold Answer:[/bold]", f"[cyan]{result['gold']}[/cyan]")
        info_table.add_row("[bold]Prediction:[/bold]", f"[cyan]{result['prediction']}[/cyan]")
        
        if result.get('gold_parsed'):
            info_table.add_row("[bold]Gold Parsed:[/bold]", f"[dim]{result['gold_parsed']}[/dim]")
        if result.get('pred_parsed'):
            info_table.add_row("[bold]Prediction Parsed:[/bold]", f"[dim]{result['pred_parsed']}[/dim]")
        
        info_table.add_row("[bold]Details:[/bold]", f"[yellow]{result.get('details', 'N/A')}[/yellow]")
        
        if result.get('error_type'):
            info_table.add_row(
                "[bold red]Error Classification:[/bold red]",
                f"[red]{result['error_type']}[/red]"
            )
        
        # Create panel
        panel_content = f"{status_text_rich}\n\n{info_table}"
        panel = Panel(
            panel_content,
            title="[bold]Verification Result[/bold]",
            border_style=status_color,
            padding=(1, 2)
        )
        
        self.console.print(panel)
        self.console.print()  # Empty line
    
    def _display_verification_basic(self, result: Dict[str, Any]):
        """Display verification result with basic formatting."""
        status = "✓ CORRECT" if result['valid'] else "✗ INCORRECT"
        print(f"\n{'='*60}")
        print(f"Verification Result: {status}")
        print(f"{'='*60}")
        print(f"Gold Answer: {result['gold']}")
        print(f"Prediction: {result['prediction']}")
        if result.get('gold_parsed'):
            print(f"Gold Parsed: {result['gold_parsed']}")
        if result.get('pred_parsed'):
            print(f"Prediction Parsed: {result['pred_parsed']}")
        print(f"Details: {result.get('details', 'N/A')}")
        if result.get('error_type'):
            print(f"Error Classification: {result['error_type']}")
        print(f"{'='*60}\n")
    
    def verify_batch_command(self, gold_file: str, pred_file: str):
        """
        Verify a batch of answers from files with progress bar.
        
        Args:
            gold_file: Path to file with gold answers (one per line)
            pred_file: Path to file with predictions (one per line)
        """
        # Read files
        with open(gold_file, 'r', encoding='utf-8') as f:
            gold_answers = [line.strip() for line in f if line.strip()]
        
        with open(pred_file, 'r', encoding='utf-8') as f:
            predictions = [line.strip() for line in f if line.strip()]
        
        if len(gold_answers) != len(predictions):
            error_msg = f"Error: Files have different lengths ({len(gold_answers)} vs {len(predictions)})"
            if RICH_AVAILABLE:
                self.console.print(f"[bold red]{error_msg}[/bold red]")
            else:
                print(error_msg)
            return []
        
        # Verify batch with progress
        results = []
        error_types = []
        
        if RICH_AVAILABLE and TQDM_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("Verifying batch...", total=len(gold_answers))
                for gold, pred in zip(gold_answers, predictions):
                    result = self.verifier.verify_answer(gold, pred)
                    results.append(result)
                    
                    if not result:
                        gold_parsed = self.verifier.parse_expression(gold, is_gold=True)
                        pred_parsed = self.verifier.parse_expression(pred, is_gold=False)
                        error_types.append(self.classify_error(gold_parsed, pred_parsed))
                    else:
                        error_types.append(None)
                    
                    progress.update(task, advance=1)
        else:
            # Basic progress with tqdm or simple iteration
            if TQDM_AVAILABLE:
                iterator = tqdm(zip(gold_answers, predictions), total=len(gold_answers), desc="Verifying")
            else:
                iterator = zip(gold_answers, predictions)
            
            for gold, pred in iterator:
                result = self.verifier.verify_answer(gold, pred)
                results.append(result)
                
                if not result:
                    gold_parsed = self.verifier.parse_expression(gold, is_gold=True)
                    pred_parsed = self.verifier.parse_expression(pred, is_gold=False)
                    error_types.append(self.classify_error(gold_parsed, pred_parsed))
                else:
                    error_types.append(None)
        
        # Display results
        correct = sum(results)
        total = len(results)
        accuracy = 100 * correct / total if total > 0 else 0
        
        if RICH_AVAILABLE:
            self._display_batch_results_rich(gold_answers, predictions, results, error_types, correct, total, accuracy)
        else:
            self._display_batch_results_basic(gold_answers, predictions, results, error_types, correct, total, accuracy)
        
        return results
    
    def _display_batch_results_rich(self, gold_answers, predictions, results, error_types, correct, total, accuracy):
        """Display batch results with rich formatting."""
        # Summary panel
        summary_text = f"[bold]Correct:[/bold] [green]{correct}[/green] / [bold]{total}[/bold] "
        summary_text += f"([bold green]{accuracy:.2f}%[/bold green])"
        
        summary_panel = Panel(
            summary_text,
            title="[bold]Batch Verification Summary[/bold]",
            border_style="blue"
        )
        self.console.print(summary_panel)
        
        # Results table
        table = Table(title="Batch Results", show_header=True, header_style="bold magenta")
        table.add_column("Index", style="dim", width=6)
        table.add_column("Gold", style="cyan", width=20)
        table.add_column("Prediction", style="cyan", width=20)
        table.add_column("Result", width=10)
        table.add_column("Error Type", style="red", width=30)
        
        for i, (gold, pred, result, error) in enumerate(zip(gold_answers, predictions, results, error_types), 1):
            result_text = "[green]✓ CORRECT[/green]" if result else "[red]✗ INCORRECT[/red]"
            error_text = error if error else "[dim]None[/dim]"
            table.add_row(str(i), gold[:18], pred[:18], result_text, error_text)
        
        self.console.print(table)
        self.console.print()
    
    def _display_batch_results_basic(self, gold_answers, predictions, results, error_types, correct, total, accuracy):
        """Display batch results with basic formatting."""
        print(f"\n{'='*60}")
        print(f"Batch Verification Results")
        print(f"{'='*60}")
        print(f"Correct: {correct} / {total} ({accuracy:.2f}%)")
        print(f"{'='*60}\n")
        
        for i, (gold, pred, result, error) in enumerate(zip(gold_answers, predictions, results, error_types), 1):
            status = "✓ CORRECT" if result else "✗ INCORRECT"
            print(f"{i}. Gold: {gold} | Pred: {pred} | {status}")
            if error:
                print(f"   Error: {error}")
    
    def transcribe_command(self, inkml_path: str, model_path: Optional[str] = None):
        """
        Transcribe an InkML file to LaTeX.
        
        Args:
            inkml_path: Path to InkML file
            model_path: Path to model checkpoint (optional)
        """
        if model_path:
            self.transcriber = HandwritingTranscriber(model_path=model_path)
        elif self.transcriber is None:
            # Try to find default model
            default_model = os.path.join(
                os.path.dirname(__file__), '..', 
                'handwritten-math-transcription', 'model', 'model_best_0.pth'
            )
            if os.path.exists(default_model):
                self.transcriber = HandwritingTranscriber(model_path=default_model)
            else:
                error_msg = "Error: No model specified and no default model found"
                if RICH_AVAILABLE:
                    self.console.print(f"[bold red]{error_msg}[/bold red]")
                else:
                    print(error_msg)
                return
        
        if RICH_AVAILABLE:
            self.console.print("[yellow]Transcribing InkML file...[/yellow]")
        
        latex, gt = self.transcriber.transcribe_inkml(inkml_path)
        
        if RICH_AVAILABLE:
            panel_content = f"[bold]Predicted LaTeX:[/bold]\n[cyan]{latex}[/cyan]"
            if gt:
                panel_content += f"\n\n[bold]Ground Truth:[/bold]\n[green]{gt}[/green]"
            
            panel = Panel(
                panel_content,
                title="[bold]Transcription Result[/bold]",
                border_style="blue"
            )
            self.console.print(panel)
        else:
            print(f"\nPredicted LaTeX: {latex}")
            if gt:
                print(f"Ground Truth: {gt}")
    
    @staticmethod
    def main():
        """Main CLI entry point."""
        parser = argparse.ArgumentParser(
            description='Mathematical Reasoning Verification System - CLI',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python main.py --mode cli verify --gold "1/2" --pred "0.5"
  python main.py --mode cli batch-verify --gold-file gold.txt --pred-file pred.txt
  python main.py --mode cli transcribe --inkml file.inkml --model model.pth
            """
        )
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Verify command
        verify_parser = subparsers.add_parser('verify', help='Verify a single answer')
        verify_parser.add_argument('--gold', required=True, help='Gold answer')
        verify_parser.add_argument('--pred', required=True, help='Prediction')
        
        # Batch verify command
        batch_parser = subparsers.add_parser('batch-verify', help='Verify batch of answers')
        batch_parser.add_argument('--gold-file', required=True, help='File with gold answers (one per line)')
        batch_parser.add_argument('--pred-file', required=True, help='File with predictions (one per line)')
        
        # Transcribe command
        transcribe_parser = subparsers.add_parser('transcribe', help='Transcribe InkML to LaTeX')
        transcribe_parser.add_argument('--inkml', required=True, help='Path to InkML file')
        transcribe_parser.add_argument('--model', help='Path to model checkpoint (optional)')
        
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return
        
        cli = MathVerifyCLI()
        
        if args.command == 'verify':
            cli.verify_command(args.gold, args.pred)
        elif args.command == 'batch-verify':
            cli.verify_batch_command(args.gold_file, args.pred_file)
        elif args.command == 'transcribe':
            cli.transcribe_command(args.inkml, args.model)
        else:
            parser.print_help()


if __name__ == '__main__':
    MathVerifyCLI.main()

