"""
Gradio demo interface for MathVerify-Integrated system.

Provides a user-friendly web interface for mathematical problem solving
with real-time verification and error classification.
"""

import gradio as gr
import json
from src.pipeline import MathVerifyPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize pipeline (lazy loading)
pipeline = None

def get_pipeline():
    """Get or create pipeline instance."""
    global pipeline
    if pipeline is None:
        logger.info("Initializing pipeline...")
        pipeline = MathVerifyPipeline()
        logger.info("Pipeline ready")
    return pipeline

# Example problems
EXAMPLE_PROBLEMS = [
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muns for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
    "Solve for x: 2x + 3 = 7",
    "Calculate: (1/2) + (1/4) = ?",
    "If a train travels 60 miles per hour for 2.5 hours, how far does it travel?",
    "Simplify: 3x^2 + 2x^2 - 5x + 3x = ?"
]

def format_step_output(step: dict) -> str:
    """Format a single step for display."""
    step_num = step.get("number", "?")
    content = step.get("content", "")
    is_valid = step.get("is_valid", True)
    error_type = step.get("error_type")
    correction = step.get("correction")
    
    # Status icon
    status = "✅" if is_valid else "❌"
    
    # Build output
    output = f"**Step {step_num}** {status}\n"
    output += f"{content}\n"
    
    if not is_valid and error_type:
        output += f"\n⚠️ **Error**: {error_type}\n"
        if correction:
            output += f"💡 **Suggestion**: {correction}\n"
    
    return output

def solve_problem(problem_text: str) -> tuple:
    """
    Process a mathematical problem through the pipeline.
    
    Args:
        problem_text: Problem statement
        
    Returns:
        Tuple of (formatted_output, json_output)
    """
    if not problem_text or not problem_text.strip():
        return "Please enter a problem.", ""
    
    try:
        # Get pipeline
        pipe = get_pipeline()
        
        # Process problem
        result = pipe.process_problem(problem_text)
        
        # Format output
        output_lines = []
        output_lines.append("## Problem\n")
        output_lines.append(f"{result['problem']}\n\n")
        
        output_lines.append("## Solution Steps\n")
        steps = result['solution'].get('steps', [])
        
        if not steps:
            output_lines.append("No steps generated.\n\n")
        else:
            for step in steps:
                output_lines.append(format_step_output(step))
                output_lines.append("\n")
        
        # Final answer
        output_lines.append("## Final Answer\n")
        final_answer = result.get('final_answer', 'Not available')
        confidence = result.get('confidence', 0.0)
        
        output_lines.append(f"**Answer**: {final_answer}\n\n")
        output_lines.append(f"**Confidence**: {confidence * 100:.1f}%\n\n")
        
        # Error summary
        errors = result.get('errors', {})
        total_errors = errors.get('total_errors', 0)
        
        if total_errors > 0:
            output_lines.append("## Error Summary\n")
            output_lines.append(f"**Total Errors**: {total_errors}\n\n")
            
            by_type = errors.get('by_type', {})
            if by_type:
                output_lines.append("**Errors by Type**:\n")
                for error_type, count in by_type.items():
                    percentage = errors.get('percentage', {}).get(error_type, 0)
                    output_lines.append(f"- {error_type}: {count} ({percentage}%)\n")
        else:
            output_lines.append("## Verification Status\n")
            output_lines.append("✅ All steps verified successfully!\n")
        
        formatted_output = "".join(output_lines)
        
        # JSON output for export
        json_output = json.dumps(result, indent=2)
        
        return formatted_output, json_output
        
    except Exception as e:
        error_msg = f"Error processing problem: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return f"❌ {error_msg}", ""

def create_demo():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(
        title="MathVerify-Integrated",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown(
            """
            # 🧮 MathVerify-Integrated
            ## End-to-End Mathematical Reasoning with Real-Time Verification
            
            Enter a mathematical problem below and get step-by-step solutions with automatic verification.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                problem_input = gr.Textbox(
                    label="Mathematical Problem",
                    placeholder="Enter your problem here...",
                    lines=5,
                    value=EXAMPLE_PROBLEMS[0]
                )
                
                solve_btn = gr.Button("Solve", variant="primary", size="lg")
                
                with gr.Row():
                    example_btn = gr.Button("Load Example", variant="secondary")
                    export_btn = gr.Button("Export Results", variant="secondary")
            
            with gr.Column(scale=1):
                gr.Markdown("### Example Problems")
                example_dropdown = gr.Dropdown(
                    choices=[
                        "Word Problem (Janet's Ducks)",
                        "Algebra (Solve for x)",
                        "Fractions",
                        "Distance Problem",
                        "Polynomial Simplification"
                    ],
                    label="Select Example",
                    value="Word Problem (Janet's Ducks)"
                )
        
        with gr.Row():
            output_display = gr.Markdown(
                label="Solution",
                value="Enter a problem and click 'Solve' to see the solution."
            )
        
        with gr.Row():
            json_output = gr.Textbox(
                label="JSON Export",
                lines=10,
                visible=False
            )
        
        # Event handlers
        solve_btn.click(
            fn=solve_problem,
            inputs=problem_input,
            outputs=[output_display, json_output]
        )
        
        def load_example(example_name: str):
            """Load example problem based on selection."""
            example_map = {
                "Word Problem (Janet's Ducks)": EXAMPLE_PROBLEMS[0],
                "Algebra (Solve for x)": EXAMPLE_PROBLEMS[1],
                "Fractions": EXAMPLE_PROBLEMS[2],
                "Distance Problem": EXAMPLE_PROBLEMS[3],
                "Polynomial Simplification": EXAMPLE_PROBLEMS[4]
            }
            return example_map.get(example_name, EXAMPLE_PROBLEMS[0])
        
        example_dropdown.change(
            fn=load_example,
            inputs=example_dropdown,
            outputs=problem_input
        )
        
        example_btn.click(
            fn=lambda: load_example(example_dropdown.value),
            outputs=problem_input
        )
        
        def toggle_json():
            """Toggle JSON output visibility."""
            return gr.update(visible=True)
        
        export_btn.click(
            fn=toggle_json,
            outputs=json_output
        )
        
        gr.Markdown(
            """
            ---
            ### Features
            - ✅ Real-time step-by-step verification
            - 🔍 Automatic error detection and classification
            - 💡 Correction suggestions
            - 📊 Confidence scoring
            - 📥 Export results as JSON
            """
        )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        share=True,  # Create public URL
        server_name="0.0.0.0",
        server_port=7860
    )

