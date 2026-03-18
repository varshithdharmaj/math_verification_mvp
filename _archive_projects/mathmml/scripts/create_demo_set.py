"""Create 5-question demo set as specified in PRD."""

import json
from pathlib import Path

# PRD-specified demo questions with known LLM failures
DEMO_QUESTIONS = [
    {
        "problem": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "steps_correct": [
            "Natalia sold 48/2 = 24 clips in May.",
            "Natalia sold 48+24 = 72 clips altogether in April and May."
        ],
        "steps_incorrect": [
            "Natalia sold 48/2 = 25 clips in May.",  # Arithmetic error
            "Natalia sold 48+24 = 72 clips altogether in April and May."
        ],
        "expected_error": "arithmetic_error",
        "error_location": 0
    },
    {
        "problem": "A store has 15 apples. They sell 8 apples. How many apples are left?",
        "steps_correct": [
            "The store has 15 apples.",
            "They sell 8 apples.",
            "15 - 8 = 7 apples left."
        ],
        "steps_incorrect": [
            "The store has 15 apples.",
            "They sell 8 apples.",
            "15 - 8 = 6 apples left."  # Arithmetic error: should be 7
        ],
        "expected_error": "arithmetic_error",
        "error_location": 2
    },
    {
        "problem": "Calculate 5 + 3",
        "steps_correct": [
            "5 + 3 = 8"
        ],
        "steps_incorrect": [
            "5 + 3 = 9"  # Arithmetic error
        ],
        "expected_error": "arithmetic_error",
        "error_location": 0
    },
    {
        "problem": "Add 10 and 5, then subtract 3",
        "steps_correct": [
            "10 + 5 = 15",
            "15 - 3 = 12"
        ],
        "steps_incorrect": [
            "10 + 5 = 15",
            "15 - 3 = 13"  # Arithmetic error
        ],
        "expected_error": "arithmetic_error",
        "error_location": 1
    },
    {
        "problem": "Multiply 6 by 4",
        "steps_correct": [
            "6 × 4 = 24"
        ],
        "steps_incorrect": [
            "6 × 4 = 20"  # Arithmetic error
        ],
        "expected_error": "arithmetic_error",
        "error_location": 0
    }
]


def create_demo_set():
    """Create demo set JSON file."""
    output_dir = Path("data/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "demo_questions.json"
    
    with open(output_file, 'w') as f:
        json.dump(DEMO_QUESTIONS, f, indent=2)
    
    print(f"✅ Created demo set with {len(DEMO_QUESTIONS)} questions")
    print(f"📁 Saved to: {output_file}")
    
    # Also create a markdown file for easy viewing
    md_file = output_dir / "demo_questions.md"
    with open(md_file, 'w') as f:
        f.write("# Demo Questions (PRD Requirement)\n\n")
        for i, q in enumerate(DEMO_QUESTIONS, 1):
            f.write(f"## Question {i}\n\n")
            f.write(f"**Problem:** {q['problem']}\n\n")
            f.write(f"**Correct Steps:**\n")
            for step in q['steps_correct']:
                f.write(f"- {step}\n")
            f.write(f"\n**Incorrect Steps (with error):**\n")
            for step in q['steps_incorrect']:
                f.write(f"- {step}\n")
            f.write(f"\n**Expected Error:** {q['expected_error']}\n")
            f.write(f"**Error Location:** Step {q['error_location'] + 1}\n\n")
            f.write("---\n\n")
    
    print(f"📄 Markdown version: {md_file}")


if __name__ == "__main__":
    create_demo_set()

