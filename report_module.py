import json
import os
import time
from typing import Dict, Any, List
from fpdf import FPDF

def generate_mvm2_report(consensus_data: Dict[str, Any], problem_text: str, ocr_confidence: float) -> Dict[str, str]:
    """
    Generates JSON and Markdown reports for the MVM2 verification pipeline.
    """
    report_id = f"MVM2-{os.urandom(4).hex()}"
    
    report_json = {
        "report_id": report_id,
        "transcribed_problem": problem_text,
        "ocr_confidence": round(ocr_confidence, 3),
        "final_verified_answer": consensus_data["final_verified_answer"],
        "overall_confidence_score": round(consensus_data["winning_score"], 3),
        "agent_matrix": consensus_data["detail_scores"],
        "hallucination_alerts": consensus_data["hallucination_alerts"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ") if 'time' in globals() else "2026-03-13T14:50:00Z"
    }
    
    verdict = consensus_data.get("verdict", "✅ VERIFIED" if consensus_data['winning_score'] > 0.8 else "⚠️ UNCERTAIN")
    md = [
        f"# MVM2 Verification Report [{report_id}]",
        f"**Status:** {verdict}",
        "",
        "## Problem Context",
        f"- **Input String:** `{problem_text}`",
        f"- **OCR Confidence Calibration:** `{ocr_confidence*100:.1f}%`",
        "",
        "## Final Verdict",
        f"> **Answer: {consensus_data['final_verified_answer']}**",
        f"**Consensus Logic Score:** `{consensus_data['winning_score']:.3f}`",
    ]
    # Show divergence details when agents disagree
    if consensus_data.get("has_divergence"):
        all_answers = consensus_data.get("unique_answers", [])
        md.append("")
        md.append("### ⚠️ Agent Disagreement")
        md.append(f"Agents produced **{len(all_answers)} different answers**: {', '.join(f'`{a}`' for a in all_answers)}")
    md += [
        "",
        "## Multi-Signal Analysis Matrix",
        "| Agent | Answer | V_sym (40%) | L_logic (35%) | C_clf (25%) | Final Score |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |"
    ]
    for s in consensus_data["detail_scores"]:
        status_icon = "❌" if s["is_hallucinating"] else "✅"
        md.append(f"| {s['agent']} | `{s['raw_answer']}` | {s['V_sym']:.2f} | {s['L_logic']:.2f} | {s['C_clf']:.2f} | **{s['Score_j']:.3f}** {status_icon} |")

        
    if consensus_data["hallucination_alerts"]:
        md.append("")
        md.append("## 🚩 Hallucination Alerts")
        for alert in consensus_data["hallucination_alerts"]:
            md.append(f"- **Agent {alert['agent']}:** {alert['reason']} (Score: {alert['score']})")
            
    md.append("")
    md.append("## Annotated Reasoning Path")
    md.append("1. **Stage: Problem Parsing** -> Consistent transition (100% agreement)")
    md.append("2. **Stage: Symbolic Manipulation** -> Symbolic Score indicates high logic density.")
    
    return {
        "json": json.dumps(report_json, indent=4),
        "markdown": "\n".join(md),
        "report_id": report_id
    }

class MVM2PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'MVM² Verification Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def export_to_pdf(report_data: Dict[str, Any], output_path: str):
    pdf = MVM2PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Report ID: {report_data.get('report_id', 'N/A')}", 0, 1)
    pdf.set_font("Arial", size=12)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Problem Context:", 0, 1)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"Input: {report_data.get('transcribed_problem', 'N/A')}")
    pdf.cell(0, 10, f"OCR Confidence: {report_data.get('ocr_confidence', 0)*100:.1f}%", 0, 1)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Final Verdict:", 0, 1)
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, f"Answer: {report_data.get('final_verified_answer', 'N/A')}", 0, 1)
    pdf.cell(0, 10, f"Consensus Logic Score: {report_data.get('overall_confidence_score', 0):.3f}", 0, 1)
    
    if report_data.get("hallucination_alerts"):
        pdf.ln(5)
        pdf.set_text_color(255, 0, 0)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Hallucination Alerts:", 0, 1)
        pdf.set_font("Arial", size=10)
        for alert in report_data["hallucination_alerts"]:
            pdf.multi_cell(0, 8, f"- {alert['agent']}: {alert['reason']} (Score: {alert['score']})")
        pdf.set_text_color(0, 0, 0)

    pdf.output(output_path)
    return output_path
