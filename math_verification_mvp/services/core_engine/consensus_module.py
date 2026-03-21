import math
from typing import List, Dict, Any
import re
from services.core_engine.verification_module import calculate_symbolic_score

def _normalize_answer(ans: str) -> str:
    s = str(ans).strip()
    s = re.sub(r'\$', '', s)
    s = re.sub(r'\\(?:approx|approx|cdot|,|;|\s)', ' ', s)
    s = s.replace("\\", "").replace("{", "").replace("}", "")
    s = s.replace(" ", "").lower()
    try:
        f = float(s)
        s = str(int(f)) if f == int(f) else str(round(f, 6))
    except: pass
    return s

def normalize_answers(answers: List[str]) -> Dict[str, List[int]]:
    normalized_groups = {}
    for idx, ans in enumerate(answers):
        clean = _normalize_answer(ans)
        matched = False
        for key in list(normalized_groups.keys()):
            if _normalize_answer(key) == clean:
                normalized_groups[key].append(idx)
                matched = True
                break
        if not matched:
            normalized_groups[ans] = [idx]
    return normalized_groups

def _calculate_logical_score(trace: List[str]) -> float:
    if not trace: return 0.0
    contradiction_terms = ["incorrect", "divergence", "wrong", "error", "divergent", "hallucin"]
    score = 1.0
    for step in trace:
        if any(t in step.lower() for t in contradiction_terms):
            score -= 0.3
    score += min(0.1 * (len(trace) - 1), 0.3)
    return max(0.0, min(1.0, score))

def _calculate_classifier_score(conf_exp: str, is_divergent: bool) -> float:
    if is_divergent: return 0.1
    text = conf_exp.lower()
    if any(w in text for w in ["high confidence", "certain", "guaranteed", "verified", "proof"]):
        return 0.95
    elif any(w in text for w in ["divergent", "divergence", "wrong", "hallucin", "low confidence"]):
        return 0.1
    elif any(w in text for w in ["likely", "confident", "probably"]):
        return 0.75
    elif any(w in text for w in ["unsure", "guess", "uncertain"]):
        return 0.3
    return 0.55 

def evaluate_consensus(agent_responses: List[Dict[str, Any]], ocr_confidence: float = 1.0) -> Dict[str, Any]:
    if not agent_responses:
        return {"final_verified_answer": "No agents responded", "winning_score": 0.0, "detail_scores": [], "divergence_groups": {}, "hallucination_alerts": [], "verdict": "ERROR"}

    scores = []
    hallucination_alerts = []
    answers = [res["response"].get("Answer", "N/A") for res in agent_responses]
    answer_groups = normalize_answers(answers)
    has_divergence = len(answer_groups) > 1

    for idx, agent_data in enumerate(agent_responses):
        res = agent_data["response"]
        trace = res.get("Reasoning Trace", [])
        conf_exp = res.get("Confidence Explanation", "")
        raw_ans = res.get("Answer", "N/A")

        is_self_flagged = any(t in conf_exp.lower() for t in ["divergent", "wrong", "hallucin", "low confidence", "divergence"])
        v_sym = calculate_symbolic_score(trace)
        l_logic = _calculate_logical_score(trace)
        c_clf = _calculate_classifier_score(conf_exp, is_self_flagged)

        score_j = (0.40 * v_sym) + (0.35 * l_logic) + (0.25 * c_clf)
        # Normalize to [0, 1] range
        final_conf = score_j * (0.9 + 0.1 * ocr_confidence)

        is_hallucinating = False
        alert_reason = None
        if score_j < 0.65:
            alert_reason = f"Low consensus score ({score_j:.3f} < 0.65)"
        elif is_self_flagged:
            alert_reason = "Agent self-reported divergent reasoning path"
        elif v_sym == 0.0 and c_clf > 0.7:
            alert_reason = "High-confidence answer with zero symbolic validity"

        if alert_reason:
            is_hallucinating = True
            hallucination_alerts.append({"agent": agent_data["agent"], "answer": raw_ans, "reason": alert_reason, "score": round(score_j, 3)})

        scores.append({
            "agent": agent_data["agent"], "raw_answer": raw_ans,
            "V_sym": round(v_sym, 3), "L_logic": round(l_logic, 3), "C_clf": round(c_clf, 3),
            "Score_j": round(score_j, 3), "FinalConf": round(final_conf, 3),
            "is_hallucinating": is_hallucinating
        })

    final_consensus = {}
    top_score = -1.0
    best_answer = "Unresolvable Divergence"

    num_agents = len(agent_responses)
    for rep_ans, indices in answer_groups.items():
        valid_idx = [i for i in indices if not scores[i]["is_hallucinating"]]
        base_idx = valid_idx if valid_idx else indices
        
        # Expert Calibration: Group score is averaged by total agents to maintain [0, 1] scale
        group_score = sum(scores[i]["FinalConf"] for i in base_idx) / num_agents
        
        # Consistency bonus: scaled by agreement ratio
        agreement_ratio = len(base_idx) / num_agents
        weighted = group_score * (1.0 + 0.5 * agreement_ratio)

        final_consensus[rep_ans] = {"agents_supporting": [scores[i]["agent"] for i in indices], "valid_agent_count": len(valid_idx), "aggregate_score": round(weighted, 3)}

        if weighted > top_score:
            top_score = weighted
            best_answer = rep_ans

    # Adjusted Verdict Thresholds for [0, 1.5] range
    if top_score >= 0.85 and not has_divergence and not hallucination_alerts:
        verdict = "✅ STRONGLY VERIFIED"
    elif top_score >= 0.7 and len(hallucination_alerts) == 0:
        verdict = "✅ VERIFIED"
    elif has_divergence and len(hallucination_alerts) > 0:
        verdict = "❌ DIVERGENCE DETECTED — LIKELY WRONG"
    elif has_divergence:
        verdict = "⚠️ UNCERTAIN — AGENTS DISAGREE"
    elif hallucination_alerts:
        verdict = "⚠️ UNCERTAIN — HALLUCINATION RISK"
    else:
        verdict = "⚠️ LOW CONFIDENCE"

    return {
        "final_verified_answer": best_answer,
        "winning_score": round(top_score, 3),
        "detail_scores": scores,
        "divergence_groups": final_consensus,
        "hallucination_alerts": hallucination_alerts,
        "has_divergence": has_divergence,
        "unique_answers": list(answer_groups.keys()),
        "verdict": verdict
    }
