from typing import List, Dict, Any
import re

def _normalize_answer(ans: str) -> str:
    """Normalize an answer string for comparison (remove spaces, lowercase, strip LaTeX wrappers)."""
    s = str(ans).strip()
    s = re.sub(r'\$', '', s)
    s = re.sub(r'\\(?:approx|approx|cdot|,|;|\s)', ' ', s)
    s = s.replace("\\", "").replace("{", "").replace("}", "")
    s = s.replace(" ", "").lower()
    # Normalize floats: "3.0" == "3"
    try:
        f = float(s)
        s = str(int(f)) if f == int(f) else str(round(f, 6))
    except:
        pass
    return s

def normalize_answers(answers: List[str]) -> Dict[str, List[int]]:
    """Group answers that are numerically/symbolically equivalent."""
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
    """
    L_logic: measures intra-agent logical flow.
    Checks for contradiction signals, empty steps, and step count.
    """
    if not trace:
        return 0.0
    contradiction_terms = ["incorrect", "divergence", "wrong", "error", "divergent", "hallucin"]
    score = 1.0
    for step in trace:
        if any(t in step.lower() for t in contradiction_terms):
            score -= 0.3
    # Longer traces with more reasoning steps are rewarded slightly
    score += min(0.1 * (len(trace) - 1), 0.3)
    return max(0.0, min(1.0, score))

def _calculate_classifier_score(conf_exp: str, is_divergent: bool) -> float:
    """
    C_clf: maps confidence explanation to numerical probability.
    """
    if is_divergent:
        return 0.1
    text = conf_exp.lower()
    if any(w in text for w in ["high confidence", "certain", "guaranteed", "verified", "proof"]):
        return 0.95
    elif any(w in text for w in ["divergent", "divergence", "wrong", "hallucin", "low confidence"]):
        return 0.1
    elif any(w in text for w in ["likely", "confident", "probably"]):
        return 0.75
    elif any(w in text for w in ["unsure", "guess", "uncertain"]):
        return 0.3
    return 0.55  # Neutral default

def evaluate_consensus(
    agent_responses: List[Dict[str, Any]],
    ocr_confidence: float = 1.0
) -> Dict[str, Any]:
    """
    Adaptive Multi-Signal Consensus:
    Score_j = 0.40 * V_sym + 0.35 * L_logic + 0.25 * C_clf
    FinalConf = Score_j * (0.9 + 0.1 * OCR_conf)

    Also detects:
    - Answer divergence (agents disagree → flag as uncertain)
    - Individual hallucination (score < 0.65 OR marked as divergent by agent)
    - High-confidence wrong answers
    """
    if not agent_responses:
        return {
            "final_verified_answer": "No agents responded",
            "winning_score": 0.0,
            "detail_scores": [],
            "divergence_groups": {},
            "hallucination_alerts": [],
            "verdict": "ERROR"
        }

    # Import compute symbolic score
    try:
        from verification_service import calculate_symbolic_score
    except ImportError:
        def calculate_symbolic_score(trace): return 1.0 if trace else 0.0

    scores = []
    hallucination_alerts = []
    answers = [res["response"].get("Answer", "N/A") for res in agent_responses]
    answer_groups = normalize_answers(answers)

    # Determine if there is significant divergence between agents
    num_unique_answers = len(answer_groups)
    has_divergence = num_unique_answers > 1

    for idx, agent_data in enumerate(agent_responses):
        res = agent_data["response"]
        trace = res.get("Reasoning Trace", [])
        conf_exp = res.get("Confidence Explanation", "")
        raw_ans = res.get("Answer", "N/A")

        # Check if the agent itself marked this as divergent/hallucinating
        is_self_flagged = any(t in conf_exp.lower() for t in ["divergent", "wrong", "hallucin", "low confidence", "divergence"])

        # V_sym: SymPy symbolic reasoning verification (weight 0.40)
        v_sym = calculate_symbolic_score(trace)

        # L_logic: logical consistency & step quality (weight 0.35)
        l_logic = _calculate_logical_score(trace)

        # C_clf: confidence classifier (weight 0.25)
        c_clf = _calculate_classifier_score(conf_exp, is_self_flagged)

        # Core scoring formula
        score_j = (0.40 * v_sym) + (0.35 * l_logic) + (0.25 * c_clf)

        # OCR calibration
        final_conf = score_j * (0.9 + 0.1 * ocr_confidence)

        # Hallucination detection — flag if:
        # 1. Score is below threshold (lowered from 0.7 to 0.65 for better sensitivity)
        # 2. Agent self-flagged as divergent
        # 3. High-confidence answer but symbolic score is 0 (contradiction)
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
            hallucination_alerts.append({
                "agent": agent_data["agent"],
                "answer": raw_ans,
                "reason": alert_reason,
                "score": round(score_j, 3)
            })

        scores.append({
            "agent": agent_data["agent"],
            "raw_answer": raw_ans,
            "V_sym": round(v_sym, 3),
            "L_logic": round(l_logic, 3),
            "C_clf": round(c_clf, 3),
            "Score_j": round(score_j, 3),
            "FinalConf": round(final_conf, 3),
            "is_hallucinating": is_hallucinating
        })

    # Aggregate: find the most supported, highest-scoring answer group
    final_consensus = {}
    top_score = -1.0
    best_answer = "Unresolvable Divergence"

    for rep_ans, indices in answer_groups.items():
        # Prefer non-hallucinating agents when aggregating
        valid_idx = [i for i in indices if not scores[i]["is_hallucinating"]]
        base_idx = valid_idx if valid_idx else indices

        group_score = sum(scores[i]["FinalConf"] for i in base_idx)
        # Consistency bonus: more agents agreeing on same answer → stronger signal
        consistency_multiplier = 1.0 + (0.15 * (len(base_idx) - 1))
        weighted = group_score * consistency_multiplier

        final_consensus[rep_ans] = {
            "agents_supporting": [scores[i]["agent"] for i in indices],
            "valid_agent_count": len(valid_idx),
            "aggregate_score": round(weighted, 3)
        }

        if weighted > top_score:
            top_score = weighted
            best_answer = rep_ans

    # Determine overall verdict with clearer thresholds
    if top_score >= 1.5 and not has_divergence and not hallucination_alerts:
        verdict = "✅ STRONGLY VERIFIED"
    elif top_score >= 1.0 and len(hallucination_alerts) == 0:
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
