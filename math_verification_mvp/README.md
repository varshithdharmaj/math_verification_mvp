---
title: MVM2 Math Verification System
emoji: 🧮
colorFrom: red
colorTo: gray
sdk: streamlit
sdk_version: 1.42.0
app_file: services/dashboard/app.py
pinned: false
license: mit
short_description: Multi-Model Math Verification System with Explainable AI
---

# MVM² — Multi-Model Math Verification System

A multi-agent AI system that verifies mathematical reasoning step-by-step using symbolic computation (SymPy) and LLM-based logic checking.

![Status](https://img.shields.io/badge/status-production--ready-green)
![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Docker](https://img.shields.io/badge/docker-enabled-blue)

## Features
- 🔬 **Parallel Verification**: Runs SymPy, LLM, and Ensemble models simultaneously
- 📊 **Consensus Fusion**: Weighted scoring across multiple agents
- 🧠 **Explainable AI**: Step-by-step error classification with natural language explanations
- 📄 **Report Export**: Download verification reports as PDF, Word, or Markdown (powered by VibeDoc)

## How to Use
1. Enter your math problem and each step on a new line in the text box
2. Select which AI verifiers to enable in the sidebar
3. Click **Run Verification Pipeline**
4. Review the verdict, confidence score, and detailed breakdown
5. Download the full analysis report in your preferred format

## System Architecture
```
Input → SymPy Symbolic Verifier ─┐
      → LLM Logical Checker   ──→ Consensus Fusion → Verdict + Report
      → Ensemble Neural Check ─┘
```

## Based on Research Paper
*Mathematical Reasoning Enhancement in Large Language Models* — VNRVJIET, Hyderabad
