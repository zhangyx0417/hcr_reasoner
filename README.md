# AC-Reason: Theory-Guided Actual Causality Reasoning with Large Language Models

## Framework

**AC-Reason** is a novel framework that integrates Actual Causality (AC) theory into large language models (LLMs) to enhance their ability to perform accurate and interpretable actual causal reasoning. By combining structured causal knowledge with theory-guided algorithms, AC-Reason identifies causally relevant events, evaluates formal causal factors (such as sufficiency, necessity, and normality), and determines actual causal relationships.

## Benchmark

**AC-Bench** is a newly introduced benchmark designed to evaluate actual causality reasoning in large language models. It consists of ~1K carefully annotated samples, each providing detailed reasoning steps for determining actual causal relationships.

## Features

- **Theory-Guided Actual Causal Reasoning**: AC-Reason incorporates formal AC theory into the reasoning process, offering better interpretability and more accurate causal conclusions.
- **More Comprehensive Evaluation**: AC-Reason includes **AC-Bench**, a benchmark dataset with ~1K carefully annotated samples that provides detailed reasoning steps for testing and improving LLM performance.
- **Improved Model Performance**: AC-Reason demonstrates superior performance compared to baseline models on actual causal reasoning tasks on **Big-Bench Hard Causal Judgment** and **AC-Bench**.

## Project Directory Structure

The repository is organized as follows:

```
AC_REASON/
├── code/                      # Source code for AC-Reason and baselines
│   ├── prompts/               # Prompt files for each stage of AC-Reason
│   │   ├── step-1.txt         # Prompt for Stage 1
│   │   ├── step-2.txt         # Prompt for Stage 2
│   ├── ac_reason.py           # Implementation of AC-Reason
│   ├── vanilla.py             # Baseline: Vanilla
│   ├── zero.py                # Baseline: Zero-shot CoT
│   ├── manual.py              # Baseline: Manual CoT
│   ├── s1.py                  # Ablation: Stage 1 only
│   ├── s12.py                 # Ablation: Stage 1 and Stage 2
│   ├── utils.py               # Utility functions
├── data/                      # Datasets
│   ├── ac_bench.json          # Our proposed AC-BENCH dataset
│   ├── bbh_cj.json            # Our processed Big-Bench Hard Causal Judgment dataset
├── results/                   # Experiment results
│   ├── ablation/              # Results from our ablation study
│   ├── main/                  # Results from our main experiment
│   ├── pilot/                 # Results from our pilot study
```

## Getting Started

1. **Clone the Repository**

2. **Install Dependencies**

   ```bash
   pip install openai==1.78.1
   ```

3. **Run the Framework**

   ```bash
   python code/ac_reason.py
   ```
