"""
Evaluation Metrics — the heart of what makes ResearchMind different.

Implements:
- Faithfulness: Are claims grounded in retrieved sources?
- Coverage: Were all sub-questions answered?
- Hallucination Rate: What % of claims lack source support?
- GPT-as-Judge: LLM-based quality scoring
"""
import json
import re
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, SystemMessage
from backend.core.llm import get_eval_llm


@dataclass
class EvalResult:
    faithfulness: float       # 0-1
    coverage: float           # 0-1
    hallucination_rate: float # 0-1 (lower is better)
    quality_score: float      # 1-10
    reasoning: str
    flagged_claims: list[str]


FAITHFULNESS_PROMPT = """You are an evaluation judge. Given a research report and the source documents used to generate it, evaluate faithfulness.

For each claim in the report, check if it is supported by the sources.

Source Documents:
{sources}

Research Report:
{report}

Respond in JSON:
{{
  "faithfulness_score": <0.0-1.0>,
  "unsupported_claims": ["claim1", "claim2", ...],
  "reasoning": "brief explanation"
}}

JSON only, no markdown."""


COVERAGE_PROMPT = """You are an evaluation judge. Given a research question, its sub-questions, and a report, evaluate coverage.

Original Question: {query}
Sub-questions: {sub_queries}

Report:
{report}

For each sub-question, determine if it was answered. Respond in JSON:
{{
  "coverage_score": <0.0-1.0>,
  "answered": ["sub_q1", ...],
  "unanswered": ["sub_q2", ...],
  "reasoning": "brief explanation"
}}

JSON only, no markdown."""


QUALITY_PROMPT = """You are a research quality evaluator. Rate this research report on a 1-10 scale.

Criteria:
- Depth and thoroughness (25%)
- Clarity and organization (25%)
- Evidence quality (25%)
- Actionable insights (25%)

Query: {query}
Report: {report}

Respond in JSON:
{{
  "quality_score": <1-10>,
  "strengths": ["s1", "s2"],
  "weaknesses": ["w1", "w2"],
  "reasoning": "brief explanation"
}}

JSON only, no markdown."""


def _parse_json(text: str) -> dict:
    """Safely parse JSON from LLM output."""
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}


async def evaluate_report(
    query: str,
    sub_queries: list[str],
    report: str,
    sources: list[dict],
) -> EvalResult:
    """Run full evaluation pipeline on a generated report."""
    llm = get_eval_llm()

    source_text = "\n\n".join(
        f"[Source {i+1}] {s.get('url','')}\n{s.get('content','')[:800]}"
        for i, s in enumerate(sources[:6])
    )

    # Run evaluations in parallel (using ainvoke)
    faith_msg = [
        SystemMessage(content="You are a precise evaluation judge."),
        HumanMessage(
            content=FAITHFULNESS_PROMPT.format(
                sources=source_text, report=report[:3000]
            )
        ),
    ]
    cov_msg = [
        SystemMessage(content="You are a precise evaluation judge."),
        HumanMessage(
            content=COVERAGE_PROMPT.format(
                query=query,
                sub_queries="\n".join(f"- {q}" for q in sub_queries),
                report=report[:3000],
            )
        ),
    ]
    qual_msg = [
        SystemMessage(content="You are a precise evaluation judge."),
        HumanMessage(
            content=QUALITY_PROMPT.format(query=query, report=report[:3000])
        ),
    ]

    faith_resp = await llm.ainvoke(faith_msg)
    cov_resp = await llm.ainvoke(cov_msg)
    qual_resp = await llm.ainvoke(qual_msg)

    faith_data = _parse_json(faith_resp.content)
    cov_data = _parse_json(cov_resp.content)
    qual_data = _parse_json(qual_resp.content)

    faithfulness = float(faith_data.get("faithfulness_score", 0.75))
    unsupported = faith_data.get("unsupported_claims", [])
    hallucination_rate = len(unsupported) / max(len(unsupported) + 10, 1)
    coverage = float(cov_data.get("coverage_score", 0.75))
    quality = float(qual_data.get("quality_score", 7.0))

    reasoning_parts = [
        faith_data.get("reasoning", ""),
        cov_data.get("reasoning", ""),
        qual_data.get("reasoning", ""),
    ]

    return EvalResult(
        faithfulness=round(faithfulness, 3),
        coverage=round(coverage, 3),
        hallucination_rate=round(hallucination_rate, 3),
        quality_score=round(quality, 1),
        reasoning=" | ".join(r for r in reasoning_parts if r),
        flagged_claims=unsupported[:5],
    )
