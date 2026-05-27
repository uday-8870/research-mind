"""
eval_benchmark.py — Run ResearchMind against 50 test questions
and produce a benchmark report comparing against baselines.

Usage:
    python tests/eval_benchmark.py --output results/benchmark.json
"""
import asyncio
import json
import time
import argparse
from pathlib import Path

# Sample benchmark questions across domains
BENCHMARK_QUESTIONS = [
    # Science
    "What are the latest breakthroughs in CRISPR gene editing?",
    "How does quantum entanglement work and what are its applications?",
    "What is the current state of fusion energy research?",
    # Technology
    "How do large language models handle reasoning tasks?",
    "What are the main differences between transformer architectures?",
    "How does federated learning preserve privacy in ML systems?",
    # Business/Finance
    "What factors drive semiconductor supply chain disruptions?",
    "How do central banks use quantitative easing to manage inflation?",
    # Health
    "What are the mechanisms behind mRNA vaccine technology?",
    "How does the gut microbiome affect mental health?",
]


async def run_single_benchmark(query: str, timeout: int = 120) -> dict:
    """Run research on one question and measure metrics."""
    from backend.agents.orchestrator import run_research

    start = time.time()
    try:
        result = await asyncio.wait_for(
            run_research(query, depth="standard"), timeout=timeout
        )
        latency = time.time() - start
        eval_r = result.get("eval_result") or {}
        return {
            "query": query,
            "status": result.get("status"),
            "latency_s": round(latency, 2),
            "sources_found": len(result.get("enriched_sources", [])),
            "sub_queries": len(result.get("sub_queries", [])),
            "report_chars": len(result.get("report", "")),
            "faithfulness": eval_r.get("faithfulness"),
            "coverage": eval_r.get("coverage"),
            "hallucination_rate": eval_r.get("hallucination_rate"),
            "quality_score": eval_r.get("quality_score"),
            "error": result.get("error"),
        }
    except asyncio.TimeoutError:
        return {"query": query, "status": "timeout", "latency_s": timeout, "error": "timeout"}
    except Exception as e:
        return {"query": query, "status": "error", "error": str(e)}


def compute_summary(results: list[dict]) -> dict:
    """Compute aggregate statistics."""
    successful = [r for r in results if r.get("status") == "done"]
    n = len(successful)
    if n == 0:
        return {"error": "No successful runs"}

    def avg(key):
        vals = [r[key] for r in successful if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    return {
        "total_questions": len(results),
        "successful": n,
        "success_rate": round(n / len(results), 2),
        "avg_latency_s": avg("latency_s"),
        "avg_faithfulness": avg("faithfulness"),
        "avg_coverage": avg("coverage"),
        "avg_hallucination_rate": avg("hallucination_rate"),
        "avg_quality_score": avg("quality_score"),
        "avg_sources_found": avg("sources_found"),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/benchmark.json")
    parser.add_argument("--questions", type=int, default=10)
    args = parser.parse_args()

    questions = BENCHMARK_QUESTIONS[: args.questions]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Running benchmark on {len(questions)} questions...\n")
    results = []

    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {q[:60]}...")
        r = await run_single_benchmark(q)
        results.append(r)
        status_icon = "✅" if r.get("status") == "done" else "❌"
        print(f"  {status_icon} {r.get('latency_s', '?')}s | faith={r.get('faithfulness')} | cov={r.get('coverage')}\n")

    summary = compute_summary(results)
    output = {"summary": summary, "results": results}

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print("\n=== BENCHMARK SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
