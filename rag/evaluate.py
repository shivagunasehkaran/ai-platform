"""RAG evaluation script for measuring retrieval quality."""

import json
import logging
import statistics
import sys
from pathlib import Path

from shared.telemetry import Timer, metrics_store
from rag.retrieve import retrieve

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# QA pairs file
QA_PAIRS_FILE = Path(__file__).parent / "qa_pairs.json"


def load_qa_pairs(filepath: Path = QA_PAIRS_FILE) -> list[dict]:
    """Load QA pairs from JSON file.
    
    Args:
        filepath: Path to QA pairs JSON file.
        
    Returns:
        list[dict]: List of QA pair dictionaries.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"QA pairs file not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_single(
    question: str,
    expected_source: str,
    top_k: int = 5,
) -> dict:
    """Evaluate a single question.
    
    Args:
        question: The question to ask.
        expected_source: Expected source file in results.
        top_k: Number of results to retrieve.
        
    Returns:
        dict: Evaluation results including hit, rank, and latency.
    """
    with Timer() as timer:
        results = retrieve(question, top_k=top_k)
    
    # Find rank of expected source
    rank = None
    hit = False
    
    for i, result in enumerate(results, 1):
        if result["source"] == expected_source:
            rank = i
            hit = True
            break
    
    return {
        "question": question,
        "expected_source": expected_source,
        "hit": hit,
        "rank": rank,
        "reciprocal_rank": 1.0 / rank if rank else 0.0,
        "latency_ms": timer.elapsed_ms,
        "retrieved_sources": [r["source"] for r in results],
    }


def calculate_metrics(results: list[dict]) -> dict:
    """Calculate aggregate metrics from evaluation results.
    
    Args:
        results: List of individual evaluation results.
        
    Returns:
        dict: Aggregate metrics including Recall@5, MRR@5, latencies.
    """
    total = len(results)
    hits = sum(1 for r in results if r["hit"])
    
    # Recall@5: percentage of questions where correct source in top 5
    recall_at_5 = hits / total if total > 0 else 0.0
    
    # MRR@5: Mean Reciprocal Rank
    mrr_at_5 = (
        sum(r["reciprocal_rank"] for r in results) / total
        if total > 0 else 0.0
    )
    
    # Latency statistics
    latencies = [r["latency_ms"] for r in results]
    latencies_sorted = sorted(latencies)
    
    median_latency = statistics.median(latencies) if latencies else 0.0
    mean_latency = statistics.mean(latencies) if latencies else 0.0
    
    # P95 latency (95th percentile)
    p95_index = int(len(latencies_sorted) * 0.95)
    p95_latency = latencies_sorted[p95_index] if latencies_sorted else 0.0
    
    return {
        "total_questions": total,
        "hits": hits,
        "recall_at_5": recall_at_5,
        "mrr_at_5": mrr_at_5,
        "median_latency_ms": median_latency,
        "mean_latency_ms": mean_latency,
        "p95_latency_ms": p95_latency,
    }


def print_results(results: list[dict], metrics: dict) -> None:
    """Print evaluation results in formatted output.
    
    Args:
        results: List of individual evaluation results.
        metrics: Aggregate metrics dictionary.
    """
    print("\n" + "=" * 70)
    print("RAG EVALUATION RESULTS")
    print("=" * 70)
    
    # Summary metrics
    print("\n📊 SUMMARY METRICS")
    print("-" * 40)
    print(f"  Total Questions:    {metrics['total_questions']}")
    print(f"  Hits (source in top 5): {metrics['hits']}")
    print(f"  Recall@5:           {metrics['recall_at_5']:.2%}")
    print(f"  MRR@5:              {metrics['mrr_at_5']:.4f}")
    
    print("\n⏱️  LATENCY METRICS")
    print("-" * 40)
    print(f"  Median Latency:     {metrics['median_latency_ms']:.0f}ms")
    print(f"  Mean Latency:       {metrics['mean_latency_ms']:.0f}ms")
    print(f"  P95 Latency:        {metrics['p95_latency_ms']:.0f}ms")
    
    # Detailed results
    print("\n📋 DETAILED RESULTS")
    print("-" * 70)
    print(f"{'Question':<40} {'Expected':<20} {'Hit':<5} {'Rank':<5}")
    print("-" * 70)
    
    for r in results:
        question = r["question"][:38] + ".." if len(r["question"]) > 40 else r["question"]
        expected = r["expected_source"][:18] if len(r["expected_source"]) > 20 else r["expected_source"]
        hit = "✅" if r["hit"] else "❌"
        rank = str(r["rank"]) if r["rank"] else "-"
        print(f"{question:<40} {expected:<20} {hit:<5} {rank:<5}")
    
    # Failures analysis
    failures = [r for r in results if not r["hit"]]
    if failures:
        print("\n⚠️  MISSED QUESTIONS")
        print("-" * 70)
        for r in failures:
            print(f"  Q: {r['question']}")
            print(f"     Expected: {r['expected_source']}")
            print(f"     Got: {', '.join(r['retrieved_sources'])}")
            print()
    
    print("=" * 70)


def run_evaluation(top_k: int = 5) -> dict:
    """Run full evaluation on QA pairs.
    
    Args:
        top_k: Number of results to retrieve per question.
        
    Returns:
        dict: Aggregate metrics from evaluation.
    """
    # Load QA pairs
    qa_pairs = load_qa_pairs()
    logger.info(f"Loaded {len(qa_pairs)} QA pairs")
    
    # Evaluate each question
    results = []
    for i, qa in enumerate(qa_pairs, 1):
        print(f"Evaluating {i}/{len(qa_pairs)}: {qa['question'][:50]}...")
        
        result = evaluate_single(
            question=qa["question"],
            expected_source=qa["expected_source"],
            top_k=top_k,
        )
        results.append(result)
        
        # Log to metrics store
        metrics_store.log_retrieval_metrics(
            query=qa["question"],
            latency_ms=result["latency_ms"],
            recall=1.0 if result["hit"] else 0.0,
            mrr=result["reciprocal_rank"],
        )
    
    # Calculate aggregate metrics
    metrics = calculate_metrics(results)
    
    # Print results
    print_results(results, metrics)
    
    return metrics


def main() -> None:
    """Entry point for evaluation CLI."""
    print("\n🔬 Starting RAG Evaluation...\n")
    
    try:
        metrics = run_evaluation()
        
        # Save metrics
        from shared.config import METRICS_FILE
        metrics_store.save_metrics(METRICS_FILE)
        print(f"\n💾 Metrics saved to {METRICS_FILE}")
        
        # Exit with error if recall is below threshold
        if metrics["recall_at_5"] < 0.5:
            print("\n⚠️  Warning: Recall@5 is below 50%")
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
