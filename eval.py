import argparse
import json

import nltk
from bert_score import score as bert_score
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from rag_chatbot import ask_astronomy_bot

# ── Helpers ────────────────────────────────────────────────────────────────────


def ensure_nltk_data() -> None:
    """Download required NLTK corpora if not already present."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"[setup] Downloading NLTK resource: {pkg}")
            nltk.download(pkg, quiet=True)


def compute_rouge(hypothesis: str, reference: str) -> dict:
    """Return ROUGE-1, ROUGE-2, and ROUGE-L F1 / precision / recall."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)

    result = {}
    for key, val in scores.items():
        result[key] = {
            "precision": round(val.precision, 4),
            "recall": round(val.recall, 4),
            "f1": round(val.fmeasure, 4),
        }
    return result


def compute_bertscore(hypothesis: str, reference: str, lang: str = "en") -> dict:
    """Return BERTScore precision, recall, and F1."""
    P, R, F1 = bert_score(
        [hypothesis],
        [reference],
        lang=lang,
        verbose=False,
    )
    return {
        "precision": round(P[0].item(), 4),
        "recall": round(R[0].item(), 4),
        "f1": round(F1[0].item(), 4),
    }


def compute_meteor(hypothesis: str, reference: str) -> dict:
    """Return METEOR score."""
    hyp_tokens = word_tokenize(hypothesis.lower())
    ref_tokens = word_tokenize(reference.lower())
    score = meteor_score([ref_tokens], hyp_tokens)
    return {"score": round(score, 4)}


def evaluate(hypothesis: str, reference: str, lang: str = "en") -> dict:
    """Run all three metrics and return a combined results dict."""
    print("\n[1/3] Computing ROUGE …")
    rouge = compute_rouge(hypothesis, reference)

    print("[2/3] Computing BERTScore …")
    bscore = compute_bertscore(hypothesis, reference, lang=lang)

    print("[3/3] Computing METEOR …")
    meteor = compute_meteor(hypothesis, reference)

    return {
        "rouge": rouge,
        "bertscore": bscore,
        "meteor": meteor,
    }


def pretty_print(results: dict, title: str = "EVALUATION RESULTS") -> None:
    print("\n" + "=" * 55)
    print(f"  {title}")
    print("=" * 55)

    # ROUGE
    print("\n  ROUGE")
    print(f"  {'Metric':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("  " + "-" * 44)
    for key, vals in results["rouge"].items():
        print(
            f"  {key.upper():<12} {vals['precision']:>10.4f} "
            f"{vals['recall']:>10.4f} {vals['f1']:>10.4f}"
        )

    # BERTScore
    bs = results["bertscore"]
    print("\n  BERTScore")
    print(f"  {'Metric':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("  " + "-" * 44)
    print(
        f"  {'BERTScore':<12} {bs['precision']:>10.4f} "
        f"{bs['recall']:>10.4f} {bs['f1']:>10.4f}"
    )

    # METEOR
    print("\n  METEOR")
    print(f"  Score: {results['meteor']['score']:.4f}")

    print("\n" + "=" * 55)


def compute_means(all_results: list[dict]) -> dict:
    """Average all metric values across multiple evaluation results."""
    n = len(all_results)

    rouge_keys = all_results[0]["rouge"].keys()  # rouge1, rouge2, rougeL
    sub_keys = ["precision", "recall", "f1"]

    mean_rouge = {
        rk: {
            sk: round(sum(r["rouge"][rk][sk] for r in all_results) / n, 4)
            for sk in sub_keys
        }
        for rk in rouge_keys
    }
    mean_bert = {
        sk: round(sum(r["bertscore"][sk] for r in all_results) / n, 4)
        for sk in sub_keys
    }
    mean_meteor = {
        "score": round(sum(r["meteor"]["score"] for r in all_results) / n, 4)
    }

    return {"rouge": mean_rouge, "bertscore": mean_bert, "meteor": mean_meteor}


# ── CLI ────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a chatbot answer against a human reference answer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pass answers directly
  python evaluate_chatbot.py \\
      --hypothesis "The cat sat on the mat." \\
      --reference  "A cat was sitting on the mat."

  # Read from a JSON file  { "hypothesis": "...", "reference": "..." }
  python evaluate_chatbot.py --input pairs.json

  # Save results to JSON
  python evaluate_chatbot.py \\
      --hypothesis "..." --reference "..." \\
      --output results.json
        """,
    )
    p.add_argument(
        "--lang", default="en", help="Language code for BERTScore (default: en)"
    )
    return p


def main() -> None:
    ensure_nltk_data()
    astronomy_qa_list = [
        {
            "question": "What is the difference in the shape of the Hubble's orbit and Chandra's orbit around the Earth?",
            "answer": "Hubble has a circular orbit, while Chandra has highly elliptical or oval-shaped orbit.",
        },
        {
            "question": "What negatively affects visual perception during visual meteor observations?",
            "answer": "A lack of vitamin A, consumation of alcohol and nicotine.",
        },
        {
            "question": "Why waas the name of the Large Synoptic Survey Telescope changed to Vera C. Rubin Observatory?",
            "answer": "Because the new name honors an accomplished American astronomer and acknowledges the contributions to astronomy and astrophysics made by women.",
        },
        {
            "question": "What is the opposition in astronomy?",
            "answer": "Oposition is when a planet or asteroid is opposite the Sun in the sky. At such times the object is visible all night — rising at sunset and setting at sunrise.",
        },
        {
            "question": "What are the classes of intrinsic variable stars?",
            "answer": "Pulsating variable stars, cataclysmic variable stars, and eruptive variable stars.",
        },
        {
            "question": "After a star has consumed the helium at its core, into which part of the Hertzsprung-Russel diagram does it go to?",
            "answer": "Asymptotic giant branch.",
        },
        {
            "question": "What software package can be used to plot the signal received by a radio telescope onto a strip chart?",
            "answer": "Radio-Skypipe.",
        },
    ]
    parser = build_parser()
    args = parser.parse_args()

    all_results = []
    for i in astronomy_qa_list:
        reference = i["answer"]
        question = i["question"]
        hypothesis = ask_astronomy_bot(question, verbose=False)
        print(f"Q: {question}")
        print(f"A: {hypothesis}")
        print(f"R: {reference}\n")
        assert hypothesis is not None, "Hypothesis should be a string"
        all_results.append(evaluate(hypothesis, reference, lang=args.lang))

    means = compute_means(all_results)
    pretty_print(means, title=f"MEAN RESULTS OVER {len(all_results)} PAIRS")

    # ── Optional JSON output ───────────────────────────────────────────────────
    with open("eval.json", "w", encoding="utf-8") as f:
        json.dump(means, f, indent=2)


if __name__ == "__main__":
    main()
