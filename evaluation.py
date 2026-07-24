"""
evaluation.py - RAG evaluation metrics: Precision@K, MRR, and LLM-as-judge
faithfulness / relevance scoring.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import google.generativeai as genai

_MODEL_NAME = "gemini-2.0-flash"
_judge_model = None


def _get_judge_model():
    global _judge_model
    if _judge_model is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
        _judge_model = genai.GenerativeModel(model_name=_MODEL_NAME)
    return _judge_model


@dataclass
class EvalResult:
    precision_at_k: float
    mrr: float
    faithfulness: float
    relevance: float
    details: list = field(default_factory=list)


def precision_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / k


def mean_reciprocal_rank(retrieved_ids, relevant_ids):
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def _parse_judge_json(raw_text):
    try:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        preview = raw_text[:200]
        return {"score": 0.0, "explanation": "Failed to parse judge output: " + preview}


async def llm_judge_faithfulness(query, answer, context):
    prompt = f"""You are grading whether an AI answer is faithful to the given context.

Context:
{context}

Question: {query}

Answer: {answer}

Score the answer's faithfulness to the context on a 0.0-1.0 scale:
1.0 = fully supported by the context
0.5 = partially supported
0.0 = contradicts or fabricates information not in the context

Respond with ONLY a JSON object of the form:
{{"score": <float>, "explanation": "<short reason>"}}"""
    model = _get_judge_model()
    response = await model.generate_content_async(prompt)
    return _parse_judge_json(response.text or "")


async def llm_judge_relevance(query, answer):
    prompt = f"""You are grading whether an AI answer actually addresses the question asked.

Question: {query}

Answer: {answer}

Score the answer's relevance to the question on a 0.0-1.0 scale:
1.0 = directly and completely addresses the question
0.5 = partially addresses the question
0.0 = does not address the question at all

Respond with ONLY a JSON object of the form:
{{"score": <float>, "explanation": "<short reason>"}}"""
    model = _get_judge_model()
    response = await model.generate_content_async(prompt)
    return _parse_judge_json(response.text or "")


async def run_evaluation(eval_dataset, retrieval_fn, generation_fn, k=5):
    precisions = []
    mrrs = []
    faithfulness_scores = []
    relevance_scores = []
    details = []

    for item in eval_dataset:
        query = item["query"]
        relevant_ids = set(item.get("relevant_doc_ids", []))

        chunks = await retrieval_fn(query)
        retrieved_ids = [c.get("id") or c.get("doc_id") for c in chunks]

        p_at_k = precision_at_k(retrieved_ids, relevant_ids, k)
        mrr = mean_reciprocal_rank(retrieved_ids, relevant_ids)

        generation_result = await generation_fn(query, chunks)
        answer = getattr(generation_result, "answer", None)
        if answer is None:
            answer = generation_result.get("answer", "")
        context_str = chr(10).join(c.get("text", "") for c in chunks)

        faithfulness_result = await llm_judge_faithfulness(query, answer, context_str)
        relevance_result = await llm_judge_relevance(query, answer)

        faithfulness_score = float(faithfulness_result.get("score", 0.0))
        relevance_score = float(relevance_result.get("score", 0.0))

        precisions.append(p_at_k)
        mrrs.append(mrr)
        faithfulness_scores.append(faithfulness_score)
        relevance_scores.append(relevance_score)

        details.append({
            "query": query,
            "precision_at_k": p_at_k,
            "mrr": mrr,
            "faithfulness": faithfulness_score,
            "relevance": relevance_score,
            "answer": answer,
        })

    def _avg(values):
        return sum(values) / len(values) if values else 0.0

    return EvalResult(
        precision_at_k=_avg(precisions),
        mrr=_avg(mrrs),
        faithfulness=_avg(faithfulness_scores),
        relevance=_avg(relevance_scores),
        details=details,
    )
