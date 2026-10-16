#!/usr/bin/env python3
"""
run_local_qwen_batch.py
=======================
Runs Qwen 2.5 1.5B (local Ollama instance on Apple Silicon) across all 400 validation posts.
Uses few-shot Chain-of-Thought prompting and canonical 8-way taxonomy labels.
Saves intermediate outputs to qwen_predictions_n400.json every 25 records.
"""

import os
import sys
import json
import time
import urllib.request
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "corpus_metadata.csv")
QWEN_OUT = os.path.join(BASE_DIR, "qwen_predictions_n400.json")

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
MODEL_QWEN = "qwen2.5:1.5b"

SYSTEM_PROMPT = """You are an expert qualitative coding researcher in Human-Computer Interaction (HCI) analyzing algorithmic anxiety and trust breakdown in automated healthcare and financial triage systems.

Classify the following consumer grievance narrative into exactly ONE of these 8 failure taxonomies:
1. Opaque_Algorithmic_Reasoning: Bot issues a denial, triage determination, or fraud freeze without providing interpretable clinical evidence or policy rationale.
2. Recursive_Dead_End_Loops: Bot traps user in repetitive dialogue loops, re-asking previously answered questions or cycling through the same menus.
3. Performative_Uncanny_Empathy: Bot offers formulaic, robotic apologies ("I understand your frustration") while refusing practical assistance during high-stakes emergencies.
4. Context_Amnesia: Bot loses conversation history or user data when transferring across session states or escalating to another tier.
5. Premature_Escalation_Blocking: Bot intentionally conceals human transfer options, disables escalation keywords, or terminates chats to enforce containment.
6. Hallucinated_Policy_Clinical_Errors: Bot outputs clinically inaccurate triage guidance, fabricated insurance terms, or false regulatory claims.
7. Asynchronous_Limbo_Black_Holes: Bot defers issues to unmonitored ticket queues without SLA commitments, status tracking, or callbacks.
8. Biometric_Auth_Lockout_Loops: Bot demands impossible verification (facial scans, SMS codes) during crises (e.g. lost phone, injury), locking out user completely.

FEW-SHOT EXAMPLES:
Example 1:
Narrative: "The insurance chatbot told me my pre-authorization was rejected due to automated rule 402B. It gave zero medical explanation or doctor notes, just a code."
Output: {"taxonomy": "Opaque_Algorithmic_Reasoning", "confidence": 0.95, "reasoning": "Decision issued without clinically interpretable explanation or rationale."}

Example 2:
Narrative: "I answered 'severe chest pain' three times. Each time it asked 'Do you have pain?' and routed me back to the start menu."
Output: {"taxonomy": "Recursive_Dead_End_Loops", "confidence": 0.92, "reasoning": "Bot is stuck in a repetitive dialogue cycle disregarding prior inputs."}

Output MUST be a valid JSON object with keys:
{
  "taxonomy": "<exactly one of the 8 taxonomy keys above>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<concise justification explaining which interaction failure occurred>"
}"""

TAXONOMIES = [
    "Opaque_Algorithmic_Reasoning",
    "Recursive_Dead_End_Loops",
    "Performative_Uncanny_Empathy",
    "Context_Amnesia",
    "Premature_Escalation_Blocking",
    "Hallucinated_Policy_Clinical_Errors",
    "Asynchronous_Limbo_Black_Holes",
    "Biometric_Auth_Lockout_Loops"
]

CANONICAL_MAP = {
    "recursive_containment_loops": "Recursive_Dead_End_Loops",
    "recursive_dead_end_loops": "Recursive_Dead_End_Loops",
    "asynchronous_administrative_limbo": "Asynchronous_Limbo_Black_Holes",
    "asynchronous_limbo_black_holes": "Asynchronous_Limbo_Black_Holes"
}

def call_qwen(text: str):
    prompt = f"{SYSTEM_PROMPT}\n\nGrievance Narrative to Code:\n\"\"\"{text}\"\"\"\n\nReturn JSON:"
    payload = {
        "model": MODEL_QWEN,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 180}
    }
    req = urllib.request.Request(
        OLLAMA_ENDPOINT,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8")
    )
    for _ in range(2):
        try:
            with urllib.request.urlopen(req, timeout=35) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                res = json.loads(data.get("response", "{}"))
                tax = res.get("taxonomy", "")
                if tax in TAXONOMIES:
                    return res
                if tax.lower() in CANONICAL_MAP:
                    res["taxonomy"] = CANONICAL_MAP[tax.lower()]
                    return res
                for k in TAXONOMIES:
                    if k.lower() in tax.lower():
                        res["taxonomy"] = k
                        return res
                res["taxonomy"] = "Opaque_Algorithmic_Reasoning"
                return res
        except Exception as e:
            time.sleep(0.5)
    return {"taxonomy": "Opaque_Algorithmic_Reasoning", "confidence": 0.5, "reasoning": "Fallback on timeout"}

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    n_val = 400
    df_val = df.groupby("primary_taxonomy_label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 50), random_state=42)
    ).head(n_val)
    if len(df_val) < n_val:
        remainder = df[~df["record_id"].isin(df_val["record_id"])].sample(n_val - len(df_val), random_state=42)
        df_val = pd.concat([df_val, remainder])
        
    record_ids = df_val["record_id"].tolist()
    narratives = df_val["narrative_excerpt"].tolist()
    true_labels = df_val["primary_taxonomy_label"].tolist()
    
    print(f"Starting local Qwen 2.5 1.5B batch across {len(df_val)} posts...")
    t0 = time.time()
    results = {}
    
    for i, (rec_id, text, true_lbl) in enumerate(zip(record_ids, narratives, true_labels)):
        res = call_qwen(text)
        results[rec_id] = {
            "record_id": rec_id,
            "true_ground_truth": true_lbl,
            "qwen_prediction": res.get("taxonomy"),
            "qwen_confidence": res.get("confidence", 0.0),
            "qwen_reasoning": res.get("reasoning", "")
        }
        if (i + 1) % 25 == 0 or (i + 1) == len(record_ids):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  Qwen progress: {i+1}/{len(record_ids)} in {elapsed:.1f}s ({rate:.2f} posts/s)", flush=True)
            with open(QWEN_OUT, "w") as f:
                json.dump(results, f, indent=2)
                
    print(f"All {len(record_ids)} posts processed by Qwen 2.5 1.5B. Saved to {QWEN_OUT}")
