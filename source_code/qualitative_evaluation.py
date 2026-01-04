import csv, json, re, argparse
from typing import List, Dict, Optional
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer

"""
The evaluate the model- MD-judge with the scam baiting dataset within federated learning
This code evaluates the AI scam-baiter turns against scammer messages in terms of Novelty, Relevance, Scam risk score, Engagement, PII risk score.
We show the results in the paper at the Table 6 for 30 rounds.
We have evaluation results for three different setting without differential privacy (DP) and with DP (0.1, 0.8).
"""

# ====== Basics ======
def normalize(t: str) -> str:
    if not t: return ""
    t = t.strip().lower()
    return re.sub(r"\s+", " ", t)

def toks(t: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", normalize(t))

def safe_div(a, b, default=0.0): return (a/b) if b else default

# ====== Parrot / Novelty vs. scammer ======
def overlap_fraction(cand: str, src: str) -> float:
    c = toks(cand); sset = set(toks(src))
    if not c: return 0.0
    return sum(1 for x in c if x in sset) / len(c)

def jaccard(cand: str, src: str) -> float:
    C, S = set(toks(cand)), set(toks(src))
    if not C or not S: return 0.0
    return len(C & S) / len(C | S)

def novelty_vs_scammer(cand: str, scammer: str) -> float:
    # 1 - mean(overlap, jaccard), clipped to [0,1]
    base = (overlap_fraction(cand, scammer) + jaccard(cand, scammer)) / 2
    return max(0.0, 1.0 - base)

# ====== Engagement heuristics ======
def lexical_diversity(txt: str) -> float:
    T = toks(txt)
    return safe_div(len(set(T)), len(T))

def length_score(txt: str, min_len=5, max_len=80) -> float:
    n = len(toks(txt))
    if n <= 0: return 0.0
    if n < min_len: return max(0.0, n/min_len * 0.6)
    if n > max_len: return max(0.2, 1.0 - (n-max_len)/max_len)
    mid = (min_len + max_len)/2
    return max(0.6, 1.0 - abs(n-mid)/mid * 0.4)

def engagement_score(txt: str) -> float:
    ls = length_score(txt)
    ld = min(1.0, lexical_diversity(txt)/0.7)
    qb = 0.1 if "?" in txt else 0.0
    return min(1.0, max(0.0, 0.5*ls + 0.4*ld + qb))

def bertscore_f1(cand: str, ref: str) -> Optional[float]:
    try:
        P, R, F1 = bert_score([cand], [ref], lang="en", rescale_with_baseline=True)
        return float(F1[0].item())
    except Exception as e:
        print("[warn] BERTScore not available:", e)
        return None
    
def cosine(u, v):
    import numpy as np
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu==0 or nv==0: return 0.0
    return float(np.dot(u, v)/(nu*nv))

def relevance_scammer_to_ai(model, scammer: str, cand: str) -> Optional[float]:
    if model is None: return None
    vecs = model.encode([scammer, cand], normalize_embeddings=False)
    # map [-1,1] -> [0,1]
    score = (cosine(vecs[0], vecs[1]) + 1)/2
    # print("Relevance: ", score)
    return score

# ====== Composite ======
def composite(
    bert_f1: Optional[float],
    rel_s: Optional[float],
    engage_s: float,
    novelty_s: float,
    w_bert=0.25, w_rel=0.10, w_eng=0.10, w_nov=0.10
) -> float:
    parts, weights = [], []
    if bert_f1  is not None: parts.append(bert_f1);  weights.append(w_bert)
    if rel_s    is not None: parts.append(rel_s);    weights.append(w_rel)
    parts.append(engage_s); weights.append(w_eng)
    parts.append(novelty_s);weights.append(w_nov)
    s = sum(weights)
    weights = [w/s for w in weights]
    return float(sum(w*p for w,p in zip(weights, parts)))

# ====== I/O ======
def read_rows(path: str) -> List[Dict]:
    if path.endswith(".jsonl"):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
        return out
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_csv(path: str, rows: List[Dict]):
    if not rows: return
    fields = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader(); wr.writerows(rows)

# ====== Main ======
def main(data_path: str, out_path: str, use_bertscore: bool = True, use_rel: bool = True):
    rows = read_rows(data_path)

    out = []
    agg = {"novelty":0,"engagement":0,"bertscore":0,"relevance":0,"composite":0}
    n = 0

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    for i, r in enumerate(rows):
        rnd = r.get("round", "")
        conv_id = r.get("conv_id", "")
        turn_id = r.get("turn_id", "")
        scammer   = r.get("scammer", None)
        reference = r.get("reference",None)
        cand      = r.get("ai_response",None)
        scam_risk = r.get("scam_risk", None)
        engage_score = r.get("engagement_score", None)
        pii_risk = r.get("pii_risk", None)

        # print(f"Round: {rnd}, Conv ID: {conv_id}, Turn ID: {turn_id}")

        # if scammer == None or reference == None or cand == None \
        # or scam_risk == None or engage_score ==  None or pii_risk == None:
        #     continue

        nov = novelty_vs_scammer(cand, scammer)
        eng = engagement_score(cand)
        bs  = bertscore_f1(cand, reference) if use_bertscore else None
        rel = relevance_scammer_to_ai(model, scammer, cand) if use_rel else None

        # comp = composite(blt, bs, rel, eng, nov)

        row = {
            "idx": i,
            "round": rnd,
            "conv_id": conv_id,
            "turn_id": turn_id,
            "scammer": scammer,
            "reference": reference,
            "ai_response": cand,
            "novelty_vs_scammer": round(nov,4),
            "engagement": round(eng,4),
            "bertscore_f1": round(bs,4) if bs is not None else "",
            "relevance_scammer_ai": round(rel,4) if rel is not None else "",
            # "composite": round(comp,4),
            "scam_risk": scam_risk,
            "engagement_score": engage_score,
            "pii_risk": pii_risk
        }
        out.append(row)

        # aggregates
        n += 1
        agg["novelty"]   += nov
        agg["engagement"]+= eng
        if bs  is not None: agg["bertscore"] += bs
        if rel is not None: agg["relevance"] += rel
        # agg["composite"] += comp

    write_csv(out_path, out)

    def avg(x, denom): return (x/denom) if denom else 0.0
    print(f"[ok] wrote {out_path} ({n} rows)")
    print("Averages (on available values):")
    print("  novelty_vs_scammer:", round(avg(agg["novelty"], n), 4))
    print("  engagement:",         round(avg(agg["engagement"], n), 4))
    print("  bertscore_f1:",       "n/a" if use_bertscore is False else round(avg(agg["bertscore"], n), 4))
    print("  relevance_scammer_ai:", "n/a" if use_rel is False else round(avg(agg["relevance"], n), 4))
    # print("  composite:",          round(avg(agg["composite"], n), 4))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate AI scam-baiter turns against reference and scammer msg.")
    ap.add_argument("--data", required=True, help="CSV or JSONL with fields: scammer,reference,ai_response")
    ap.add_argument("--out", required=True, help="Output CSV filepath")
    ap.add_argument("--no_bertscore", action="store_true", help="Disable BERTScore")
    ap.add_argument("--no_relevance", action="store_true", help="Disable reference-free relevance")
    args = ap.parse_args()
    main(args.data, args.out,
         use_bertscore=(not args.no_bertscore), use_rel=(not args.no_relevance))

# CUDA_LAUNCH_BLOCKING=3 python qualitative_evaluation.py --data ai-in-the-loop/results/reports/turns.csv --out ai-in-the-loop/results/reports/scores.csv --no_bertscore
