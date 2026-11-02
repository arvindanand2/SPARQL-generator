#!/usr/bin/env python3
"""
hybrid_detector_hybridbm25_toggle.py
------------------------------------
Hybrid detector with a toggle to enable/disable BM25 blending.

- Uses lexical tries + spaCy to find mentions
- Retrieves cosine hits via retrieval.Retrieval (cosine + schema-aware)
- Optionally computes BM25 scores via bm25_index.BM25Store
- Blends scores (alpha * BM25 + beta * cosine + gamma * context)
- Emits a semantic sketch

Constructor args:
  artifacts_dir: path to artifacts dir
  model: embedding model for Retrieval
  spacy_model: spaCy model name
  use_bm25: bool, default True — set False to see cosine-only behavior
  alpha, beta, gamma: blending weights (floats)

Usage:
  from hybrid_detector_hybridbm25_toggle import HybridDetector
  hd = HybridDetector("./artifacts", use_bm25=False)  # cosine-only
"""

import re, json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from retrieval import Retrieval               # cosine + schema
from bm25_index import BM25Store             # BM25 here (optional)
import spacy

def norm_text(s: str) -> str:
    """Lowercase and collapse whitespace for trie comparisons."""
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

class Trie:
    """Token-level trie storing payloads for lexical lookup."""

    def __init__(self):
        """Initialize empty trie nodes with a sentinel end marker."""
        self.root = {}; self.end = "_end_"
    def insert(self, phrase: str, payload: Any):
        """Insert a whitespace-tokenized phrase and attach the payload."""
        node = self.root
        for t in phrase.split():
            node = node.setdefault(t, {})
        node[self.end] = node.get(self.end, []); node[self.end].append(payload)
    def find_spans(self, tokens: List[str]):
        """Return matching spans (start, end, payloads) for the provided tokens."""
        hits = []; n=len(tokens); i=0
        while i<n:
            node=self.root; j=i; last=None
            while j<n and tokens[j] in node:
                node=node[tokens[j]]; j+=1
                if self.end in node: last=(i,j,node[self.end])
            if last: hits.append(last); i=last[1]
            else: i+=1
        return hits

class HybridDetector:
    """Blend lexical, cosine, and BM25 signals to map mentions to ontology IRIs."""

    def __init__(self, artifacts_dir: str, model: str = "text-embedding-3-large",
                 spacy_model: str = "en_core_web_sm",
                 use_bm25: bool = True,
                 alpha: float = 0.45, beta: float = 0.45, gamma: float = 0.10):
        """Initialize retrieval resources, BM25 store, and spaCy pipeline."""
        self.artifacts_dir = artifacts_dir
        self.model = model
        self.use_bm25 = use_bm25
        # Blending weights
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

        # Cosine retriever
        self.retrieval = Retrieval(artifacts_dir=self.artifacts_dir, model=self.model)

        # Optional BM25 store
        self.bm25 = BM25Store(artifacts_dir=self.artifacts_dir) if self.use_bm25 else None

        # Records for tries / metadata
        self.class_records = self.retrieval.class_records
        self.pred_records  = self.retrieval.pred_records

        self.trie_classes, self.trie_preds = self._build_tries()

        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            raise RuntimeError(f"spaCy model '{spacy_model}' not found. Run: python -m spacy download {spacy_model}")

    # ---- tries ----
    def _build_tries(self) -> Tuple[Trie,Trie]:
        """Create tries populated with labels and aliases for classes/predicates."""
        tc,tp = Trie(),Trie()
        for r in self.class_records:
            cands = [r.get("label",""), r.get("lexical_text","")] + r.get("aliases_enriched", [])
            for c in cands:
                c = norm_text(c);
                if c: tc.insert(c, {"iri": r.get("iri"), "kind":"class", "label": r.get("label","")})
        for r in self.pred_records:
            cands = [r.get("label",""), r.get("lexical_text","")] + r.get("aliases_enriched", [])
            for c in cands:
                c = norm_text(c);
                if c: tp.insert(c, {"iri": r.get("iri"), "kind":"predicate", "label": r.get("label","")})
        return tc,tp

    # ---- tiny context overlap (0..1) ----
    def _context_sim(self, a: str, b: str) -> float:
        """Compute a simple token-overlap score between two snippets."""
        A=set(norm_text(a).split()); B=set(norm_text(b).split())
        return (len(A & B)/max(1,len(A|B))) if A and B else 0.0

    def detect_mentions(self, text: str) -> List[Dict[str, Any]]:
        """Use tries, noun chunks, and dependency heuristics to extract mentions."""
        doc = self.nlp(text); toks=[t.text.lower() for t in doc]
        mentions=[]
        for start,end,payloads in self.trie_classes.find_spans(toks):
            span=" ".join(toks[start:end])
            for p in payloads: mentions.append({"span":span,"kind":"class","ctx":text,"source":"trie","label":p["label"]})
        for start,end,payloads in self.trie_preds.find_spans(toks):
            span=" ".join(toks[start:end])
            for p in payloads: mentions.append({"span":span,"kind":"predicate","ctx":text,"source":"trie","label":p["label"]})
        for np in doc.noun_chunks:
            s=np.text.strip()
            if len(s)>=2: mentions.append({"span":s,"kind":"class","ctx":doc[max(0,np.start-3):min(len(doc),np.end+3)].text,"source":"np"})
        for tok in doc:
            if tok.pos_ in {"VERB","ADP"}:
                span=" ".join([t.text for t in tok.subtree])
                mentions.append({"span":span,"kind":"predicate","ctx":doc[max(0,tok.i-3):min(len(doc),tok.i+4)].text,"source":"dep"})
        dedup={}
        for m in mentions:
            key=(norm_text(m["span"]), m["kind"])
            if key not in dedup: dedup[key]=m
        return list(dedup.values())

    def _direction_hint(self, span: str) -> str:
        """Guess predicate direction based on surface text cues."""
        s=span.lower()
        if " by " in s or s.startswith("by "): return "inverse"
        if s.startswith("located in") or " located in " in s or s.startswith("in "): return "forward"
        return "forward"

    def map_mention(self, mention: Dict[str,Any], subj_types: Optional[List[str]] = None, obj_types: Optional[List[str]] = None, k: int = 15) -> Dict[str,Any]:
        """Rank candidate IRIs for a single mention, optionally blending BM25.

        Parameters
        ----------
        mention : dict
            Mention dictionary produced by ``detect_mentions``.
        subj_types : list[str] | None
            Optional subject type hints for predicate searches.
        obj_types : list[str] | None
            Optional object type hints for predicate searches.
        k : int, default 15
            Number of candidates to keep when blending signals.

        Returns
        -------
        dict
            Candidate summary containing ``top`` choice and alternates.
        """
        kind = mention["kind"]; query = mention["span"]

        # 1) cosine candidates via Retrieval
        if kind=="class":
            cos_hits = self.retrieval.search_classes(query, k=max(k,25))
        else:
            cos_hits = self.retrieval.search_predicates(query, k=max(k,25),
                                                        subj_types=subj_types, obj_types=obj_types)

        records = self.class_records if kind=="class" else self.pred_records

        # If BM25 disabled, just apply small context boost and return
        if not self.use_bm25 or self.bm25 is None:
            def _ctx_boost(h):
                """Apply a light context overlap boost to cosine scores."""
                return h["score"] + self.gamma * self._context_sim(mention.get("ctx",""), h.get("label",""))
            hits = sorted(cos_hits, key=_ctx_boost, reverse=True)[:k]
            direction = None
            if kind=="predicate":
                direction = "o->s" if self._direction_hint(query)=="inverse" else "s->o"
            results = [{
                "iri": h["iri"],
                "label": h["label"],
                "score": float(_ctx_boost(h)),
                "raw": h["record"]
            } for h in hits]
            return {
                "mention": mention,
                "kind": kind,
                "direction": direction,
                "candidates": results,
                "top": results[0] if results else None,
                "alternates": results[1:4] if len(results) > 1 else []
            }

        # 2) Otherwise, compute BM25 scores and blend
        bm_scores = self.bm25.get_scores("class" if kind=="class" else "predicate", query)
        idx_by_iri = {r.get("iri"): i for i, r in enumerate(records)}

        merged = {}
        for h in cos_hits:
            iri = h["iri"]; i = idx_by_iri.get(iri)
            if i is None: continue
            bm = bm_scores[i]
            bm_norm = bm / (bm + 1.5) if bm > 0 else 0.0
            base = self.alpha*bm_norm + self.beta*((h["score"] + 1.0)/2.0)
            base += self.gamma * self._context_sim(mention.get("ctx",""), h.get("label",""))
            merged[iri] = {"score": base, "label": h["label"], "idx": i}

        # Also bring top BM25-only items not seen in cosine top list
        top_bm_idx = sorted(range(len(bm_scores)), key=lambda i: bm_scores[i], reverse=True)[:k]
        for i in top_bm_idx:
            iri = records[i].get("iri")
            if iri in merged: continue
            bm = bm_scores[i]
            bm_norm = bm / (bm + 1.5) if bm > 0 else 0.0
            label = records[i].get("label","")
            score = self.alpha*bm_norm + self.gamma*self._context_sim(mention.get("ctx",""), label)
            merged[iri] = {"score": score, "label": label, "idx": i}

        ordered = sorted(merged.items(), key=lambda kv: kv[1]["score"], reverse=True)[:k]
        direction = None
        if kind=="predicate":
            direction = "o->s" if self._direction_hint(query)=="inverse" else "s->o"
        results = [{
            "iri": iri,
            "label": meta["label"],
            "score": float(meta["score"]),
            "raw": records[meta["idx"]]
        } for iri, meta in ordered]

        return {
            "mention": mention,
            "kind": kind,
            "direction": direction,
            "candidates": results,
            "top": results[0] if results else None,
            "alternates": results[1:4] if len(results) > 1 else []
        }

    def detect_and_map(self, text: str) -> Dict[str, Any]:
        """Detect mentions in free text and return a compact semantic sketch."""
        mentions = self.detect_mentions(text)
        class_maps = [self.map_mention(m, k=10) for m in mentions if m["kind"]=="class"]
        seen_cls=[];
        for cm in class_maps:
            if cm["top"] and cm["top"]["iri"] and cm["top"]["iri"] not in seen_cls:
                seen_cls.append(cm["top"]["iri"])
        subj_types = [seen_cls[0]] if seen_cls else []
        obj_types  = [seen_cls[1]] if len(seen_cls)>1 else []
        pred_maps = [self.map_mention(m, subj_types=subj_types, obj_types=obj_types, k=10)
                     for m in mentions if m["kind"]=="predicate"]

        classes_out=[]; seen_out=set()
        for cm in class_maps:
            if cm["top"] and cm["top"]["iri"] not in seen_out:
                classes_out.append({"span": cm["mention"]["span"], "iri": cm["top"]["iri"], "label": cm["top"]["label"], "conf": round(cm["top"]["score"],3)})
                seen_out.add(cm["top"]["iri"])
        predicates_out=[]; alts=[]
        for pm in pred_maps:
            if pm["top"]:
                predicates_out.append({"span": pm["mention"]["span"], "iri": pm["top"]["iri"], "label": pm["top"]["label"], "dir": pm["direction"] or "s->o", "conf": round(pm["top"]["score"],3)})
                for a in pm["alternates"]:
                    alts.append({"iri": a["iri"], "label": a["label"]})

        return {
            "classes": classes_out[:3],
            "predicates": predicates_out[:3],
            "alternates": {"predicates": alts[:5]},
            "notes": ["direction 'o->s' means inverse phrasing (e.g., 'made by').",
                      f"bm25_enabled={self.use_bm25}, weights(alpha={self.alpha}, beta={self.beta}, gamma={self.gamma})"]
        }
