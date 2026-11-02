#!/usr/bin/env python3
"""Minimal demonstration of the ``Retrieval`` helper for ontology lookups."""

from retrieval import Retrieval

if __name__ == "__main__":
    r = Retrieval(artifacts_dir="./artifacts", model="text-embedding-3-large")
    print("\nTop classes for: 'sports and leisure industry'")
    for hit in r.search_classes("sports and leisure industry", k=5):
        print(f"{hit['score']:.3f}  {hit['label']}  {hit['iri']}")

    print("\nTop predicates for: 'serving industry' (assuming Organization -> Industry)")
    subj = ["https://spec.industrialontologies.org/ontology/core/Core/Organization"]
    obj  = ["https://spec.industrialontologies.org/ontology/supplychain/SupplyChain/Industry"]
    for hit in r.search_predicates("serving industry", k=5, subj_types=subj, obj_types=obj):
        print(f"{hit['score']:.3f}  {hit['label']}  {hit['iri']}")
