"""Baoulé question in, French documents out: retrieval with the Abyshire Baoulé–French embeddings.

The embedding model puts Baoulé and French sentences in one vector space, so a question asked in
Baoulé finds French documents (and the other way round) without translating anything first. This
example indexes a few short French notes in LangChain's ``InMemoryVectorStore`` and searches them
with Baoulé questions.

The embeddings (``abyshire/embed-baoule``) are served by the Abyshire API and are available by
approval: set ``IVOIRE_API_KEY`` to an approved key, then::

    python examples/baoule_embeddings_rag.py
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_ivoire import IvoireEmbeddings

# French notes to search (written for this example).
DOCS = [
    (
        "eau",
        "Pour avoir de l'eau potable, faites bouillir l'eau du puits ou filtrez-la avant de la boire.",
    ),
    (
        "repas",
        "Le repas du soir est servi à 19 heures ; ce soir il y a du foutou banane et de la sauce graine.",
    ),
    (
        "famille",
        "La famille Kouassi a quatre enfants : deux garçons et deux filles, tous à l'école primaire.",
    ),
    ("voyage", "Le car pour Bouaké part à 6 heures de la gare ; prévoyez de l'eau pour la route."),
    ("sommeil", "Les enfants doivent se coucher tôt pour être en forme à l'école le lendemain."),
    (
        "salutations",
        "Le matin, on salue les anciens en premier et on demande des nouvelles de la famille.",
    ),
    ("vin", "Le vin de palme se boit frais ; on le sert aux invités lors des fêtes au village."),
]

# Baoulé questions (from the package's phrasebook) and the note each one should find.
QUESTIONS = [
    ("Nian, nzue we dɛ.", "I am thirsty.", "eau"),
    ("Awe kuun min.", "I am hungry.", "repas"),
    ("An lɛ ba nnyɛ?", "How many children do you have?", "famille"),
    ("Amun wun jɔ amun fɔuun.", "Have a safe journey.", "voyage"),
    ("Kɔ ko la.", "Go to sleep.", "sommeil"),
    ("Man min nzan'n.", "Pass me the wine.", "vin"),
]


def main() -> int:
    store = InMemoryVectorStore(embedding=IvoireEmbeddings())  # abyshire/embed-baoule
    store.add_documents([Document(page_content=t, metadata={"topic": k}) for k, t in DOCS])
    hits_at_1 = 0
    for bci, eng, expected in QUESTIONS:
        results = store.similarity_search_with_score(bci, k=2)
        hits_at_1 += results[0][0].metadata["topic"] == expected
        print(f"\n{bci}   ({eng})   expected: {expected}")
        for doc, score in results:
            print(f"   {score:.3f}  [{doc.metadata['topic']}] {doc.page_content}")
    print(f"\ntop-1 correct: {hits_at_1}/{len(QUESTIONS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
