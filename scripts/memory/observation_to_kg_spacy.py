import argparse
import csv
import json
from pathlib import Path


def load_json(file_path):
    with Path(file_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_spacy():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")


AUX_VERBS = {
    "is", "are", "was", "were", "been", "being", "am",
    "has", "have", "had", "having",
    "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "can", "could", "must",
    "get", "gets", "got", "gotten",
    "feel", "feels", "felt",
    "find", "finds", "found",
    "believe", "believes", "believed",
    "think", "thinks", "thought",
    "know", "knows", "knew",
    "want", "wants", "wanted",
    "need", "needs", "needed",
    "like", "likes", "liked",
    "love", "loves", "loved",
    "realize", "realizes", "realized",
    "hope", "hopes", "hoped",
    "become", "becomes", "became",
    "seem", "seems", "seemed",
    "look", "looks", "looked",
    "keep", "keeps", "kept",
    "let", "lets",
    "make", "makes", "made",
    "help", "helps", "helped",
    "say", "says", "said",
    "tell", "tells", "told",
    "mean", "means", "meant",
}


def get_subtree_text(token):
    tokens = sorted(token.subtree, key=lambda t: t.i)
    return " ".join(t.text.lower() for t in tokens)


def get_span_text(tokens):
    return " ".join(t.text.lower() for t in tokens)


def extract_main_verb(doc):
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_ not in AUX_VERBS:
            return token
    for token in doc:
        if token.pos_ == "VERB":
            return token
    return None


def extract_subject(verb_token):
    for child in verb_token.children:
        if child.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass"):
            return get_subtree_text(child)
    return None


def extract_direct_object(verb_token):
    objs = []
    for child in verb_token.children:
        if child.dep_ in ("dobj", "obj"):
            objs.append(get_subtree_text(child))
    if objs:
        return objs[0]

    for child in verb_token.children:
        if child.dep_ in ("attr", "ccomp", "xcomp", "acomp"):
            if child.pos_ in ("NOUN", "PROPN"):
                return get_subtree_text(child)

    for child in verb_token.children:
        if child.dep_ == "prep" and child.pos_ == "ADP":
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    prep_text = child.lemma_.lower()
                    obj_text = get_subtree_text(grandchild)
                    return f"{prep_text}_{obj_text.replace(' ', '_')}"

    for child in verb_token.rights:
        if child.pos_ in ("NOUN", "PROPN") and child.dep_ not in ("nsubj", "nsubjpass"):
            return get_subtree_text(child)

    return None


def extract_ner_entities(doc):
    entities = {}
    for ent in doc.ents:
        label = ent.label_
        text = ent.text.lower().replace(" ", "_")
        if label == "PERSON":
            entities.setdefault("person", []).append(text)
        elif label in ("ORG", "GPE", "LOC", "FAC"):
            entities.setdefault("organization", []).append(text)
        elif label in ("DATE", "TIME"):
            entities.setdefault("date", []).append(text)
        elif label in ("EVENT", "WORK_OF_ART"):
            entities.setdefault("event", []).append(text)
        else:
            entities.setdefault("entity", []).append(text)
    return entities


def clean_id(text):
    if not text:
        return None
    words = text.replace("-", " ").replace("/", " ").split()
    return "_".join(words[:8]).lower()


def extract_kg_edges(data):
    nlp = load_spacy()
    samples = data if isinstance(data, list) else data.get("data", data.get("samples", []))
    edges = []

    for sample in samples:
        if not isinstance(sample, dict):
            continue
        graph_id = sample.get("sample_id", "unknown")
        observations = sample.get("observation", {})

        for obs_key, obs_value in observations.items():
            if not isinstance(obs_value, dict):
                continue
            for speaker_name, obs_list in obs_value.items():
                if not isinstance(obs_list, list):
                    continue
                for item in obs_list:
                    if not isinstance(item, list) or len(item) < 2:
                        continue
                    fact_text = str(item[0]).strip()
                    evidence_id = str(item[1]).strip()
                    doc = nlp(fact_text)
                    src_id = speaker_name.lower().replace(" ", "_")

                    main_verb = extract_main_verb(doc)
                    if main_verb is None:
                        first_nouns = [t.lemma_ for t in doc if t.pos_ in ("NOUN", "PROPN")]
                        dst_id = "_".join(first_nouns[:5]) if first_nouns else "memory"
                        relation = "has_memory"
                        dst_type = "entity"
                    else:
                        relation = main_verb.lemma_
                        obj_text = extract_direct_object(main_verb)
                        if obj_text:
                            dst_id = clean_id(obj_text)
                        else:
                            dst_id = relation
                        dst_type = "entity"

                    if not dst_id:
                        dst_id = relation

                    edges.append({
                        "graph_id": graph_id,
                        "src_id": src_id,
                        "src_type": "person",
                        "relation": relation,
                        "dst_id": dst_id,
                        "dst_type": dst_type,
                        "confidence": 1.0,
                        "source": "locomo_observation_spacy",
                        "evidence": evidence_id,
                    })

    return edges


def write_csv(file_path, records, fieldnames):
    output_dir = Path(file_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(file_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(
        description="Convert LoCoMo observations to KG edges using spaCy dependency parsing."
    )
    parser.add_argument("--file", default="external_data/locomo10.json")
    parser.add_argument("--output", default="results/locomo_kg_edges_spacy.csv")
    args = parser.parse_args()

    data = load_json(args.file)
    edges = extract_kg_edges(data)

    fieldnames = [
        "graph_id", "src_id", "src_type", "relation",
        "dst_id", "dst_type", "confidence", "source", "evidence",
    ]
    write_csv(args.output, edges, fieldnames)

    relation_counts = {}
    for e in edges:
        rel = e["relation"]
        relation_counts[rel] = relation_counts.get(rel, 0) + 1

    print(f"Total KG edges: {len(edges)}")
    print(f"Top 20 relations:")
    for rel, count in sorted(relation_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {rel:25s} {count}")

    has_memory = relation_counts.get("has_memory", 0)
    print(f"Non-has_memory edges: {len(edges) - has_memory} ({(len(edges)-has_memory)/len(edges)*100:.1f}%)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
