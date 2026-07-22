import argparse
import csv
import json
import re
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


def extract_main_verb(doc):
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_ not in AUX_VERBS:
            return token
    for token in doc:
        if token.pos_ == "VERB":
            return token
    return None


def extract_direct_object(verb_token):
    for child in verb_token.children:
        if child.dep_ in ("dobj", "obj"):
            return get_subtree_text(child)

    for child in verb_token.children:
        if child.dep_ in ("attr", "ccomp", "xcomp", "acomp"):
            if child.pos_ in ("NOUN", "PROPN"):
                return get_subtree_text(child)

    for child in verb_token.children:
        if child.dep_ == "prep" and child.pos_ == "ADP":
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    return f"{child.lemma_.lower()}_{get_subtree_text(grandchild).replace(' ', '_')}"

    for child in verb_token.rights:
        if child.pos_ in ("NOUN", "PROPN") and child.dep_ not in ("nsubj", "nsubjpass"):
            return get_subtree_text(child)

    return None


def clean_id(text):
    if not text:
        return None
    words = re.sub(r"[^a-zA-Z0-9 ]", " ", text.lower()).split()
    return "_".join(words[:6])


def extract_observation_edges(nlp, sample):
    graph_id = sample.get("sample_id", "unknown")
    observations = sample.get("observation", {})
    edges = []

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
                else:
                    relation = main_verb.lemma_
                    obj_text = extract_direct_object(main_verb)
                    dst_id = clean_id(obj_text) if obj_text else relation

                edges.append({
                    "graph_id": graph_id,
                    "src_id": src_id,
                    "src_type": "person",
                    "relation": relation,
                    "dst_id": dst_id,
                    "dst_type": "entity",
                    "confidence": 1.0,
                    "source": "locomo_observation_spacy",
                    "evidence": evidence_id,
                })

    return edges


def build_session_texts(conversation):
    session_turns = {}
    for key, value in conversation.items():
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue
        if not isinstance(value, list):
            continue
        texts = []
        for turn in value:
            if isinstance(turn, dict):
                texts.append({
                    "dia_id": turn.get("dia_id", ""),
                    "text": turn.get("text", "").lower(),
                })
        if texts:
            session_turns[key] = texts
    return session_turns


def match_fact_to_turn(fact_words, session_turns, min_overlap=1):
    best_dia_id = ""
    best_overlap = 0
    fact_set = set(fact_words)
    if len(fact_set) < 2:
        return "", []
    for turn in session_turns:
        turn_words = set(re.findall(r"[a-z]+", turn["text"]))
        overlap = len(fact_set & turn_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_dia_id = turn["dia_id"]
    if best_overlap >= min_overlap:
        return best_dia_id, [best_dia_id]
    all_dia_ids = [t["dia_id"] for t in session_turns]
    return "", all_dia_ids


def extract_event_summary_edges(nlp, sample, session_turns):
    graph_id = sample.get("sample_id", "unknown")
    event_summary = sample.get("event_summary", {})
    edges = []

    for es_key, es_value in event_summary.items():
        if not isinstance(es_value, dict):
            continue
        session_num_match = re.search(r"session_(\d+)", es_key)
        if not session_num_match:
            continue
        session_key = f"session_{session_num_match.group(1)}"
        turns = session_turns.get(session_key, [])

        for person_name, events in es_value.items():
            if person_name == "date" or not isinstance(events, list):
                continue
            for event_text in events:
                if not isinstance(event_text, str) or not event_text.strip():
                    continue
                event_text = event_text.strip()
                doc = nlp(event_text)
                src_id = person_name.lower().replace(" ", "_")

                fact_words = [t.lemma_.lower() for t in doc if t.pos_ in ("NOUN", "VERB", "PROPN")]
                evidence_id, all_ev_ids = match_fact_to_turn(fact_words, turns)
                evidence_list = [evidence_id] if evidence_id else all_ev_ids

                main_verb = extract_main_verb(doc)
                if main_verb is None:
                    first_nouns = [t.lemma_ for t in doc if t.pos_ in ("NOUN", "PROPN")]
                    dst_id = "_".join(first_nouns[:5]) if first_nouns else "event"
                    relation = "has_event"
                else:
                    relation = main_verb.lemma_
                    obj_text = extract_direct_object(main_verb)
                    dst_id = clean_id(obj_text) if obj_text else relation

                for ev_id in evidence_list:
                    edges.append({
                        "graph_id": graph_id,
                        "src_id": src_id,
                        "src_type": "person",
                        "relation": relation,
                        "dst_id": dst_id,
                        "dst_type": "entity",
                        "confidence": 1.0,
                        "source": "locomo_event_summary",
                        "evidence": ev_id,
                    })

    return edges


def extract_session_summary_edges(nlp, sample, session_turns):
    graph_id = sample.get("sample_id", "unknown")
    session_summary = sample.get("session_summary", {})
    edges = []

    for ss_key, ss_text in session_summary.items():
        if not isinstance(ss_text, str) or not ss_text.strip():
            continue
        session_num_match = re.search(r"session_(\d+)", ss_key)
        if not session_num_match:
            continue
        session_key = f"session_{session_num_match.group(1)}"
        turns = session_turns.get(session_key, [])

        sentences = re.split(r"(?<=[.!?])\s+", ss_text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            doc = nlp(sentence)

            speaker_found = ""
            for token in doc:
                if token.text.lower() in ("caroline", "melanie"):
                    speaker_found = token.text.lower()
                    break

            if not speaker_found:
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        speaker_found = ent.text.lower()
                        break

            if not speaker_found:
                continue

            fact_words = [t.lemma_.lower() for t in doc if t.pos_ in ("NOUN", "VERB", "PROPN")]
            evidence_id, all_ev_ids = match_fact_to_turn(fact_words, turns)
            evidence_list = [evidence_id] if evidence_id else all_ev_ids

            main_verb = extract_main_verb(doc)
            if main_verb is None:
                first_nouns = [t.lemma_ for t in doc if t.pos_ in ("NOUN", "PROPN")]
                dst_id = "_".join(first_nouns[:5]) if first_nouns else "summary"
                relation = "summary_mentions"
            else:
                relation = main_verb.lemma_
                obj_text = extract_direct_object(main_verb)
                dst_id = clean_id(obj_text) if obj_text else relation

            for ev_id in evidence_list:
                edges.append({
                    "graph_id": graph_id,
                    "src_id": speaker_found,
                    "src_type": "person",
                    "relation": relation,
                    "dst_id": dst_id,
                    "dst_type": "entity",
                    "confidence": 1.0,
                    "source": "locomo_session_summary",
                    "evidence": ev_id,
                })

    return edges


def extract_all_edges(data):
    nlp = load_spacy()
    samples = data if isinstance(data, list) else data.get("data", data.get("samples", []))
    all_edges = []

    for sample in samples:
        if not isinstance(sample, dict):
            continue

        obs_edges = extract_observation_edges(nlp, sample)
        all_edges.extend(obs_edges)

        session_turns = build_session_texts(sample.get("conversation", {}))

        event_edges = extract_event_summary_edges(nlp, sample, session_turns)
        all_edges.extend(event_edges)

        summary_edges = extract_session_summary_edges(nlp, sample, session_turns)
        all_edges.extend(summary_edges)

    return all_edges


def write_csv(file_path, records, fieldnames):
    output_dir = Path(file_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(file_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(
        description="Extended KG extraction: observations + event_summary + session_summary."
    )
    parser.add_argument("--file", default="external_data/locomo10.json")
    parser.add_argument("--output", default="results/locomo_kg_edges_extended.csv")
    args = parser.parse_args()

    data = load_json(args.file)
    edges = extract_all_edges(data)

    fieldnames = [
        "graph_id", "src_id", "src_type", "relation",
        "dst_id", "dst_type", "confidence", "source", "evidence",
    ]
    write_csv(args.output, edges, fieldnames)

    from collections import Counter
    src_counts = Counter(e["source"] for e in edges)
    ev_ok = sum(1 for e in edges if e["evidence"])
    ev_none = sum(1 for e in edges if not e["evidence"])
    unique_pairs = len(set((e["graph_id"], e["evidence"]) for e in edges if e["evidence"]))

    print(f"Total KG edges: {len(edges)}")
    print(f"  Source breakdown:")
    for src, cnt in src_counts.most_common():
        print(f"    {src:30s} {cnt}")
    print(f"  Edges with evidence: {ev_ok}, without: {ev_none}")
    print(f"  Unique (graph, evidence) pairs: {unique_pairs}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
