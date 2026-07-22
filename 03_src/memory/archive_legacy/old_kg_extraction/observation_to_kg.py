import argparse
import csv
import json
import re
from pathlib import Path


VERB_PATTERNS = [
    ("attended", "attended", "event"),
    ("gave a talk", "gave_talk", "event"),
    ("gave a speech", "gave_talk", "event"),
    ("spoke at", "gave_talk", "event"),
    ("painted", "painted", "object"),
    ("organized", "organized", "event"),
    ("volunteered at", "volunteered_at", "event"),
    ("volunteered for", "volunteered_for", "organization"),
    ("signed up for", "enrolled_in", "activity"),
    ("started transitioning", "started", "activity"),
    ("started working", "started_work", "activity"),
    ("started a new", "started", "activity"),
    ("adopted a", "adopted", "activity"),
    ("adopted an", "adopted", "activity"),
    ("is researching", "researched", "topic"),
    ("researched", "researched", "topic"),
    ("is planning to", "planning", "activity"),
    ("planning to", "planning", "activity"),
    ("is considering", "considering", "activity"),
    ("considering a career in", "considering_career", "activity"),
    ("explore career options in", "exploring_career", "activity"),
    ("ran a charity", "participated_in", "event"),
    ("ran a marathon", "participated_in", "event"),
    ("went to", "visited", "location"),
    ("visited", "visited", "location"),
    ("moved to", "moved_to", "location"),
    ("traveled to", "traveled_to", "location"),
    ("studied", "studied", "topic"),
    ("read", "read", "object"),
    ("wrote", "wrote", "object"),
    ("joined", "joined", "group"),
    ("cooked", "cooked", "object"),
    ("baked", "baked", "object"),
    ("built", "built", "object"),
    ("created", "created", "object"),
    ("donated", "donated", "activity"),
    ("volunteered", "volunteered", "activity"),
    ("watched", "watched", "event"),
    ("listened to", "listened_to", "object"),
    ("played", "played", "activity"),
    ("swimming with", "went_swimming", "activity"),
    ("going camping", "going_camping", "activity"),
    ("going hiking", "going_hiking", "activity"),
    ("going to the beach", "going_to_beach", "activity"),
    ("is taking a", "taking", "activity"),
    ("took a", "taking", "activity"),
    ("took her", "took_to", "activity"),
    ("is learning", "learning", "topic"),
    ("learned", "learned", "topic"),
    ("attended a workshop", "attended_workshop", "event"),
    ("attended a conference", "attended_conference", "event"),
    ("opened up about", "shared_about", "topic"),
    ("shared about", "shared_about", "topic"),
    ("talked about", "discussed", "topic"),
    ("discussed", "discussed", "topic"),
    ("looking into", "researching", "topic"),
    ("working on", "working_on", "activity"),
    ("signed up", "enrolled_in", "activity"),
    ("made a", "made", "object"),
    ("bought a", "bought", "object"),
    ("bought an", "bought", "object"),
    ("received", "received", "object"),
    ("won", "won", "event"),
    ("hosted", "hosted", "event"),
    ("celebrated", "celebrated", "event"),
    ("chose an", "chose", "organization"),
    ("chose a", "chose", "organization"),
    ("collects", "collects", "object"),
    ("collected", "collected", "object"),
]

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "during",
    "is", "was", "are", "were", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "can", "shall", "this", "that", "these", "those", "it",
    "its", "her", "his", "their", "my", "our", "your", "she", "he",
    "they", "we", "you", "me", "him", "us", "them", "not", "no",
    "just", "very", "really", "so", "too", "also", "now", "then",
    "here", "there", "some", "any", "much", "many", "more", "most",
    "other", "each", "every", "all", "both", "few", "still", "already",
    "yet", "even", "only", "own", "same", "last", "next", "first",
    "new", "old", "good", "great", "big", "small", "long", "right",
    "different", "recently", "currently", "always", "never", "often",
    "because", "while", "since", "after", "before", "when", "where",
    "which", "who", "whom", "what", "how", "if", "as", "like",
}


def load_json(file_path):
    with Path(file_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_verb_and_object(text):
    text_lower = text.lower()

    for verb_phrase, relation, _ in VERB_PATTERNS:
        pos = text_lower.find(verb_phrase)
        if pos == -1:
            continue

        after_verb = text[pos + len(verb_phrase):].strip()
        words = re.findall(r"[a-zA-Z]+", after_verb)
        meaningful = [w for w in words if w.lower() not in STOP_WORDS]

        if meaningful:
            obj = "_".join(meaningful[:4]).lower()
        else:
            obj = relation

        return relation, obj

    return "has_memory", _default_object(text)


def _default_object(text):
    words = re.findall(r"[a-zA-Z]+", text)
    meaningful = [w for w in words if w.lower() not in STOP_WORDS]
    obj_words = meaningful[:5] if meaningful else ["memory"]
    return "_".join(obj_words).lower()


def classify_type(obj_id):
    event_keywords = {
        "group", "meeting", "party", "wedding", "event", "workshop",
        "conference", "race", "marathon", "parade", "festival", "celebration",
        "talk", "speech", "presentation", "seminar", "session",
    }
    location_keywords = {
        "beach", "park", "school", "hospital", "office", "store", "shop",
        "restaurant", "cafe", "library", "museum", "gym", "pool",
        "lake", "mountain", "river", "forest", "garden", "home", "house",
        "city", "town", "country", "street", "road",
    }
    activity_keywords = {
        "swimming", "hiking", "camping", "running", "cooking", "baking",
        "painting", "drawing", "writing", "reading", "singing", "dancing",
        "yoga", "meditation", "exercise", "workout", "class", "course",
        "lesson", "training", "practice", "therapy",
    }
    person_keywords = {
        "mom", "dad", "mother", "father", "sister", "brother", "daughter",
        "son", "friend", "partner", "husband", "wife", "kid", "child",
        "doctor", "teacher", "therapist", "counselor",
    }
    organization_keywords = {
        "agency", "organization", "company", "school", "university",
        "college", "institute", "center", "foundation", "charity",
        "church", "community", "club", "team",
    }

    for kw in event_keywords:
        if kw in obj_id:
            return "event"
    for kw in location_keywords:
        if kw in obj_id:
            return "location"
    for kw in activity_keywords:
        if kw in obj_id:
            return "activity"
    for kw in person_keywords:
        if kw in obj_id:
            return "person"
    for kw in organization_keywords:
        if kw in obj_id:
            return "organization"
    return "object"


def extract_kg_edges(data):
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

                    src_id = speaker_name.replace(" ", "_").lower()
                    relation, dst_id = extract_verb_and_object(fact_text)
                    src_type = "person"
                    dst_type = classify_type(dst_id)

                    edges.append({
                        "graph_id": graph_id,
                        "src_id": src_id,
                        "src_type": src_type,
                        "relation": relation,
                        "dst_id": dst_id,
                        "dst_type": dst_type,
                        "confidence": 1.0,
                        "source": "locomo_observation",
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
        description="Convert LoCoMo observations to KG edge CSV."
    )
    parser.add_argument("--file", default="external_data/locomo10.json")
    parser.add_argument("--output", default="results/locomo_kg_edges.csv")
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
    print(f"Top relations:")
    for rel, count in sorted(relation_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {rel:20s} {count}")

    unique_graphs = len(set(e["graph_id"] for e in edges))
    unique_src = len(set(e["src_id"] for e in edges))
    unique_dst = len(set(e["dst_id"] for e in edges))
    print(f"Graphs: {unique_graphs}, Source nodes: {unique_src}, Target nodes: {unique_dst}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
