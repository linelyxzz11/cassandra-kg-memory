from cassmem.representation.graph_event import GraphRecord, parse_triples


def test_parse_multiple_triples_and_embedded_commas():
    value = "(alice, meet, bob); (alice, say, hello_,_world)"
    assert parse_triples(value) == (
        ("alice", "meet", "bob"),
        ("alice", "say", "hello_,_world"),
    )


def test_graph_projection_is_deterministic():
    record = GraphRecord("s", "m", 1, "text", "bob, alice, bob", "meet", "k", "(alice, meet, bob)", "h")
    assert record.mentions == ("alice", "bob")
    assert record.graph_entities == ("alice", "bob")
    assert record.edges == ({"ordinal": 0, "src": "alice", "relation": "meet", "dst": "bob"},)
    assert record.expected_mutations("cassandra-base") == 9
