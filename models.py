from causy.sample_generator import IIDSampleGenerator, SampleEdge, NodeReference

# Example five nodes and X -> Z <-Y as well as V <- Z -> W, different effect sizes
classical_five_node_example_small_effect_size = IIDSampleGenerator(
    edges=[
        SampleEdge(NodeReference("X"), NodeReference("Z"), 1),
        SampleEdge(NodeReference("Y"), NodeReference("Z"), 1),
        SampleEdge(NodeReference("Z"), NodeReference("W"), 1),
        SampleEdge(NodeReference("Z"), NodeReference("V"), 1),
    ],
)

classical_five_node_example_medium_effect_size = IIDSampleGenerator(
    edges=[
        SampleEdge(NodeReference("X"), NodeReference("Z"), 5),
        SampleEdge(NodeReference("Y"), NodeReference("Z"), 5),
        SampleEdge(NodeReference("Z"), NodeReference("W"), 5),
        SampleEdge(NodeReference("Z"), NodeReference("V"), 5),
    ],
)

classical_five_node_example_large_effect_size = IIDSampleGenerator(
    edges=[
        SampleEdge(NodeReference("X"), NodeReference("Z"), 10),
        SampleEdge(NodeReference("Y"), NodeReference("Z"), 10),
        SampleEdge(NodeReference("Z"), NodeReference("W"), 10),
        SampleEdge(NodeReference("Z"), NodeReference("V"), 10),
    ],
)

faithfulness_violation_three_nodes = IIDSampleGenerator(
    edges=[
        SampleEdge(NodeReference("X"), NodeReference("Y"), 1),
        SampleEdge(NodeReference("Y"), NodeReference("Z"), -1),
        SampleEdge(NodeReference("X"), NodeReference("Z"), 1),
    ],
)

# Example for faithfulness violation with four nodes X -> V -> W -> Y, X-> Y
faithfulness_violation_four_nodes = IIDSampleGenerator(
    edges=[
        SampleEdge(NodeReference("X"), NodeReference("V"), 1),
        SampleEdge(NodeReference("V"), NodeReference("W"), -1),
        SampleEdge(NodeReference("W"), NodeReference("Y"), 1),
        SampleEdge(NodeReference("X"), NodeReference("Y"), 1),
    ],
)

faithfulness_violation_five_nodes = IIDSampleGenerator(
    edges=[
        SampleEdge(NodeReference("X"), NodeReference("V"), 1),
        SampleEdge(NodeReference("V"), NodeReference("W"), 1),
        SampleEdge(NodeReference("W"), NodeReference("U"), -1),
        SampleEdge(NodeReference("U"), NodeReference("Y"), 1),
        SampleEdge(NodeReference("X"), NodeReference("Y"), 1),
    ],
)

faithfulness_violation_six_nodes = IIDSampleGenerator(
    edges=[
        SampleEdge(NodeReference("X"), NodeReference("V"), 1),
        SampleEdge(NodeReference("V"), NodeReference("W"), 1),
        SampleEdge(NodeReference("W"), NodeReference("U"), 1),
        SampleEdge(NodeReference("U"), NodeReference("Z"), -1),
        SampleEdge(NodeReference("Z"), NodeReference("Y"), 1),
        SampleEdge(NodeReference("X"), NodeReference("Y"), 1),
    ],
)

causal_insufficiency_three_nodes = IIDSampleGenerator(
    edges=[
        SampleEdge(NodeReference("U1"), NodeReference("X"), 1),
        SampleEdge(NodeReference("U1"), NodeReference("Y"), 1),
        SampleEdge(NodeReference("U2"), NodeReference("Y"), 1),
        SampleEdge(NodeReference("U2"), NodeReference("Z"), 1),
        SampleEdge(NodeReference("U3"), NodeReference("Z"), 1),
        SampleEdge(NodeReference("U3"), NodeReference("X"), 1),
    ],
)

causal_insufficiency_four_nodes = IIDSampleGenerator(
    edges=[
        SampleEdge(NodeReference("U1"), NodeReference("X"), 1),
        SampleEdge(NodeReference("U1"), NodeReference("Y"), 1),
        SampleEdge(NodeReference("U2"), NodeReference("Y"), 1),
        SampleEdge(NodeReference("U2"), NodeReference("Z"), 1),
        SampleEdge(NodeReference("U3"), NodeReference("Z"), 1),
        SampleEdge(NodeReference("U3"), NodeReference("V"), 1),
        SampleEdge(NodeReference("U4"), NodeReference("V"), 1),
        SampleEdge(NodeReference("U4"), NodeReference("X"), 1),
    ],
)

mediated_path_very_small_effects = IIDSampleGenerator(
    edges=[
        SampleEdge(NodeReference("X"), NodeReference("Y"), 0.2),
        SampleEdge(NodeReference("Y"), NodeReference("Z"), 0.2),
        SampleEdge(NodeReference("Z"), NodeReference("W"), 0.2),
        SampleEdge(NodeReference("W"), NodeReference("V"), 0.2),
        SampleEdge(NodeReference("V"), NodeReference("U"), 0.2),
        SampleEdge(NodeReference("U"), NodeReference("T"), 0.2),
    ],
)

mediated_path_small_effects = IIDSampleGenerator(
    edges=[
        SampleEdge(NodeReference("X"), NodeReference("Y"), 0.5),
        SampleEdge(NodeReference("Y"), NodeReference("Z"), 0.5),
        SampleEdge(NodeReference("Z"), NodeReference("W"), 0.5),
        SampleEdge(NodeReference("W"), NodeReference("V"), 0.5),
        SampleEdge(NodeReference("V"), NodeReference("U"), 0.5),
        SampleEdge(NodeReference("U"), NodeReference("T"), 0.5),
    ],
)

mediated_path_almost_one_as_effect = IIDSampleGenerator(
    edges=[
        SampleEdge(NodeReference("X"), NodeReference("Y"), 0.9),
        SampleEdge(NodeReference("Y"), NodeReference("Z"), 0.9),
        SampleEdge(NodeReference("Z"), NodeReference("W"), 0.9),
        SampleEdge(NodeReference("W"), NodeReference("V"), 0.9),
        SampleEdge(NodeReference("V"), NodeReference("U"), 0.9),
        SampleEdge(NodeReference("U"), NodeReference("T"), 0.9),
    ],
)

mediated_path_one_as_effect = IIDSampleGenerator(
    edges=[
        SampleEdge(NodeReference("X"), NodeReference("Y"), 1.0),
        SampleEdge(NodeReference("Y"), NodeReference("Z"), 1.0),
        SampleEdge(NodeReference("Z"), NodeReference("W"), 1.0),
        SampleEdge(NodeReference("W"), NodeReference("V"), 1.0),
        SampleEdge(NodeReference("V"), NodeReference("U"), 1.0),
        SampleEdge(NodeReference("U"), NodeReference("T"), 1.0),
    ],
)

mediated_path_five_as_effect = IIDSampleGenerator(
    edges=[
        SampleEdge(NodeReference("X"), NodeReference("Y"), 5.0),
        SampleEdge(NodeReference("Y"), NodeReference("Z"), 5.0),
        SampleEdge(NodeReference("Z"), NodeReference("W"), 5.0),
        SampleEdge(NodeReference("W"), NodeReference("V"), 5.0),
        SampleEdge(NodeReference("V"), NodeReference("U"), 5.0),
        SampleEdge(NodeReference("U"), NodeReference("T"), 5.0),
    ],
)

mediated_path_ten_as_effect = IIDSampleGenerator(
    edges=[
        SampleEdge(NodeReference("X"), NodeReference("Y"), 10.0),
        SampleEdge(NodeReference("Y"), NodeReference("Z"), 10.0),
        SampleEdge(NodeReference("Z"), NodeReference("W"), 10.0),
        SampleEdge(NodeReference("W"), NodeReference("V"), 10.0),
        SampleEdge(NodeReference("V"), NodeReference("U"), 10.0),
        SampleEdge(NodeReference("U"), NodeReference("T"), 10.0),
    ],
)






