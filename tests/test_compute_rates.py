from causy.causal_discovery.constraint.algorithms.pc import PCClassic
from causy.sample_generator import IIDSampleGenerator, SampleEdge, NodeReference
from compute_rates import _get_all_tested_triples, compute_faithfulness_coherency_rate, compute_markov_coherency_rate, \
    compute_total_coherency_rate
from tests.utils_for_tests import CausyTestCase

class ConherencyTestCase(CausyTestCase):

    def test_compute_faithfulness_coherency_rate(self):
        rdnv = self.seeded_random.normalvariate
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 6),
            ],
            random=lambda: rdnv(0, 1),
        )
        sample_size = 1000
        test_data, graph = model.generate(sample_size)

        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

        rate = compute_faithfulness_coherency_rate(tst)

        self.assertEqual(rate, 1.0)

    def test_compute_faithfulness_coherency_rate_four_nodes(self):
        rdnv = self.seeded_random.normalvariate
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Z"), NodeReference("Y"), 6),
                SampleEdge(NodeReference("Y"), NodeReference("W"), 3),
            ],
            random=lambda: rdnv(0, 1),
        )
        sample_size = 100000
        test_data, graph = model.generate(sample_size)

        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

        rate = compute_faithfulness_coherency_rate(tst)

        self.assertEqual(rate, 1.0)

    def test_faithfulness_coherency_rate_partially_directed_output(self):
        rdnv = self.seeded_random.normalvariate
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Z"), NodeReference("Y"), 6),
                SampleEdge(NodeReference("X"), NodeReference("Z"), 3),
            ],
            random=lambda: rdnv(0, 1),
        )
        sample_size = 1000
        test_data, graph = model.generate(sample_size)

        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

        rate = compute_faithfulness_coherency_rate(tst)

        self.assertEqual(rate, 1.0)

    def test_markov_coherency_rate_mediator(self):
        rdnv = self.seeded_random.normalvariate
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 6),
            ],
            random=lambda: rdnv(0, 1),
        )
        sample_size = 1000
        test_data, graph = model.generate(sample_size)

        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

        rate = compute_markov_coherency_rate(tst)

        self.assertEqual(rate, 1.0)

    def test_markov_coherency_rate_collider(self):
        rdnv = self.seeded_random.normalvariate
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Z"), NodeReference("Y"), 6),
            ],
            random=lambda: rdnv(0, 1),
        )
        sample_size = 1000
        test_data, graph = model.generate(sample_size)

        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

        rate = compute_markov_coherency_rate(tst)

        self.assertEqual(rate, 1.0)


    def test_markov_coherency_rate_confounder(self):
        rdnv = self.seeded_random.normalvariate
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("Y"), NodeReference("X"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 6),
            ],
            random=lambda: rdnv(0, 1),
        )
        sample_size = 1000
        test_data, graph = model.generate(sample_size)

        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

        rate = compute_markov_coherency_rate(tst)

        self.assertEqual(rate, 1.0)

    def test_markov_coherency_rate_partially_directed_output(self):
        rdnv = self.seeded_random.normalvariate
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Z"), NodeReference("Y"), 6),
                SampleEdge(NodeReference("X"), NodeReference("Z"), 3),
            ],
            random=lambda: rdnv(0, 1),
        )
        sample_size = 1000
        test_data, graph = model.generate(sample_size)

        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

        rate = compute_markov_coherency_rate(tst)

        self.assertEqual(rate, 1.0)

    def test_total_coherency_rate_mediator(self):
        rdnv = self.seeded_random.normalvariate
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 6),
            ],
            random=lambda: rdnv(0, 1),
        )
        sample_size = 1000
        test_data, graph = model.generate(sample_size)

        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

        rate = compute_total_coherency_rate(tst)

        self.assertEqual(rate, 1.0)

    def test_total_coherency_rate_collider(self):
        rdnv = self.seeded_random.normalvariate
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Z"), NodeReference("Y"), 6),
            ],
            random=lambda: rdnv(0, 1),
        )
        sample_size = 1000
        test_data, graph = model.generate(sample_size)

        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

        rate = compute_total_coherency_rate(tst)

        self.assertEqual(rate, 1.0)

    def test_total_coherency_rate_confounder(self):
        rdnv = self.seeded_random.normalvariate
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("Y"), NodeReference("X"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 6),
            ],
            random=lambda: rdnv(0, 1),
        )
        sample_size = 1000
        test_data, graph = model.generate(sample_size)

        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

        rate = compute_total_coherency_rate(tst)

        self.assertEqual(rate, 1.0)

    def test_total_coherency_rate_partially_directed_output(self):
        rdnv = self.seeded_random.normalvariate
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Z"), NodeReference("Y"), 6),
                SampleEdge(NodeReference("X"), NodeReference("Z"), 3),
            ],
            random=lambda: rdnv(0, 1),
        )
        sample_size = 1000
        test_data, graph = model.generate(sample_size)

        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

        rate = compute_total_coherency_rate(tst)

        self.assertEqual(rate, 1.0)

    def test_get_all_tested_triples_collider(self):
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Z"), NodeReference("Y"), 6),
            ],
            random=lambda: rdnv(0, 1),
        )
        test_data, graph = sample_generator.generate(10000)
        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

        self.assertEqual(len(_get_all_tested_triples(tst.graph.action_history)), 5)

    def test_get_all_tested_triples_mediator(self):
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 6),
            ],
            random=lambda: rdnv(0, 1),
        )
        test_data, graph = sample_generator.generate(10000)
        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)
        self.assertEqual(len(_get_all_tested_triples(tst.graph.action_history)), 6)

    def test_get_all_tested_triples_confounder(self):
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("Y"), NodeReference("X"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 6),
            ],
            random=lambda: rdnv(0, 1),
        )
        test_data, graph = sample_generator.generate(10000)
        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)
        self.assertEqual(len(_get_all_tested_triples(tst.graph.action_history)), 6)

    def test_get_all_tested_triples_three_nodes_fully_connected(self):
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Z"), NodeReference("Y"), 6),
                SampleEdge(NodeReference("X"), NodeReference("Z"), 3),
            ],
            random=lambda: rdnv(0, 1),
        )
        test_data, graph = sample_generator.generate(10000)
        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)
        self.assertEqual(len(_get_all_tested_triples(tst.graph.action_history)), 6)

    def test_get_all_tested_triples_faithfulness_violation(self):
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("V"), 2),
                SampleEdge(NodeReference("V"), NodeReference("W"), 2),
                SampleEdge(NodeReference("W"), NodeReference("Y"), -2),
                SampleEdge(NodeReference("X"), NodeReference("Y"), 8),
            ],
            random=lambda: rdnv(0, 1),
        )
        test_data, graph = sample_generator.generate(10000)
        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        # total number of tests depends on the order, ranges from 18 to 20
        self.assertIn(len(_get_all_tested_triples(tst.graph.action_history)), [18,19,20])

    def test_faithfulness_score_for_faithfulness_violation(self):
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("V"), 2),
                SampleEdge(NodeReference("V"), NodeReference("W"), 2),
                SampleEdge(NodeReference("W"), NodeReference("Y"), -2),
                SampleEdge(NodeReference("X"), NodeReference("Y"), 8),
            ],
            random=lambda: rdnv(0, 1),
        )
        test_data, graph = sample_generator.generate(10000)
        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        rate = compute_faithfulness_coherency_rate(tst)

        self.assertLess(rate, 1.0)
        self.assertAlmostEqual(rate, 0.95, places=1)

    def test_markov_score_for_faithfulness_violation(self):
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("V"), 2),
                SampleEdge(NodeReference("V"), NodeReference("W"), 2),
                SampleEdge(NodeReference("W"), NodeReference("Y"), -2),
                SampleEdge(NodeReference("X"), NodeReference("Y"), 8),
            ],
            random=lambda: rdnv(0, 1),
        )
        test_data, graph = sample_generator.generate(10000)
        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        rate = compute_markov_coherency_rate(tst)

        self.assertLess(rate, 1.0)
        self.assertAlmostEqual(rate, 0.95, places=1)
        self.assertIn(round(100*rate), [93, 94, 95, 96, 97])

    def test_total_score_for_faithfulness_violation(self):
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("V"), 2),
                SampleEdge(NodeReference("V"), NodeReference("W"), 2),
                SampleEdge(NodeReference("W"), NodeReference("Y"), -2),
                SampleEdge(NodeReference("X"), NodeReference("Y"), 8),
            ],
            random=lambda: rdnv(0, 1),
        )
        test_data, graph = sample_generator.generate(10000)
        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        rate = compute_total_coherency_rate(tst)
        m_rate = compute_markov_coherency_rate(tst)
        f_rate = compute_faithfulness_coherency_rate(tst)



        self.assertLess(rate, 1.0)
        self.assertAlmostEqual(rate, 0.9, places=1)
        self.assertIn(round(100*rate), [88, 89, 90, 91, 92])

        self.assertAlmostEqual(1-rate, ((1-f_rate)+(1-m_rate)), places=3)

    def test_scores_for_faithfulness_violation_undetectable(self):
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("V"), 2),
                SampleEdge(NodeReference("V"), NodeReference("W"), -2),
                SampleEdge(NodeReference("X"), NodeReference("W"), 4),
            ],
            random=lambda: rdnv(0, 1),
        )
        test_data, graph = sample_generator.generate(10000)
        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        rate = compute_total_coherency_rate(tst)
        m_rate = compute_markov_coherency_rate(tst)
        f_rate = compute_faithfulness_coherency_rate(tst)

        self.assertEqual(rate, 1.0)
        self.assertEqual(m_rate, 1.0)
        self.assertEqual(f_rate, 1.0)

    def test_track_triples_insufficiency(self):
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

        test_data, graph = causal_insufficiency_four_nodes.generate(10000)
        test_data.pop("U1")
        test_data.pop("U2")
        test_data.pop("U3")
        test_data.pop("U4")
        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertEqual(len(_get_all_tested_triples(tst.graph.action_history)), 14)

