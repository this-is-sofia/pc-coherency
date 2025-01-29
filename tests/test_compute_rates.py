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
        self.assertIn(len(_get_all_tested_triples(tst.graph.action_history)), [6,7,8])

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
        self.assertIn(len(_get_all_tested_triples(tst.graph.action_history)), [6,7,8])

