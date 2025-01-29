from causy.causal_discovery.constraint.algorithms.pc import PCClassic
from causy.sample_generator import IIDSampleGenerator, SampleEdge, NodeReference
from utils_counting import count_conditionally_independent_triples, count_d_separated_triples
from tests.utils_for_tests import CausyTestCase


class CountingTestCase(CausyTestCase):

    def test_count_conditionally_independent_triples(self):
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

        nb_of_triples, triples = count_conditionally_independent_triples(tst)

        self.assertEqual(nb_of_triples, 1)

    def test_count_conditionally_independent_triples_with_restriction_to_subset_no_independent_triple_in_subset(self):
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

        nb_of_triples, triples = count_conditionally_independent_triples(tst, [tst.graph.node_by_id("X"), tst.graph.node_by_id("Y"), [tst.graph.node_by_id("Z")]])

        self.assertEqual(nb_of_triples, 0)

    def test_count_conditionally_independent_triples_with_restriction_to_subset_independent_triple_in_subset(self):
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

        nb_of_triples, triples = count_conditionally_independent_triples(tst, [[tst.graph.node_by_id("X"), tst.graph.node_by_id("Z"), [tst.graph.node_by_id("Y")]]])

        self.assertEqual(nb_of_triples, 1)

    def test_count_conditionally_independent_triples_confounder(self):
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

        nb_of_triples, triples = count_conditionally_independent_triples(tst)
        self.assertEqual(nb_of_triples, 1)


    def test_count_conditionally_independent_triples_collider(self):
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

        nb_of_triples, triples = count_conditionally_independent_triples(tst)
        self.assertEqual(nb_of_triples, 1)


    def test_count_d_separated_triples_three_nodes_mediated_path(self):
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

        nb_of_triples, triples = count_d_separated_triples(tst, [(tst.graph.node_by_id('X'), tst.graph.node_by_id('Z'), [tst.graph.node_by_id('Y')])])
        self.assertEqual(nb_of_triples, 1)

        nb_of_triples, triples = count_d_separated_triples(tst, [(tst.graph.node_by_id('X'), tst.graph.node_by_id('Z'), []), (tst.graph.node_by_id('X'), tst.graph.node_by_id('Y'), [])])
        self.assertEqual(nb_of_triples, 0)

    def test_count_d_separated_triples_three_nodes_confounder(self):
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

        nb_of_triples, triples = count_d_separated_triples(tst, [(tst.graph.node_by_id('X'), tst.graph.node_by_id('Z'), [tst.graph.node_by_id('Y')])])
        self.assertEqual(nb_of_triples, 1)

        nb_of_triples, triples = count_d_separated_triples(tst, [(tst.graph.node_by_id('X'), tst.graph.node_by_id('Z'), []), (tst.graph.node_by_id('X'), tst.graph.node_by_id('Y'), [])])
        self.assertEqual(nb_of_triples, 0)

    def test_count_d_separated_triples_three_nodes_collider(self):
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

        nb_of_triples, triples = count_d_separated_triples(tst, [(tst.graph.node_by_id('X'), tst.graph.node_by_id('Z'), [tst.graph.node_by_id('Y')])])
        self.assertEqual(nb_of_triples, 0)

        nb_of_triples, triples = count_d_separated_triples(tst, [(tst.graph.node_by_id('X'), tst.graph.node_by_id('Z'), []), (tst.graph.node_by_id('X'), tst.graph.node_by_id('Y'), [])])
        self.assertEqual(nb_of_triples, 1)
