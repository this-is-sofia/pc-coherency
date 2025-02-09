from causy.causal_discovery.constraint.algorithms.pc import PCClassic
from causy.sample_generator import IIDSampleGenerator, SampleEdge, NodeReference

from models import causal_insufficiency_four_nodes
from tests.utils_for_tests import CausyTestCase
from utils import _get_all_tested_triples, _get_number_of_orientation_conflicts


class UtilsTestCase(CausyTestCase):
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
        self.assertIn(len(_get_all_tested_triples(tst.graph.action_history)), [18, 19, 20])

    def test_number_of_orientation_conflicts_no_conflicts(self):
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

        self.assertEqual(_get_number_of_orientation_conflicts(tst.graph.action_history), 0)

    def test_number_of_orientation_conflicts_two_conflicts(self):
        test_data, graph = causal_insufficiency_four_nodes.generate(10000)
        test_data.pop("U1")
        test_data.pop("U2")
        test_data.pop("U3")
        test_data.pop("U4")
        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertEqual(_get_number_of_orientation_conflicts(tst.graph.action_history), 4)
