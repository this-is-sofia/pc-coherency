from causy.causal_discovery.constraint.algorithms.pc import PCClassic
from causy.sample_generator import IIDSampleGenerator, SampleEdge, NodeReference
from compute_scores import compute_faithfulness_coherency_score, \
    compute_markov_coherency_score, \
    compute_total_coherency_score, weight_by_exponential_decay_of_cardinality_of_conditioning_set
from utils import _get_all_tested_triples
from counting import count_conditionally_independent_triples
from models import mediated_path_one_as_effect, mediated_path_very_small_effects
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

        rate = compute_faithfulness_coherency_score(tst)

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

        rate = compute_faithfulness_coherency_score(tst)

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

        rate = compute_faithfulness_coherency_score(tst)

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

        rate = compute_markov_coherency_score(tst)

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

        rate = compute_markov_coherency_score(tst)

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

        rate = compute_markov_coherency_score(tst)

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

        rate = compute_markov_coherency_score(tst)

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

        rate = compute_total_coherency_score(tst)

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

        rate = compute_total_coherency_score(tst)

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

        rate = compute_total_coherency_score(tst)

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

        rate = compute_total_coherency_score(tst)

        self.assertEqual(rate, 1.0)

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

        rate = compute_faithfulness_coherency_score(tst)

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

        rate = compute_markov_coherency_score(tst)

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

        rate = compute_total_coherency_score(tst)
        m_rate = compute_markov_coherency_score(tst)
        f_rate = compute_faithfulness_coherency_score(tst)



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

        rate = compute_total_coherency_score(tst)
        m_rate = compute_markov_coherency_score(tst)
        f_rate = compute_faithfulness_coherency_score(tst)

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

    def test_weighted_score_faithfulness_violation(self):
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("V"), 2),
                SampleEdge(NodeReference("V"), NodeReference("Q"), -2),
                SampleEdge(NodeReference("Q"), NodeReference("W"), 2),
                SampleEdge(NodeReference("X"), NodeReference("B"), 2),
                SampleEdge(NodeReference("B"), NodeReference("Z"), -2),
                SampleEdge(NodeReference("Z"), NodeReference("W"), 2),
                SampleEdge(NodeReference("X"), NodeReference("W"), 16),
            ],
            random=lambda: rdnv(0, 1),
        )
        test_data, graph = sample_generator.generate(10000)
        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        rate = compute_total_coherency_score(tst)
        print(tst)
        weighted_score = compute_total_coherency_score(tst, weight_by_exponential_decay_of_cardinality_of_conditioning_set)
        print(weighted_score, rate)
        self.assertLess(weighted_score, rate)

    def test_weighted_scores_mediated_path(self):
        sample_generator = mediated_path_one_as_effect
        test_data, graph = sample_generator.generate(10000)
        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

        results = tst.graph.action_history

        for action_history_step in results:
            for action in action_history_step.actions:
                if "separatedBy" in action.data:
                    triple = action.data["triple"]
                    print(triple[0].name, triple[1].name, [node.name for node in triple[2]])

        faithfulness_rate = compute_faithfulness_coherency_score(tst)
        markov_rate = compute_markov_coherency_score(tst)

        weighted_faithfulness_rate = compute_faithfulness_coherency_score(tst, weight_by_exponential_decay_of_cardinality_of_conditioning_set)
        weighted_markov_rate = compute_markov_coherency_score(tst, weight_by_exponential_decay_of_cardinality_of_conditioning_set)

        print(faithfulness_rate, markov_rate)
        print(weighted_faithfulness_rate, weighted_markov_rate)

    def test_very_small_effects_graph(self):
        sample_generator = mediated_path_very_small_effects
        test_data, graph = sample_generator.generate(1000)
        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        #self.assertGraphStructureIsEqual(tst.graph, graph)

        results = tst.graph.action_history

        for action_history_step in results:
            for action in action_history_step.actions:
                if "separatedBy" in action.data:
                    triple = action.data["triple"]
                    print(triple[0].name, triple[1].name, [node.name for node in triple[2]])

        faithfulness_rate = compute_faithfulness_coherency_score(tst)
        markov_rate = compute_markov_coherency_score(tst)

        weighted_faithfulness_rate = compute_faithfulness_coherency_score(tst, weight_by_exponential_decay_of_cardinality_of_conditioning_set)
        weighted_markov_rate = compute_markov_coherency_score(tst, weight_by_exponential_decay_of_cardinality_of_conditioning_set)

        print(faithfulness_rate, markov_rate)
        print(weighted_faithfulness_rate, weighted_markov_rate)
