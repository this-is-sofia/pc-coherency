from causy.causal_discovery.constraint.algorithms.pc import PC_ORIENTATION_RULES, PC_EDGE_TYPES, PC_GRAPH_UI_EXTENSION, \
    PC_DEFAULT_THRESHOLD
from causy.causal_discovery.constraint.independence_tests.common import CorrelationCoefficientTest, \
    ExtendedPartialCorrelationTestMatrix
from causy.causal_effect_estimation.multivariate_regression import ComputeDirectEffectsInDAGsMultivariateRegression
from causy.common_pipeline_steps.calculation import CalculatePearsonCorrelations
from causy.generators import PairsWithNeighboursGenerator
from causy.graph_model import graph_model_factory
from causy.interfaces import AS_MANY_AS_FIELDS
from causy.models import Algorithm, ComparisonSettings
from causy.variables import VariableReference, FloatVariable

PCCustom = graph_model_factory(
    Algorithm(
        pipeline_steps=[
            CalculatePearsonCorrelations(display_name="Calculate Pearson Correlations"),
            #CustomIndependenceTest(
            #    threshold=VariableReference(name="threshold"),
            #    display_name="Extended Partial Correlation Test Matrix",
            #    generator=PairsWithNeighboursGenerator(
            #        comparison_settings=ComparisonSettings(
            #            min=3, max=AS_MANY_AS_FIELDS
            #        ),
            #        shuffle_combinations=False,
            #    ),
            #),
            *PC_ORIENTATION_RULES,
            ComputeDirectEffectsInDAGsMultivariateRegression(
                display_name="Compute Direct Effects in DAGs Multivariate Regression"
            ),
        ],
        edge_types=PC_EDGE_TYPES,
        extensions=[PC_GRAPH_UI_EXTENSION],
        name="PC",
        variables=[FloatVariable(name="threshold", value=PC_DEFAULT_THRESHOLD)],
    )
)