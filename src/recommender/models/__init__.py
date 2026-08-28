"""Recommender protocols and model implementations."""

from recommender.models.artifacts import ArtifactMetadata
from recommender.models.bpr import (
    BayesianPersonalizedRankingConfig,
    BayesianPersonalizedRankingRecommender,
    BPRConfig,
    BPRRecommender,
)
from recommender.models.matrix_factorization import (
    BiasedMatrixFactorizationRecommender,
    MatrixFactorizationConfig,
    MatrixFactorizationRecommender,
)
from recommender.models.neighborhood import (
    ItemKNNRecommender,
    NeighborhoodConfig,
    UserKNNRecommender,
)
from recommender.models.popularity import PopularityRecommender
from recommender.models.protocol import (
    Interaction,
    RecommenderProtocol,
    ValidationAwareRecommenderProtocol,
)

__all__ = [
    "ArtifactMetadata",
    "BPRConfig",
    "BPRRecommender",
    "BayesianPersonalizedRankingConfig",
    "BayesianPersonalizedRankingRecommender",
    "BiasedMatrixFactorizationRecommender",
    "Interaction",
    "ItemKNNRecommender",
    "MatrixFactorizationConfig",
    "MatrixFactorizationRecommender",
    "NeighborhoodConfig",
    "PopularityRecommender",
    "RecommenderProtocol",
    "UserKNNRecommender",
    "ValidationAwareRecommenderProtocol",
]
