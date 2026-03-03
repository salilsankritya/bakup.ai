# core/analysis — log intelligence modules (confidence, trends, clustering, file aggregation)

from core.analysis.confidence import calculate_confidence, ConfidenceResult, SeverityDistribution
from core.analysis.trends import analyze_error_trends, TrendReport
from core.analysis.clusters import cluster_log_events, ClusterReport
from core.analysis.file_aggregation import aggregate_by_file, FileAggregationReport

__all__ = [
    "calculate_confidence",
    "ConfidenceResult",
    "SeverityDistribution",
    "analyze_error_trends",
    "TrendReport",
    "cluster_log_events",
    "ClusterReport",
    "aggregate_by_file",
    "FileAggregationReport",
]
