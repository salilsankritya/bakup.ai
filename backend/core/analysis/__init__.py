# core/analysis — log intelligence modules (confidence, trends, clustering, file aggregation,
#                  error pattern clustering, trend detection, causal confidence, evidence ranking)

from core.analysis.confidence import calculate_confidence, ConfidenceResult, SeverityDistribution
from core.analysis.trends import analyze_error_trends, TrendReport
from core.analysis.clusters import cluster_log_events, ClusterReport
from core.analysis.file_aggregation import aggregate_by_file, FileAggregationReport
from core.analysis.error_clustering import cluster_error_patterns, ErrorCluster, ErrorClusterReport
from core.analysis.trend_detector import detect_trends, ClusterTrend, TrendDetectionReport
from core.analysis.causal_confidence import compute_causal_confidence, CausalConfidenceResult
from core.analysis.evidence_ranker import rank_evidence, StructuredReasoningInput

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
    "cluster_error_patterns",
    "ErrorCluster",
    "ErrorClusterReport",
    "detect_trends",
    "ClusterTrend",
    "TrendDetectionReport",
    "compute_causal_confidence",
    "CausalConfidenceResult",
    "rank_evidence",
    "StructuredReasoningInput",
]
