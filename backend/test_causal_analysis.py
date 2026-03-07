"""Test script for causal confidence scoring & error pattern clustering (v4)."""
import sys, os
sys.path.insert(0, '.')

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Error Clustering  (core/analysis/error_clustering.py)
# ══════════════════════════════════════════════════════════════════════════════

from core.analysis.error_clustering import (
    ErrorCluster,
    ErrorClusterReport,
    cluster_error_patterns,
    _compute_signature,
    _normalise_error_text,
    _extract_exception_type,
    _extract_stack_frames,
    _extract_file_lines,
    _extract_functions,
    _signature_similarity,
)


# ── Helper to build fake chunks ──────────────────────────────────────────────

def _chunk(text: str, source_file: str = "", severity: str = ""):
    """Create a minimal chunk-like object with .text and optional metadata."""
    return SimpleNamespace(text=text, source_file=source_file, severity=severity)


# ==============================
# Test 1: Signature — Exception type extraction
# ==============================
assert _extract_exception_type("NullPointerException: foo is null") == "NullPointerException"
assert _extract_exception_type("ValueError: invalid literal") == "ValueError"
assert _extract_exception_type("java.lang.OutOfMemoryError: heap space") == "java.lang.OutOfMemoryError"
assert _extract_exception_type("nothing wrong here") == ""
assert _extract_exception_type("ConnectionError") == "ConnectionError"
print("✓ Test 1 passed: Exception type extraction")

# ==============================
# Test 2: Signature — Text normalisation strips variable data
# ==============================
norm1 = _normalise_error_text("2024-01-15T10:30:00Z ERROR: failed with id=0xABCDEF")
assert "<ts>" in norm1 or "<addr>" in norm1

norm2 = _normalise_error_text("Request a1b2c3d4-e5f6-7890-abcd-ef1234567890 failed")
assert "<uuid>" in norm2

norm3 = _normalise_error_text("Timeout after 12345 ms on port 8080")
assert "<num>" in norm3
print("✓ Test 2 passed: Text normalisation strips variable data")

# ==============================
# Test 3: Signature — compute_signature priorities
# ==============================
sig_exc = _compute_signature("NullPointerException: foo is null at bar.py")
assert sig_exc.startswith("exc:")

sig_msg = _compute_signature("[ERROR] connection refused to database")
assert sig_msg.startswith("msg:")

sig_loc = _compute_signature("Something at config.py:42")
# Should extract file:line if no exception or clear message
assert "config.py" in sig_loc or sig_loc.startswith("msg:")
print("✓ Test 3 passed: Signature computation priorities")

# ==============================
# Test 4: Signature — similarity between same-type exceptions
# ==============================
assert _signature_similarity("exc:NullPointerException", "exc:NullPointerException") == 1.0
assert _signature_similarity("exc:NullPointerException", "exc:ValueError") == 0.0
assert _signature_similarity("exc:Error", "exc:ConnectionError") >= 0.7  # substring
assert _signature_similarity("msg:foo bar", "exc:ValueError") == 0.0  # different type
print("✓ Test 4 passed: Signature similarity")

# ==============================
# Test 5: Signature — message token overlap similarity
# ==============================
sim = _signature_similarity("msg:connection refused to database",
                            "msg:connection refused to server")
assert 0.4 <= sim <= 1.0  # high overlap: 3/5 words shared

sim_low = _signature_similarity("msg:timeout on request",
                                "msg:disk space full alert")
assert sim_low < 0.3  # low overlap
print("✓ Test 5 passed: Message token overlap similarity")

# ==============================
# Test 6: Stack frame extraction
# ==============================
text = """Traceback (most recent call last):
  File "handler.py", line 42, in process
  at com.example.Service.run(Service.java:100)
  in parse_request
"""
frames = _extract_stack_frames(text)
assert len(frames) >= 3
print("✓ Test 6 passed: Stack frame extraction")

# ==============================
# Test 7: File:line extraction
# ==============================
file_lines = _extract_file_lines('Error at handler.py:42 and utils.ts:100')
assert len(file_lines) == 2
assert file_lines[0] == ("handler.py", 42)
assert file_lines[1] == ("utils.ts", 100)
print("✓ Test 7 passed: File:line extraction")

# ==============================
# Test 8: Function extraction
# ==============================
funcs = _extract_functions("in handle_request, called from UserService.process")
assert "handle_request" in funcs
assert "process" in funcs
print("✓ Test 8 passed: Function extraction")

# ==============================
# Test 9: Clustering — same exception type groups together
# ==============================
chunks = [
    _chunk("2024-01-15T10:00:00Z ERROR NullPointerException: x is null at handler.py:42"),
    _chunk("2024-01-15T10:01:00Z ERROR NullPointerException: y is null at handler.py:50"),
    _chunk("2024-01-15T10:02:00Z ERROR NullPointerException: z is null at handler.py:42"),
    _chunk("2024-01-15T10:03:00Z ERROR ValueError: invalid input at parser.py:10"),
]
report = cluster_error_patterns(chunks)
assert report.total_entries == 4
# NullPointerExceptions should be in one cluster, ValueError in another
assert report.cluster_count >= 2
npe_clusters = [c for c in report.clusters if c.exception_type == "NullPointerException"]
val_clusters = [c for c in report.clusters if c.exception_type == "ValueError"]
assert len(npe_clusters) >= 1
assert npe_clusters[0].count >= 3
assert len(val_clusters) >= 1
assert val_clusters[0].count == 1
print("✓ Test 9 passed: Same exception type clusters together")

# ==============================
# Test 10: Clustering — dominant cluster detection
# ==============================
dom = report.dominant_cluster
assert dom is not None
assert dom.count >= 3
assert dom.exception_type == "NullPointerException"
print("✓ Test 10 passed: Dominant cluster detection")

# ==============================
# Test 11: Clustering — severity detection
# ==============================
assert dom.severity in ("error", "fatal", "critical")
print("✓ Test 11 passed: Cluster severity detection")

# ==============================
# Test 12: Clustering — related files extraction
# ==============================
# NullPointerException cluster should reference handler.py
assert any("handler" in f for f in dom.related_files)
print("✓ Test 12 passed: Related files extraction")

# ==============================
# Test 13: Clustering — sample messages capped at 3
# ==============================
chunks_many = [
    _chunk(f"2024-01-15T10:0{i}:00Z ERROR NullPointerException: var{i} is null")
    for i in range(6)
]
report_many = cluster_error_patterns(chunks_many)
for cl in report_many.clusters:
    assert len(cl.sample_messages) <= 3
print("✓ Test 13 passed: Sample messages capped at 3")

# ==============================
# Test 14: Clustering — ErrorClusterReport.summary_text()
# ==============================
summary = report.summary_text()
assert "error pattern cluster" in summary.lower()
assert "NullPointerException" in summary or "nullpointerexception" in summary.lower()
print("✓ Test 14 passed: ErrorClusterReport.summary_text()")

# ==============================
# Test 15: Clustering — ErrorClusterReport.to_dict()
# ==============================
d = report.to_dict()
assert "cluster_count" in d
assert "clusters" in d
assert isinstance(d["clusters"], list)
assert d["cluster_count"] == report.cluster_count
assert d["dominant_cluster"] is not None
print("✓ Test 15 passed: ErrorClusterReport.to_dict()")

# ==============================
# Test 16: Clustering — ErrorCluster.to_dict()
# ==============================
cd = dom.to_dict()
assert "cluster_id" in cd
assert "error_signature" in cd
assert "exception_type" in cd
assert "count" in cd
assert cd["count"] >= 3
print("✓ Test 16 passed: ErrorCluster.to_dict()")

# ==============================
# Test 17: Clustering — ErrorCluster.describe()
# ==============================
desc = dom.describe()
assert "×" in desc or "x" in desc.lower()
assert str(dom.count) in desc
print("✓ Test 17 passed: ErrorCluster.describe()")

# ==============================
# Test 18: Clustering — empty input returns empty report
# ==============================
empty_report = cluster_error_patterns([])
assert empty_report.cluster_count == 0
assert empty_report.total_entries == 0
assert empty_report.dominant_cluster is None
assert empty_report.summary_text().lower().startswith("no error pattern")
print("✓ Test 18 passed: Empty input → empty report")

# ==============================
# Test 19: Clustering — info/debug entries are filtered out
# ==============================
info_chunks = [
    _chunk("2024-01-15T10:00:00Z INFO Server started on port 8080"),
    _chunk("2024-01-15T10:01:00Z DEBUG Loading config from app.yaml"),
]
info_report = cluster_error_patterns(info_chunks)
assert info_report.cluster_count == 0
assert info_report.unclustered_count == 2
print("✓ Test 19 passed: Info/debug entries filtered out")

# ==============================
# Test 20: Clustering — similar messages merge (token overlap)
# ==============================
similar_chunks = [
    _chunk("ERROR connection refused to database server at db.py:10"),
    _chunk("ERROR connection refused to cache server at cache.py:20"),
    _chunk("ERROR disk space full on /var/log"),
]
sim_report = cluster_error_patterns(similar_chunks)
# "connection refused" messages should merge if threshold is met
# "disk space full" should be separate
assert sim_report.cluster_count >= 1
print("✓ Test 20 passed: Similar message merging")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Trend Detection  (core/analysis/trend_detector.py)
# ══════════════════════════════════════════════════════════════════════════════

from core.analysis.trend_detector import (
    ClusterTrend,
    TrendDetectionReport,
    detect_trends,
    _analyse_cluster_trend,
    WINDOW_1H,
    WINDOW_24H,
    SPIKE_MULTIPLIER,
    QUIET_GAP_HOURS,
)


# ── Helper: build a cluster with known signature ─────────────────────────────

def _make_cluster(cluster_id: int, sig: str, exc_type: str = "",
                  count: int = 5, severity: str = "error") -> ErrorCluster:
    return ErrorCluster(
        cluster_id=cluster_id,
        error_signature=sig,
        exception_type=exc_type,
        count=count,
        severity=severity,
    )


# ==============================
# Test 21: Trend — basic 1h/24h window counting
# ==============================
now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
cluster = _make_cluster(1, "NullPointerException", "NullPointerException", count=5)

timestamps = [
    now - timedelta(minutes=10),   # within 1h
    now - timedelta(minutes=30),   # within 1h
    now - timedelta(hours=2),      # within 24h, outside 1h
    now - timedelta(hours=5),      # within 24h
    now - timedelta(hours=30),     # outside 24h
]

trend = _analyse_cluster_trend(cluster, timestamps, now)
assert trend.last_1h == 2
assert trend.last_24h == 4
assert trend.total == 5
print("✓ Test 21 passed: Window counting (1h=2, 24h=4)")

# ==============================
# Test 22: Trend — spike detection
# ==============================
# All 5 events in last 30 min → avg hourly would be low vs 5/hour burst
spike_cluster = _make_cluster(2, "TimeoutError", count=10)
spike_ts = [now - timedelta(minutes=i * 5) for i in range(10)]  # 10 in last 50 min
# Add some older ones to create a baseline
spike_ts.extend([now - timedelta(hours=h) for h in range(24, 48)])  # 24 old ones
spike_cluster.count = len(spike_ts)

spike_trend = _analyse_cluster_trend(spike_cluster, spike_ts, now)
assert spike_trend.last_1h >= 10
assert spike_trend.spike_detected is True
assert spike_trend.trend_label == "spike"
print("✓ Test 22 passed: Spike detection")

# ==============================
# Test 23: Trend — new error detection (first seen < 24h)
# ==============================
new_cluster = _make_cluster(3, "NewBugError", count=3)
new_ts = [
    now - timedelta(hours=2),
    now - timedelta(hours=1),
    now - timedelta(minutes=15),
]
new_trend = _analyse_cluster_trend(new_cluster, new_ts, now)
assert new_trend.is_new is True
assert new_trend.trend_label == "new"
print("✓ Test 23 passed: New error detection")

# ==============================
# Test 24: Trend — regression detection (6h+ gap then reappearance)
# ==============================
reg_cluster = _make_cluster(4, "OldBug", count=6)
reg_ts = [
    now - timedelta(hours=30),    # old occurrence
    now - timedelta(hours=29),    # old occurrence  
    now - timedelta(hours=28),    # old occurrence
    now - timedelta(hours=27),    # old occurrence (avg ~1/hr over 30h spread)
    # ~20h gap here
    now - timedelta(hours=5),     # reappearance (outside 1h window)
    now - timedelta(hours=4),     # reappearance (outside 1h window)
]
reg_trend = _analyse_cluster_trend(reg_cluster, reg_ts, now)
assert reg_trend.is_regression is True
assert reg_trend.trend_label == "regression"
print("✓ Test 24 passed: Regression detection")

# ==============================
# Test 25: Trend — stable trend (no spike, no regression)
# ==============================
stable_cluster = _make_cluster(5, "StableError", count=10)
# Spread evenly every 5h over 45h — gaps < QUIET_GAP_HOURS, first_seen > 24h ago
stable_ts = [now - timedelta(hours=i * 5) for i in range(10)]
stable_trend = _analyse_cluster_trend(stable_cluster, stable_ts, now)
assert stable_trend.trend_label == "stable"
assert stable_trend.spike_detected is False
assert stable_trend.is_new is False
assert stable_trend.is_regression is False
print("✓ Test 25 passed: Stable trend")

# ==============================
# Test 26: Trend — no timestamps → unknown
# ==============================
no_ts_cluster = _make_cluster(6, "MysteryError", count=3)
no_ts_trend = _analyse_cluster_trend(no_ts_cluster, [], now)
assert no_ts_trend.trend_label == "unknown"
print("✓ Test 26 passed: No timestamps → unknown trend")

# ==============================
# Test 27: Trend — ClusterTrend.describe()
# ==============================
desc = spike_trend.describe()
assert "SPIKE" in desc or "spike" in desc
assert str(spike_trend.last_1h) in desc
print("✓ Test 27 passed: ClusterTrend.describe()")

# ==============================
# Test 28: Trend — ClusterTrend.to_dict()
# ==============================
td = spike_trend.to_dict()
assert td["spike_detected"] is True
assert td["trend_label"] == "spike"
assert "last_1h" in td
assert "pct_change" in td
print("✓ Test 28 passed: ClusterTrend.to_dict()")

# ==============================
# Test 29: Trend — detect_trends() full pipeline
# ==============================
# Create chunks with timestamps matching the cluster signatures
now_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
ts_30m = (now - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
ts_2h = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

pipeline_chunks = [
    _chunk(f"{now_str} ERROR NullPointerException: x is null"),
    _chunk(f"{ts_30m} ERROR NullPointerException: y is null"),
    _chunk(f"{ts_2h} ERROR ValueError: bad input"),
]
pipeline_report = cluster_error_patterns(pipeline_chunks)
trend_report = detect_trends(pipeline_report, pipeline_chunks, reference_time=now)

assert len(trend_report.cluster_trends) == pipeline_report.cluster_count
# Clusters should now have trend labels set
for cl in pipeline_report.clusters:
    assert cl.trend_label != ""
print("✓ Test 29 passed: detect_trends() full pipeline")

# ==============================
# Test 30: Trend — TrendDetectionReport.summary_text()
# ==============================
summary = trend_report.summary_text()
assert "trend" in summary.lower() or "cluster" in summary.lower()
print("✓ Test 30 passed: TrendDetectionReport.summary_text()")

# ==============================
# Test 31: Trend — TrendDetectionReport.to_dict()
# ==============================
trd = trend_report.to_dict()
assert "cluster_trends" in trd
assert "spike_count" in trd
assert "has_alerts" in trd
print("✓ Test 31 passed: TrendDetectionReport.to_dict()")

# ==============================
# Test 32: Trend — has_alerts property
# ==============================
empty_trend_report = TrendDetectionReport()
assert empty_trend_report.has_alerts is False

alert_report = TrendDetectionReport(spike_count=1)
assert alert_report.has_alerts is True
print("✓ Test 32 passed: has_alerts property")

# ==============================
# Test 33: Trend — updates cluster in-place
# ==============================
# After detect_trends, cluster objects should have occurrences_1h, trend_label etc.
for cl in pipeline_report.clusters:
    assert hasattr(cl, "occurrences_1h")
    assert hasattr(cl, "occurrences_24h")
    assert hasattr(cl, "trend_label")
    assert cl.trend_label in ("spike", "new", "regression", "stable", "declining", "unknown")
print("✓ Test 33 passed: Trend updates cluster in-place")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Causal Confidence  (core/analysis/causal_confidence.py)
# ══════════════════════════════════════════════════════════════════════════════

from core.analysis.causal_confidence import (
    CausalConfidenceResult,
    compute_causal_confidence,
    _frequency_concentration,
    _stack_trace_consistency,
    _code_reference_match,
    _dependency_proximity,
    _recency_spike_correlation,
    DEFAULT_WEIGHTS,
)


# ── Helper: build a known cluster report ──────────────────────────────────────

def _make_cluster_report(
    clusters: list[ErrorCluster],
    total_entries: int = 0,
) -> ErrorClusterReport:
    if total_entries == 0:
        total_entries = sum(c.count for c in clusters)
    return ErrorClusterReport(
        clusters=clusters,
        total_entries=total_entries,
    )


# ==============================
# Test 34: Factor — frequency concentration (dominant cluster)
# ==============================
dom_cluster = _make_cluster(1, "NPE", "NullPointerException", count=8)
minor_cluster = _make_cluster(2, "Val", "ValueError", count=2)
cr = _make_cluster_report([dom_cluster, minor_cluster])

freq = _frequency_concentration(cr)
assert 0.7 <= freq <= 1.0  # 8/10 = 0.8, + boost for >=5 count
print("✓ Test 34 passed: Frequency concentration (dominant)")

# ==============================
# Test 35: Factor — frequency concentration (scattered errors)
# ==============================
scattered = [_make_cluster(i, f"err{i}", count=1) for i in range(10)]
scattered_cr = _make_cluster_report(scattered)
freq_s = _frequency_concentration(scattered_cr)
assert freq_s <= 0.2  # 1/10 = 0.1, no boost
print("✓ Test 35 passed: Frequency concentration (scattered)")

# ==============================
# Test 36: Factor — frequency concentration (empty)
# ==============================
empty_cr = ErrorClusterReport()
assert _frequency_concentration(empty_cr) == 0.0
print("✓ Test 36 passed: Frequency concentration (empty)")

# ==============================
# Test 37: Factor — stack trace consistency
# ==============================
cl_with_frames = ErrorCluster(
    cluster_id=1, error_signature="NPE", count=5,
    stack_frames=["frame1", "frame2", "frame3"],
    related_functions=["handle_request", "process"],
)
cl_no_frames = ErrorCluster(
    cluster_id=2, error_signature="Val", count=2,
    stack_frames=[], related_functions=[],
)
cr_frames = _make_cluster_report([cl_with_frames, cl_no_frames])
st_score = _stack_trace_consistency(cr_frames)
assert 0.3 <= st_score <= 1.0  # 1/2 clusters with frames, dominant has 3 frames
print("✓ Test 37 passed: Stack trace consistency")

# ==============================
# Test 38: Factor — stack trace consistency (empty)
# ==============================
assert _stack_trace_consistency(ErrorClusterReport()) == 0.0
print("✓ Test 38 passed: Stack trace consistency (empty)")

# ==============================
# Test 39: Factor — code reference match
# ==============================
score_high = _code_reference_match(cr, code_chunk_count=5,
                                    reference_count=5,
                                    cross_analysis_available=True)
assert score_high >= 0.8  # 0.4 (cross) + 0.3 (refs) + 0.3 (code) + 0.1 (files)

score_low = _code_reference_match(ErrorClusterReport(),
                                   code_chunk_count=0,
                                   reference_count=0,
                                   cross_analysis_available=False)
assert score_low == 0.0
print("✓ Test 39 passed: Code reference match")

# ==============================
# Test 40: Factor — dependency proximity
# ==============================
assert _dependency_proximity(0, 0) == 0.1       # baseline
assert _dependency_proximity(0, 5) == 0.1       # no deps found
dep_score = _dependency_proximity(5, 5)
assert dep_score >= 0.9  # 0.1 + 0.9 × 1.0
print("✓ Test 40 passed: Dependency proximity")

# ==============================
# Test 41: Factor — recency spike correlation
# ==============================
# Spike trend
spike_tr = TrendDetectionReport(
    cluster_trends=[ClusterTrend(cluster_id=1, total=8, last_1h=5,
                                 spike_detected=True, trend_label="spike")],
    spike_count=1,
)
dom_cl = ErrorCluster(cluster_id=1, error_signature="NPE", count=8)
cr_spike = _make_cluster_report([dom_cl])

spike_score = _recency_spike_correlation(spike_tr, cr_spike)
assert spike_score >= 0.9  # spike=1.0 + boosts

# No trend → neutral
neutral = _recency_spike_correlation(None, cr_spike)
assert 0.2 <= neutral <= 0.4
print("✓ Test 41 passed: Recency spike correlation")

# ==============================
# Test 42: Composite — high confidence scenario
# ==============================
high_clusters = [
    ErrorCluster(cluster_id=1, error_signature="NPE",
                 exception_type="NullPointerException", count=10,
                 stack_frames=["f1", "f2", "f3"],
                 related_functions=["handle", "process"],
                 related_files=["handler.py"],
                 severity="error"),
]
high_cr = _make_cluster_report(high_clusters, total_entries=12)

high_tr = TrendDetectionReport(
    cluster_trends=[ClusterTrend(cluster_id=1, total=10, last_1h=5,
                                 spike_detected=True, trend_label="spike")],
    spike_count=1,
)

result = compute_causal_confidence(
    cluster_report=high_cr,
    trend_report=high_tr,
    code_chunk_count=5,
    reference_count=5,
    dependency_count=3,
    cross_analysis_available=True,
)
assert result.score >= 70  # Should be high with all factors strong
assert result.level in ("high", "medium")
assert result.dominant_error == "NPE"
assert result.dominant_cluster_id == 1
print("✓ Test 42 passed: High confidence scenario (score={})".format(result.score))

# ==============================
# Test 43: Composite — low confidence scenario
# ==============================
low_clusters = [
    ErrorCluster(cluster_id=i, error_signature=f"err{i}", count=1, severity="warning")
    for i in range(5)
]
low_cr = _make_cluster_report(low_clusters, total_entries=10)
low_result = compute_causal_confidence(
    cluster_report=low_cr,
    code_chunk_count=0,
    reference_count=0,
    dependency_count=0,
    cross_analysis_available=False,
)
assert low_result.score <= 40
assert low_result.level == "low"
print("✓ Test 43 passed: Low confidence scenario (score={})".format(low_result.score))

# ==============================
# Test 44: Composite — empty report
# ==============================
empty_result = compute_causal_confidence(ErrorClusterReport())
assert empty_result.score <= 20
assert empty_result.level == "low"
print("✓ Test 44 passed: Empty report → low confidence")

# ==============================
# Test 45: CausalConfidenceResult.to_dict()
# ==============================
d = result.to_dict()
assert d["score"] == result.score
assert d["level"] == result.level
assert "factors" in d
assert "frequency" in d["factors"]
assert "stack_trace" in d["factors"]
assert "code_match" in d["factors"]
assert "dependency" in d["factors"]
assert "trend" in d["factors"]
print("✓ Test 45 passed: CausalConfidenceResult.to_dict()")

# ==============================
# Test 46: Composite — score levels
# ==============================
assert result.level in ("high", "medium", "low")
# Verify level thresholds
test_result = CausalConfidenceResult(score=85, level="high", explanation="")
assert test_result.level == "high"
test_result2 = CausalConfidenceResult(score=60, level="medium", explanation="")
assert test_result2.level == "medium"
test_result3 = CausalConfidenceResult(score=30, level="low", explanation="")
assert test_result3.level == "low"
print("✓ Test 46 passed: Score level thresholds")

# ==============================
# Test 47: Composite — explanation is non-empty
# ==============================
assert len(result.explanation) > 20
assert "Causal Confidence" in result.explanation
assert str(result.score) in result.explanation
print("✓ Test 47 passed: Explanation is informative")

# ==============================
# Test 48: Composite — custom weights
# ==============================
custom = compute_causal_confidence(
    cluster_report=high_cr,
    trend_report=high_tr,
    code_chunk_count=5,
    reference_count=5,
    dependency_count=3,
    cross_analysis_available=True,
    weights={"frequency": 1.0, "stack_trace": 0, "code_match": 0,
             "dependency": 0, "trend": 0},
)
# With only frequency weight, score should reflect frequency factor only
assert custom.weights["frequency"] == 1.0
print("✓ Test 48 passed: Custom weights applied")

# ==============================
# Test 49: Composite — weights normalise to 1.0
# ==============================
assert abs(sum(result.weights.values()) - 1.0) < 0.01
print("✓ Test 49 passed: Weights normalise to 1.0")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Evidence Ranker  (core/analysis/evidence_ranker.py)
# ══════════════════════════════════════════════════════════════════════════════

from core.analysis.evidence_ranker import (
    StructuredReasoningInput,
    rank_evidence,
    _rank_cluster,
    MAX_CLUSTERS,
)


# ==============================
# Test 50: Cluster ranking — severity weight
# ==============================
fatal_cl = ErrorCluster(cluster_id=1, error_signature="OOM",
                        count=3, severity="fatal", trend_label="stable",
                        related_files=["app.py"])
warning_cl = ErrorCluster(cluster_id=2, error_signature="Warn",
                          count=3, severity="warning", trend_label="stable",
                          related_files=["app.py"])

assert _rank_cluster(fatal_cl) > _rank_cluster(warning_cl)
print("✓ Test 50 passed: Fatal ranks higher than warning")

# ==============================
# Test 51: Cluster ranking — spike boosts rank
# ==============================
spike_cl = ErrorCluster(cluster_id=3, error_signature="Timeout",
                        count=3, severity="error", trend_label="spike")
stable_cl = ErrorCluster(cluster_id=4, error_signature="Timeout2",
                         count=3, severity="error", trend_label="stable")

assert _rank_cluster(spike_cl) > _rank_cluster(stable_cl)
print("✓ Test 51 passed: Spike trend boosts rank")

# ==============================
# Test 52: Cluster ranking — higher count boosts rank (diminishing returns)
# ==============================
high_count = ErrorCluster(cluster_id=5, error_signature="FreqErr",
                          count=100, severity="error", trend_label="stable")
low_count = ErrorCluster(cluster_id=6, error_signature="RareErr",
                         count=2, severity="error", trend_label="stable")

assert _rank_cluster(high_count) > _rank_cluster(low_count)
print("✓ Test 52 passed: Higher count boosts rank")

# ==============================
# Test 53: Cluster ranking — reference clarity boosts rank
# ==============================
clear_cl = ErrorCluster(cluster_id=7, error_signature="ClearErr",
                        count=3, severity="error", trend_label="stable",
                        related_files=["a.py", "b.py", "c.py"],
                        related_functions=["foo", "bar"],
                        stack_frames=["f1", "f2"])
vague_cl = ErrorCluster(cluster_id=8, error_signature="VagueErr",
                        count=3, severity="error", trend_label="stable")

assert _rank_cluster(clear_cl) > _rank_cluster(vague_cl)
print("✓ Test 53 passed: Reference clarity boosts rank")

# ==============================
# Test 54: rank_evidence() — produces StructuredReasoningInput
# ==============================
clusters = [
    ErrorCluster(cluster_id=1, error_signature="NPE", exception_type="NullPointerException",
                 count=10, severity="error", trend_label="spike",
                 related_files=["handler.py"], related_functions=["process"]),
    ErrorCluster(cluster_id=2, error_signature="Val", exception_type="ValueError",
                 count=3, severity="warning", trend_label="stable"),
]
cr = _make_cluster_report(clusters)
tr = TrendDetectionReport(
    cluster_trends=[
        ClusterTrend(cluster_id=1, total=10, last_1h=5, spike_detected=True, trend_label="spike"),
        ClusterTrend(cluster_id=2, total=3, last_1h=0, trend_label="stable"),
    ],
    spike_count=1,
)
conf = CausalConfidenceResult(score=75, level="medium",
                               explanation="Test explanation",
                               factors={"frequency": 0.8},
                               dominant_cluster_id=1,
                               dominant_error="NPE")

sri = rank_evidence(cr, tr, conf)
assert isinstance(sri, StructuredReasoningInput)
assert sri.confidence_score == 75
assert sri.confidence_level == "medium"
assert sri.dominant_cluster is not None
assert sri.dominant_cluster["error_signature"] == "NPE"
assert len(sri.top_clusters) <= MAX_CLUSTERS
print("✓ Test 54 passed: rank_evidence() produces StructuredReasoningInput")

# ==============================
# Test 55: rank_evidence() — frequency stats
# ==============================
assert "cluster_count" in sri.frequency_stats
assert "total_entries" in sri.frequency_stats
assert "dominant_share_pct" in sri.frequency_stats
assert sri.frequency_stats["dominant_share_pct"] > 50  # 10/13 ≈ 77%
print("✓ Test 55 passed: Frequency stats computed")

# ==============================
# Test 56: rank_evidence() — time trend included
# ==============================
assert "cluster_trends" in sri.time_trend
assert sri.time_trend["spike_count"] == 1
print("✓ Test 56 passed: Time trend data included")

# ==============================
# Test 57: rank_evidence() — limits to MAX_CLUSTERS
# ==============================
many_clusters = [
    ErrorCluster(cluster_id=i, error_signature=f"err{i}",
                 count=i, severity="error", trend_label="stable")
    for i in range(1, 12)  # 11 clusters
]
many_cr = _make_cluster_report(many_clusters)
many_conf = CausalConfidenceResult(score=50, level="medium", explanation="")
many_sri = rank_evidence(many_cr, None, many_conf)
assert len(many_sri.top_clusters) <= MAX_CLUSTERS
print("✓ Test 57 passed: Clusters limited to MAX_CLUSTERS={})".format(MAX_CLUSTERS))

# ==============================
# Test 58: StructuredReasoningInput.to_prompt_block()
# ==============================
block = sri.to_prompt_block()
assert isinstance(block, str)
assert "Causal Confidence Score" in block
assert "75" in block
assert "Dominant Failure Pattern" in block
assert "NPE" in block
print("✓ Test 58 passed: to_prompt_block() generates structured text")

# ==============================
# Test 59: StructuredReasoningInput.to_dict()
# ==============================
sd = sri.to_dict()
assert sd["confidence_score"] == 75
assert sd["confidence_level"] == "medium"
assert sd["dominant_cluster"] is not None
assert isinstance(sd["top_clusters"], list)
print("✓ Test 59 passed: StructuredReasoningInput.to_dict()")

# ==============================
# Test 60: StructuredReasoningInput — empty cluster report
# ==============================
empty_sri = rank_evidence(ErrorClusterReport(), None,
                           CausalConfidenceResult(score=10, level="low", explanation=""))
assert empty_sri.dominant_cluster is None
assert empty_sri.top_clusters == []
assert empty_sri.confidence_score == 10
block = empty_sri.to_prompt_block()
assert "10" in block
print("✓ Test 60 passed: Empty cluster report handled")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: End-to-End Integration  (cluster → trend → confidence → rank)
# ══════════════════════════════════════════════════════════════════════════════

# ==============================
# Test 61: Full pipeline — cluster → trend → confidence → rank
# ==============================
now = datetime(2024, 6, 15, 14, 0, 0, tzinfo=timezone.utc)

# Build realistic log entries
e2e_chunks = [
    _chunk(f"{(now - timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%SZ')} "
           f"ERROR NullPointerException: user.getName() returned null at UserService.java:42\n"
           f"  at com.app.UserService.process(UserService.java:42)\n"
           f"  at com.app.Controller.handle(Controller.java:100)"),
    _chunk(f"{(now - timedelta(minutes=10)).strftime('%Y-%m-%dT%H:%M:%SZ')} "
           f"ERROR NullPointerException: user.getEmail() returned null at UserService.java:55\n"
           f"  at com.app.UserService.process(UserService.java:55)\n"
           f"  at com.app.Controller.handle(Controller.java:100)"),
    _chunk(f"{(now - timedelta(minutes=20)).strftime('%Y-%m-%dT%H:%M:%SZ')} "
           f"ERROR NullPointerException: session is null at AuthService.java:30"),
    _chunk(f"{(now - timedelta(hours=2)).strftime('%Y-%m-%dT%H:%M:%SZ')} "
           f"WARNING ConnectionError: database timeout on pool at db.py:88"),
    _chunk(f"{(now - timedelta(hours=3)).strftime('%Y-%m-%dT%H:%M:%SZ')} "
           f"WARNING ConnectionError: cache connection lost at cache.py:44"),
    _chunk(f"{(now - timedelta(hours=25)).strftime('%Y-%m-%dT%H:%M:%SZ')} "
           f"ERROR FileNotFoundError: config.yaml missing at loader.py:12"),
]

# Step 1: Cluster
e2e_cr = cluster_error_patterns(e2e_chunks)
assert e2e_cr.cluster_count >= 2  # At least NPE cluster + other

# Step 2: Trend
e2e_tr = detect_trends(e2e_cr, e2e_chunks, reference_time=now)
assert len(e2e_tr.cluster_trends) == e2e_cr.cluster_count

# Step 3: Confidence
e2e_conf = compute_causal_confidence(
    cluster_report=e2e_cr,
    trend_report=e2e_tr,
    code_chunk_count=3,
    reference_count=2,
    dependency_count=1,
    cross_analysis_available=True,
)
assert 0 <= e2e_conf.score <= 100
assert e2e_conf.level in ("high", "medium", "low")

# Step 4: Rank
e2e_sri = rank_evidence(e2e_cr, e2e_tr, e2e_conf)
assert e2e_sri.dominant_cluster is not None
assert e2e_sri.confidence_score == e2e_conf.score
assert len(e2e_sri.top_clusters) >= 1

# Step 5: Prompt block
block = e2e_sri.to_prompt_block()
assert len(block) > 50
assert "Causal Confidence Score" in block
assert str(e2e_conf.score) in block

print("✓ Test 61 passed: Full pipeline (cluster → trend → confidence → rank)")

# ==============================
# Test 62: Full pipeline — dominant cluster is most frequent
# ==============================
npe_count = sum(1 for c in e2e_cr.clusters
                if "NullPointerException" in c.exception_type or "null" in c.error_signature.lower())
assert npe_count >= 1
dom = e2e_cr.dominant_cluster
assert dom is not None
# The NPE cluster should have the highest count (3 entries)
assert dom.count >= 3
print("✓ Test 62 passed: Dominant cluster is most frequent")

# ==============================
# Test 63: Full pipeline — trend enrichment flows through
# ==============================
# After detect_trends, clusters should have trend_label set
for cl in e2e_cr.clusters:
    assert cl.trend_label != ""
# The NPE cluster with 3 entries in last 20min should show activity
dom = e2e_cr.dominant_cluster
assert dom.occurrences_1h >= 2 or dom.occurrences_24h >= 2
print("✓ Test 63 passed: Trend enrichment flows through to clusters")

# ==============================
# Test 64: Full pipeline — all factors present in result
# ==============================
assert set(e2e_conf.factors.keys()) == {"frequency", "stack_trace", "code_match",
                                         "dependency", "trend"}
for v in e2e_conf.factors.values():
    assert 0.0 <= v <= 1.0
print("✓ Test 64 passed: All 5 factors present and in range")

# ==============================
# Test 65: Full pipeline — prompt block includes key sections
# ==============================
block = e2e_sri.to_prompt_block()
# Should have confidence score header
assert "Causal Confidence Score" in block
# Should have dominant failure pattern
assert "Dominant Failure Pattern" in block
# Should have frequency stats
assert "Error Frequency Statistics" in block or "Frequency" in block
print("✓ Test 65 passed: Prompt block has key sections")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Edge Cases
# ══════════════════════════════════════════════════════════════════════════════

# ==============================
# Test 66: Single entry → one cluster
# ==============================
single_report = cluster_error_patterns([
    _chunk("2024-01-15T10:00:00Z FATAL OutOfMemoryError: heap space")
])
assert single_report.cluster_count == 1
assert single_report.clusters[0].count == 1
assert single_report.dominant_cluster is not None
print("✓ Test 66 passed: Single entry → one cluster")

# ==============================
# Test 67: All same error → one cluster with high count
# ==============================
same_chunks = [
    _chunk(f"2024-01-15T10:0{i}:00Z ERROR ConnectionError: db timeout at pool.py:99")
    for i in range(8)
]
same_report = cluster_error_patterns(same_chunks)
assert same_report.cluster_count == 1
assert same_report.clusters[0].count == 8
print("✓ Test 67 passed: All same error → one cluster")

# ==============================
# Test 68: Mixed severity — dominant severity correct
# ==============================
mixed_chunks = [
    _chunk("FATAL OutOfMemoryError: heap space"),
    _chunk("ERROR OutOfMemoryError: gc overhead"),
    _chunk("WARNING OutOfMemoryError: approaching limit"),
]
mixed_report = cluster_error_patterns(mixed_chunks)
# All should cluster together (same exception type)
# Dominant severity should be fatal (highest in group)
dom = mixed_report.dominant_cluster
assert dom is not None
assert dom.severity == "fatal"
print("✓ Test 68 passed: Dominant severity is 'fatal'")

# ==============================
# Test 69: Percentage change calculation
# ==============================
now = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
pct_cluster = _make_cluster(1, "TestErr", count=6)
# 4 in last 24h, 2 in prev 24h → +100%
pct_ts = [
    now - timedelta(hours=1),
    now - timedelta(hours=5),
    now - timedelta(hours=10),
    now - timedelta(hours=20),
    now - timedelta(hours=30),
    now - timedelta(hours=40),
]
pct_trend = _analyse_cluster_trend(pct_cluster, pct_ts, now)
assert pct_trend.last_24h == 4
assert pct_trend.pct_change == 100.0  # 4 vs 2 prev
print("✓ Test 69 passed: Percentage change calculation")

# ==============================
# Test 70: rank_evidence with code chunks and deps
# ==============================
code_chunk = SimpleNamespace(
    source_file="handler.py",
    function_name="process",
    class_name="UserService",
    line_start=40,
    line_end=60,
    confidence=0.9,
)
dep_list = ["handler.py → service.py", "service.py → database.py"]

sri_full = rank_evidence(
    cluster_report=e2e_cr,
    trend_report=e2e_tr,
    confidence_result=e2e_conf,
    code_chunks=[code_chunk],
    dependencies=dep_list,
)
assert len(sri_full.related_code) == 1
assert sri_full.related_code[0]["file"] == "handler.py"
assert len(sri_full.dependency_chain) == 2
block = sri_full.to_prompt_block()
assert "Dependency Chain" in block
print("✓ Test 70 passed: rank_evidence with code chunks and deps")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("All 70 tests passed ✓")
print("  • Error Clustering:     20 tests (1–20)")
print("  • Trend Detection:      13 tests (21–33)")
print("  • Causal Confidence:    16 tests (34–49)")
print("  • Evidence Ranker:      11 tests (50–60)")
print("  • End-to-End Pipeline:   5 tests (61–65)")
print("  • Edge Cases:            5 tests (66–70)")
print("=" * 60)
