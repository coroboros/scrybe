#!/usr/bin/env python3
# Independent reference implementation of the pyannote speaker-diarization-3.1
# pipeline, transcribed stage by stage from pyannote/pyannote-audio 3.1.1 (MIT)
# and driven by the same ONNX models as scrybe (onnx-community exports of
# pyannote-segmentation-3.0 and wespeaker-voxceleb-resnet34-LM). It exists to
# pin scrybe's Rust port: both must produce structurally identical turns on the
# committed fixtures.
#
# Two deliberate deviations, mirrored exactly from the Rust port:
#   - (chunk, speaker) slots with no embedding (silent, or fewer than
#     MIN_FBANK_ROWS masked fbank rows) are dropped outright — stock pyannote
#     lets their NaN rows fall into an arbitrary cluster;
#   - the clean-mask gate is a flat CLEAN_MASK_GATE segmentation frames instead
#     of a value derived from the embedding model's minimum sample count.
#
# Usage: diarize_reference.py AUDIO.wav [--speakers N]
# Prints turns as JSON: [{"start": s, "end": e, "speaker": k}, ...] with
# speakers renumbered densely by first appearance in time.

import argparse
import glob
import os
import json
import sys
import wave

import numpy as np
import onnxruntime as ort
import scipy.fft
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 160000  # 10 s segmentation window
STEP_SAMPLES = 16000  # hopped by 1 s
SEG_FRAMES = 589  # segmentation output frames per window
FRAME = 10.0 / SEG_FRAMES  # frame grid resolution, anchored at 0
# Powerset classes in model output order: {}, {0}, {1}, {2}, {0,1}, {0,2}, {1,2}
POWERSET = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ],
    dtype=np.float32,
)
LOCAL_SPEAKERS = 3

# fbank (torchaudio.compliance.kaldi defaults as used by wespeaker)
FRAME_LEN = 400  # 25 ms
FRAME_SHIFT = 160  # 10 ms
FFT_SIZE = 512
NUM_MELS = 80
LOW_FREQ = 20.0
HIGH_FREQ = 8000.0
PREEMPH = 0.97

# clustering (pyannote/speaker-diarization-3.1 config)
CLUSTER_THRESHOLD = 0.7045654963945799
MIN_CLUSTER_SIZE = 12

# Rust-port deviations (see head comment)
CLEAN_MASK_GATE = 10  # segmentation frames
MIN_FBANK_ROWS = 25

SEG_MODEL_GLOB = (
    "~/.cache/huggingface/hub/models--onnx-community--pyannote-segmentation-3.0"
    "/snapshots/*/onnx/model.onnx"
)
EMB_MODEL_GLOB = (
    "~/.cache/huggingface/hub/models--onnx-community--wespeaker-voxceleb-resnet34-LM"
    "/snapshots/*/onnx/model.onnx"
)


def closest_frame(t: float) -> int:
    # pyannote.core SlidingWindow.closest_frame on the (start=0, d=step=FRAME)
    # grid: rint uses numpy's round-half-even.
    return int(np.rint(t / FRAME - 0.5))


def load_wav(path: str) -> np.ndarray:
    with wave.open(path) as w:
        assert w.getframerate() == SAMPLE_RATE, "expected 16 kHz"
        assert w.getnchannels() == 1, "expected mono"
        assert w.getsampwidth() == 2, "expected s16"
        raw = w.readframes(w.getnframes())
    return (np.frombuffer(raw, dtype=np.int16).astype(np.float32)) / 32768.0


def chunk_starts(num_samples: int) -> list[int]:
    """Chunk start offsets per pyannote Inference.slide (zero-padded tail)."""
    if num_samples >= WINDOW_SAMPLES:
        num_complete = (num_samples - WINDOW_SAMPLES) // STEP_SAMPLES + 1
    else:
        num_complete = 0
    starts = [c * STEP_SAMPLES for c in range(num_complete)]
    has_last = (
        num_samples < WINDOW_SAMPLES
        or (num_samples - WINDOW_SAMPLES) % STEP_SAMPLES > 0
    )
    if has_last:
        starts.append(num_complete * STEP_SAMPLES)
    return starts


def make_session(pattern: str) -> ort.InferenceSession:
    paths = glob.glob(pattern)
    if not paths:
        sys.exit(f"model not found: {pattern}")
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(
        paths[0], sess_options=opts, providers=["CPUExecutionProvider"]
    )


def run(session: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    # The onnx-community exports rename pyannote's "feats"/"embs" tensors;
    # resolve names from the session so the algorithm is model-file agnostic.
    return session.run(None, {session.get_inputs()[0].name: x})[0]


def mel_banks() -> np.ndarray:
    """Kaldi-scale triangular mel bank, (NUM_MELS, FFT_SIZE // 2)."""
    mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)
    mel_low, mel_high = mel(LOW_FREQ), mel(HIGH_FREQ)
    delta = (mel_high - mel_low) / (NUM_MELS + 1)
    bin_mels = mel(SAMPLE_RATE / FFT_SIZE * np.arange(FFT_SIZE // 2))
    left = mel_low + np.arange(NUM_MELS)[:, None] * delta
    center = left + delta
    right = center + delta
    up = (bin_mels - left) / (center - left)
    down = (right - bin_mels) / (right - center)
    return np.maximum(0.0, np.minimum(up, down)).astype(np.float32)


MEL_BANKS = mel_banks()
HAMMING = (
    0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(FRAME_LEN) / (FRAME_LEN - 1))
).astype(np.float32)


def fbank(waveform: np.ndarray) -> np.ndarray:
    """Kaldi fbank + per-utterance CMN, (num_frames, NUM_MELS) float32."""
    x = waveform * 32768.0
    n = 1 + (len(x) - FRAME_LEN) // FRAME_SHIFT
    idx = np.arange(FRAME_LEN)[None, :] + FRAME_SHIFT * np.arange(n)[:, None]
    frames = x[idx]
    frames = frames - frames.mean(axis=1, keepdims=True, dtype=np.float32)
    prev = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
    frames = (frames - PREEMPH * prev) * HAMMING
    padded = np.zeros((n, FFT_SIZE), dtype=np.float32)
    padded[:, :FRAME_LEN] = frames
    power = np.abs(scipy.fft.rfft(padded, axis=1)) ** 2
    mel = power[:, : FFT_SIZE // 2] @ MEL_BANKS.T
    logmel = np.log(np.maximum(mel, np.finfo(np.float32).eps))
    return logmel - logmel.mean(axis=0, keepdims=True)


def aggregate_uniform(
    per_chunk: np.ndarray, num_out: int, average: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Inference.aggregate with hamming=False, missing=0.0.

    per_chunk: (num_chunks, SEG_FRAMES, k), NaN marks missing values.
    Returns (aggregated (num_out, k), coverage mask).
    """
    _, _, k = per_chunk.shape
    total = np.zeros((num_out, k), dtype=np.float32)
    coverage = np.zeros((num_out, k), dtype=np.float32)
    seen = np.zeros((num_out, k), dtype=np.float32)
    for c, data in enumerate(per_chunk):
        start = closest_frame(c * 1.0)
        mask = 1.0 - np.isnan(data)
        total[start : start + SEG_FRAMES] += np.nan_to_num(data) * mask
        coverage[start : start + SEG_FRAMES] += mask
        seen[start : start + SEG_FRAMES] = np.maximum(
            seen[start : start + SEG_FRAMES], mask
        )
    out = total / np.maximum(coverage, 1e-12) if average else total
    out[seen == 0.0] = 0.0
    return out, seen


def cluster_train_embeddings(
    raw: np.ndarray, min_clusters: int, max_clusters: int, num_clusters
) -> np.ndarray:
    """AgglomerativeClustering.cluster (pyannote 3.1.1), verbatim port."""
    n = len(raw)
    min_cluster_size = min(MIN_CLUSTER_SIZE, max(1, round(0.1 * n)))
    if n == 1:
        return np.zeros(1, dtype=int)

    normalized = raw / np.linalg.norm(raw, axis=-1, keepdims=True)
    dendrogram = linkage(normalized, method="centroid", metric="euclidean")
    clusters = fcluster(dendrogram, CLUSTER_THRESHOLD, criterion="distance") - 1

    cluster_unique, cluster_counts = np.unique(clusters, return_counts=True)
    large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
    num_large_clusters = len(large_clusters)

    if num_large_clusters < min_clusters:
        num_clusters = min_clusters
    elif num_large_clusters > max_clusters:
        num_clusters = max_clusters

    if num_clusters is not None and num_large_clusters != num_clusters:
        # switch stopping criterion from inter-cluster distance to iteration
        # index and walk away from the optimal threshold until the number of
        # large clusters matches the target.
        _dendrogram = np.copy(dendrogram)
        _dendrogram[:, 2] = np.arange(n - 1)

        best_iteration = n - 1
        best_num_large_clusters = 1

        for iteration in np.argsort(np.abs(dendrogram[:, 2] - CLUSTER_THRESHOLD)):
            new_cluster_size = _dendrogram[iteration, 3]
            if new_cluster_size < min_cluster_size:
                continue

            clusters = fcluster(_dendrogram, iteration, criterion="distance") - 1
            cluster_unique, cluster_counts = np.unique(clusters, return_counts=True)
            large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
            num_large_clusters = len(large_clusters)

            if abs(num_large_clusters - num_clusters) < abs(
                best_num_large_clusters - num_clusters
            ):
                best_iteration = iteration
                best_num_large_clusters = num_large_clusters

            if num_large_clusters == num_clusters:
                break

        if best_num_large_clusters != num_clusters:
            clusters = fcluster(_dendrogram, best_iteration, criterion="distance") - 1
            cluster_unique, cluster_counts = np.unique(clusters, return_counts=True)
            large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
            num_large_clusters = len(large_clusters)

    if num_large_clusters == 0:
        clusters[:] = 0
        return clusters

    small_clusters = cluster_unique[cluster_counts < min_cluster_size]
    if len(small_clusters) == 0:
        return clusters

    # absorb each small cluster wholesale into the nearest large cluster,
    # by cosine distance between centroids of the NORMALIZED embeddings
    large_centroids = np.vstack(
        [np.mean(normalized[clusters == k], axis=0) for k in large_clusters]
    )
    small_centroids = np.vstack(
        [np.mean(normalized[clusters == k], axis=0) for k in small_clusters]
    )
    centroids_cdist = cdist(large_centroids, small_centroids, metric="cosine")
    for small_k, large_k in enumerate(np.argmin(centroids_cdist, axis=0)):
        clusters[clusters == small_clusters[small_k]] = large_clusters[large_k]

    _, clusters = np.unique(clusters, return_inverse=True)
    return clusters


def diarize(samples: np.ndarray, num_speakers) -> list[dict]:
    seg_session = make_session(os.path.expanduser(SEG_MODEL_GLOB))
    emb_session = make_session(os.path.expanduser(EMB_MODEL_GLOB))

    # --- segmentation: independent 10 s windows, hard powerset argmax decode
    starts = chunk_starts(len(samples))
    binary = np.zeros((len(starts), SEG_FRAMES, LOCAL_SPEAKERS), dtype=np.float32)
    windows = []
    for c, s in enumerate(starts):
        window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        tail = samples[s : s + WINDOW_SAMPLES]
        window[: len(tail)] = tail
        windows.append(window)
        logits = run(seg_session, window[None, None, :])[0]  # (SEG_FRAMES, 7)
        binary[c] = POWERSET[np.argmax(logits, axis=-1)]

    # --- frame-level speaker count: uniform average of per-frame speaker sums
    num_out = closest_frame(10.0 + (len(starts) - 1) * 1.0) + 1
    summed = binary.sum(axis=2, keepdims=True)
    count, _ = aggregate_uniform(summed, num_out, average=True)
    count = np.rint(count[:, 0]).astype(np.int64)  # half-even

    if count.max() == 0:
        return []

    # --- one embedding per (chunk, local speaker); dropped slots stay None
    embeddings = np.full((len(starts), LOCAL_SPEAKERS, 256), np.nan)
    has_embedding = np.zeros((len(starts), LOCAL_SPEAKERS), dtype=bool)
    overlap = binary.sum(axis=2) >= 2  # (num_chunks, SEG_FRAMES)
    upsample = np.arange(998) * SEG_FRAMES // 998
    for c in range(len(starts)):
        features = fbank(windows[c])  # (998, NUM_MELS)
        for s in range(LOCAL_SPEAKERS):
            mask = binary[c, :, s]
            clean = mask * ~overlap[c]
            used = clean if clean.sum() > CLEAN_MASK_GATE else mask
            rows = features[used[upsample] > 0.5]
            if len(rows) < MIN_FBANK_ROWS:
                continue
            embeddings[c, s] = run(emb_session, rows[None].astype(np.float32))[0]
            has_embedding[c, s] = True

    train_chunk, train_speaker = np.where(has_embedding)
    if len(train_chunk) == 0:
        return []
    train_raw = embeddings[train_chunk, train_speaker]  # raw, unnormalized

    # --- agglomerative clustering + reassignment to raw-mean centroids
    if num_speakers is not None:
        target = max(1, min(len(train_raw), num_speakers))
        min_clusters = max_clusters = target
        num_clusters = target
    else:
        min_clusters, max_clusters, num_clusters = 1, len(train_raw), None
        if min_clusters == max_clusters:
            num_clusters = min_clusters

    hard_clusters = np.full((len(starts), LOCAL_SPEAKERS), -2, dtype=int)
    if max_clusters < 2:
        hard_clusters[has_embedding] = 0
    else:
        train_clusters = cluster_train_embeddings(
            train_raw, min_clusters, max_clusters, num_clusters
        )
        centroids = np.vstack(
            [
                np.mean(train_raw[train_clusters == k], axis=0)
                for k in range(np.max(train_clusters) + 1)
            ]
        )
        # every slot with an embedding is reassigned to its nearest centroid
        distances = cdist(train_raw, centroids, metric="cosine")
        hard_clusters[train_chunk, train_speaker] = np.argmax(2 - distances, axis=1)

    if num_speakers is not None:
        count = np.minimum(count, num_speakers)

    # --- reconstruction: per-chunk cluster columns, summed on the frame grid
    num_clusters_final = int(hard_clusters.max()) + 1
    clustered = np.full(
        (len(starts), SEG_FRAMES, num_clusters_final), np.nan, dtype=np.float32
    )
    for c in range(len(starts)):
        for k in np.unique(hard_clusters[c]):
            if k < 0:
                continue
            clustered[c, :, k] = binary[c][:, hard_clusters[c] == k].max(axis=1)
    activations, _ = aggregate_uniform(clustered, num_out, average=False)

    if num_clusters_final < count.max():
        activations = np.pad(
            activations, ((0, 0), (0, int(count.max()) - num_clusters_final))
        )

    # per frame keep the top-count clusters (stable sort: ties break toward
    # the lower column index)
    order = np.argsort(-activations, axis=1, kind="stable")
    final = np.zeros_like(activations)
    for t in range(num_out):
        final[t, order[t, : count[t]]] = 1.0

    # --- frame runs -> segments, boundaries on frame midpoints
    mids = (np.arange(num_out) + 0.5) * FRAME
    turns = []
    for k in range(final.shape[1]):
        y = final[:, k]
        is_active = y[0] > 0.5
        start = mids[0]
        for t in range(1, num_out):
            if is_active and y[t] < 0.5:
                turns.append((start, mids[t], k))
                is_active = False
            elif not is_active and y[t] > 0.5:
                start = mids[t]
                is_active = True
        if is_active:
            turns.append((start, mids[-1], k))

    # renumber speakers densely by first appearance in time
    turns.sort(key=lambda t: (t[0], t[1], t[2]))
    renumber = {}
    for _, _, k in turns:
        if k not in renumber:
            renumber[k] = len(renumber)
    return [
        {"start": round(s, 6), "end": round(e, 6), "speaker": renumber[k]}
        for s, e, k in turns
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="path to a 16 kHz mono s16 wav file")
    parser.add_argument("--speakers", type=int, default=None, metavar="N")
    args = parser.parse_args()

    turns = diarize(load_wav(args.audio), args.speakers)
    print(json.dumps(turns, indent=2))


if __name__ == "__main__":
    main()
