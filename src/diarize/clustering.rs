//! Agglomerative clustering of (chunk, local-speaker) embeddings — the port of
//! pyannote's `AgglomerativeClustering` with the speaker-diarization-3.1
//! calibration: centroid linkage over L2-normalized embeddings in Euclidean
//! space, dendrogram cut at a calibrated distance, small clusters absorbed
//! into their nearest large neighbour, then every embedding (re)assigned to
//! the nearest raw-mean centroid.
//!
//! kodama squares the condensed input and square-roots the step heights for
//! centroid linkage, so distances in and heights out are plain Euclidean —
//! the same convention as scipy, whose calibrated threshold this uses.

use kodama::{Dendrogram, Method, linkage};

/// Dendrogram cut distance calibrated by pyannote for wespeaker resnet34-LM
/// embeddings (speaker-diarization-3.1 `config.yaml`).
const CLUSTER_THRESHOLD: f64 = 0.704_565_496_394_579_9;
/// Clusters smaller than this are absorbed into their nearest large cluster.
const MIN_CLUSTER_SIZE: usize = 12;

/// Sentinel for a (chunk, speaker) slot with no usable cluster: silent in the
/// chunk, or too little clean speech to embed. Distinct from a real cluster id.
pub(crate) const NO_CLUSTER: i32 = -2;

pub(crate) struct ClusterOutcome {
    /// Cluster id per flat (chunk * 3 + local_speaker) slot, or [`NO_CLUSTER`].
    pub labels: Vec<i32>,
    pub num_clusters: usize,
}

/// Cluster the embeddings. `embeddings[i]` is `None` when extraction failed;
/// `active[i]` is whether the local speaker has any segmentation activity.
/// `num_speakers` pins the exact cluster count (best effort, as in pyannote:
/// the constrained cut lands as close as the dendrogram allows).
pub(crate) fn cluster_speakers(
    embeddings: &[Option<Vec<f32>>],
    active: &[bool],
    num_speakers: Option<usize>,
) -> ClusterOutcome {
    debug_assert_eq!(embeddings.len(), active.len());

    // Train set: active speakers whose embedding extraction succeeded.
    let train: Vec<usize> = (0..embeddings.len())
        .filter(|&i| active[i] && embeddings[i].is_some())
        .collect();
    let num_train = train.len();

    if num_train == 0 {
        return ClusterOutcome {
            labels: vec![NO_CLUSTER; embeddings.len()],
            num_clusters: 0,
        };
    }

    let train_raw: Vec<&[f32]> = train
        .iter()
        .map(|&i| embeddings[i].as_deref().unwrap_or(&[]))
        .collect();
    let train_normed: Vec<Vec<f64>> = train_raw.iter().map(|e| l2_normalized(e)).collect();

    let train_clusters = cluster_train(&train_normed, num_speakers);
    let num_clusters = train_clusters.iter().max().map_or(0, |&m| m + 1);

    // Centroids are means of the RAW train embeddings (pyannote normalizes a
    // copy for linkage only); assignment is by cosine distance, so only the
    // centroid's direction carries the norm weighting.
    let dim = train_raw[0].len();
    let mut centroids = vec![vec![0.0_f64; dim]; num_clusters];
    let mut counts = vec![0usize; num_clusters];
    for (t, &cluster) in train_clusters.iter().enumerate() {
        counts[cluster] += 1;
        for (c, &v) in centroids[cluster].iter_mut().zip(train_raw[t]) {
            *c += f64::from(v);
        }
    }
    for (centroid, &count) in centroids.iter_mut().zip(&counts) {
        for c in centroid.iter_mut() {
            *c /= count as f64;
        }
    }

    // (Re)assign every embeddable slot to its nearest centroid — train
    // embeddings included, which may switch clusters (deliberate in pyannote).
    // Slots that are inactive or failed extraction stay unclustered; pyannote
    // instead lets NaN embeddings fall into an arbitrary cluster, which only
    // spreads noise (sherpa drops them too).
    let labels = embeddings
        .iter()
        .zip(active)
        .map(|(embedding, &is_active)| match embedding {
            Some(e) if is_active => {
                let mut best = 0;
                let mut best_d = f64::INFINITY;
                for (k, centroid) in centroids.iter().enumerate() {
                    let d = cosine_distance(e, centroid);
                    if d < best_d {
                        best_d = d;
                        best = k;
                    }
                }
                best as i32
            }
            _ => NO_CLUSTER,
        })
        .collect();

    ClusterOutcome {
        labels,
        num_clusters,
    }
}

/// Label the train embeddings: linkage, threshold (or constrained) cut, and
/// small-cluster absorption. Returns dense cluster ids `0..num_clusters`.
fn cluster_train(normed: &[Vec<f64>], num_speakers: Option<usize>) -> Vec<usize> {
    let num_train = normed.len();
    // pyannote: min(12, max(1, round(0.1 * n))) with Python's half-even round.
    let min_cluster_size =
        MIN_CLUSTER_SIZE.min(((num_train as f64) * 0.1).round_ties_even().max(1.0) as usize);

    if num_train == 1 {
        return vec![0];
    }
    // `max_clusters < 2` skips clustering entirely (pyannote set_num_clusters).
    if num_speakers.is_some_and(|n| n < 2) {
        return vec![0; num_train];
    }

    let mut condensed = condensed_euclidean(normed);
    let dendrogram = linkage(&mut condensed, num_train, Method::Centroid);

    let mut clusters = cut_by_threshold(&dendrogram, num_train, CLUSTER_THRESHOLD);
    let mut large = count_large(&clusters, min_cluster_size);

    if let Some(target) = num_speakers {
        let target = target.clamp(1, num_train);
        if large.len() != target {
            clusters = constrained_cut(&dendrogram, num_train, target, min_cluster_size);
            large = count_large(&clusters, min_cluster_size);
        }
    }

    absorb_small_clusters(&mut clusters, &large, normed);
    densify(&mut clusters);
    clusters
}

/// Condensed upper-triangle Euclidean distances, scipy `pdist` order.
fn condensed_euclidean(points: &[Vec<f64>]) -> Vec<f64> {
    let n = points.len();
    let mut out = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let d: f64 = points[i]
                .iter()
                .zip(&points[j])
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            out.push(d.sqrt());
        }
    }
    out
}

/// scipy `fcluster(criterion="distance")`: merge steps whose monotonic
/// height envelope (max over the step and its descendants — centroid linkage
/// can produce inversions) stays within `threshold`, then label the resulting
/// components. Returns non-dense labels (indices into an arbitrary numbering).
fn cut_by_threshold(dendrogram: &Dendrogram<f64>, num_obs: usize, threshold: f64) -> Vec<usize> {
    let steps = dendrogram.steps();
    let mut envelope = vec![0.0_f64; steps.len()];
    for (i, step) in steps.iter().enumerate() {
        let mut h = step.dissimilarity;
        for &child in &[step.cluster1, step.cluster2] {
            if child >= num_obs {
                h = h.max(envelope[child - num_obs]);
            }
        }
        envelope[i] = h;
    }

    let mut uf = UnionFind::new(num_obs);
    for (i, step) in steps.iter().enumerate() {
        if envelope[i] <= threshold {
            let a = leaf_of(steps, num_obs, step.cluster1);
            let b = leaf_of(steps, num_obs, step.cluster2);
            uf.union(a, b);
        }
    }
    (0..num_obs).map(|i| uf.find(i)).collect()
}

/// pyannote's constrained cut: when the threshold yields the wrong number of
/// large clusters, re-cut by merge iteration, scanning iterations ordered by
/// |height − threshold| and skipping merges that create clusters smaller than
/// `min_cluster_size`; keep the cut whose large-cluster count lands closest
/// to the target.
fn constrained_cut(
    dendrogram: &Dendrogram<f64>,
    num_obs: usize,
    target: usize,
    min_cluster_size: usize,
) -> Vec<usize> {
    let steps = dendrogram.steps();
    let mut order: Vec<usize> = (0..steps.len()).collect();
    order.sort_by(|&a, &b| {
        let da = (steps[a].dissimilarity - CLUSTER_THRESHOLD).abs();
        let db = (steps[b].dissimilarity - CLUSTER_THRESHOLD).abs();
        da.total_cmp(&db)
    });

    let mut best_iteration = steps.len().saturating_sub(1);
    let mut best_count = 1usize;
    for &iteration in &order {
        if steps[iteration].size < min_cluster_size {
            continue;
        }
        let clusters = apply_first_merges(steps, num_obs, iteration);
        let num_large = count_large(&clusters, min_cluster_size).len();
        if num_large.abs_diff(target) < best_count.abs_diff(target) {
            best_iteration = iteration;
            best_count = num_large;
        }
        if num_large == target {
            return clusters;
        }
    }
    apply_first_merges(steps, num_obs, best_iteration)
}

/// Labels after applying merge steps `0..=iteration` (pyannote's
/// `fcluster(criterion="distance")` on iteration-indexed heights).
fn apply_first_merges(steps: &[kodama::Step<f64>], num_obs: usize, iteration: usize) -> Vec<usize> {
    let mut uf = UnionFind::new(num_obs);
    for step in &steps[..=iteration] {
        let a = leaf_of(steps, num_obs, step.cluster1);
        let b = leaf_of(steps, num_obs, step.cluster2);
        uf.union(a, b);
    }
    (0..num_obs).map(|i| uf.find(i)).collect()
}

/// Any leaf observation contained in dendrogram node `id`.
fn leaf_of(steps: &[kodama::Step<f64>], num_obs: usize, id: usize) -> usize {
    let mut id = id;
    while id >= num_obs {
        id = steps[id - num_obs].cluster1;
    }
    id
}

/// The labels of clusters with at least `min_cluster_size` members.
fn count_large(clusters: &[usize], min_cluster_size: usize) -> Vec<usize> {
    let mut counts = std::collections::HashMap::new();
    for &c in clusters {
        *counts.entry(c).or_insert(0usize) += 1;
    }
    let mut large: Vec<usize> = counts
        .into_iter()
        .filter(|&(_, n)| n >= min_cluster_size)
        .map(|(c, _)| c)
        .collect();
    large.sort_unstable();
    large
}

/// Absorb small clusters wholesale into the nearest large cluster by centroid
/// cosine distance over the normalized embeddings. With no large cluster at
/// all, everything collapses to one cluster (pyannote's fallback).
fn absorb_small_clusters(clusters: &mut [usize], large: &[usize], normed: &[Vec<f64>]) {
    if large.is_empty() {
        clusters.fill(0);
        return;
    }
    let centroid = |members: &[usize]| -> Vec<f64> {
        let dim = normed[0].len();
        let mut c = vec![0.0; dim];
        for &m in members {
            for (ci, v) in c.iter_mut().zip(&normed[m]) {
                *ci += v;
            }
        }
        for ci in c.iter_mut() {
            *ci /= members.len() as f64;
        }
        c
    };

    let mut members: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, &c) in clusters.iter().enumerate() {
        members.entry(c).or_default().push(i);
    }
    let large_centroids: Vec<(usize, Vec<f64>)> =
        large.iter().map(|&c| (c, centroid(&members[&c]))).collect();

    for (&cluster, cluster_members) in &members {
        if large.contains(&cluster) {
            continue;
        }
        let small_centroid = centroid(cluster_members);
        let mut best = large_centroids[0].0;
        let mut best_d = f64::INFINITY;
        for (label, c) in &large_centroids {
            let d = cosine_distance(&small_centroid, c);
            if d < best_d {
                best_d = d;
                best = *label;
            }
        }
        for &m in cluster_members {
            clusters[m] = best;
        }
    }
}

/// Renumber labels to dense `0..k` by first appearance — deterministic. The id
/// order feeds `reconstruct`'s lower-index support-tie break, so on overlapped
/// audio with exactly tied support it may keep a different cluster than the
/// reference's `np.unique` numbering; the pick is DER-invariant (same
/// partition), and `to_turns` renumbers the final speakers by time regardless.
fn densify(clusters: &mut [usize]) {
    let mut map = std::collections::HashMap::new();
    let mut next = 0usize;
    for c in clusters.iter_mut() {
        let dense = *map.entry(*c).or_insert_with(|| {
            let d = next;
            next += 1;
            d
        });
        *c = dense;
    }
}

fn l2_normalized(e: &[f32]) -> Vec<f64> {
    let norm: f64 = e
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 {
        return vec![0.0; e.len()];
    }
    e.iter().map(|&v| f64::from(v) / norm).collect()
}

/// Cosine distance `1 - cos(a, b)`; a zero-norm side yields 1.0. Generic over
/// the left operand so raw f32 embeddings and f64 centroids share one path.
fn cosine_distance<A: Copy + Into<f64>>(a: &[A], b: &[f64]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&x, &y) in a.iter().zip(b) {
        let x: f64 = x.into();
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    1.0 - dot / (na.sqrt() * nb.sqrt())
}

struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    /// A tight bundle of nearly identical unit-ish vectors around a base
    /// direction, distinct enough per member to avoid zero distances.
    fn bundle(base: [f32; 4], n: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                let eps = 0.001 * (i as f32 + 1.0);
                vec![base[0] + eps, base[1], base[2] - eps, base[3]]
            })
            .collect()
    }

    fn as_options(groups: &[Vec<Vec<f32>>]) -> (Vec<Option<Vec<f32>>>, Vec<bool>) {
        let flat: Vec<Option<Vec<f32>>> = groups.iter().flatten().cloned().map(Some).collect();
        let active = vec![true; flat.len()];
        (flat, active)
    }

    #[test]
    fn two_distant_groups_form_two_clusters() {
        let a = bundle([1.0, 0.0, 0.0, 0.0], 14);
        let b = bundle([0.0, 1.0, 0.0, 0.0], 14);
        let (embeddings, active) = as_options(&[a, b]);

        let outcome = cluster_speakers(&embeddings, &active, None);
        assert_eq!(outcome.num_clusters, 2);
        let first = &outcome.labels[..14];
        let second = &outcome.labels[14..];
        assert!(first.iter().all(|&l| l == first[0]));
        assert!(second.iter().all(|&l| l == second[0]));
        assert_ne!(first[0], second[0]);
    }

    #[test]
    fn close_groups_merge_under_the_threshold() {
        // Two bundles two degrees apart: far below the calibrated cut.
        let a = bundle([1.0, 0.0, 0.0, 0.0], 14);
        let b = bundle([1.0, 0.035, 0.0, 0.0], 14);
        let (embeddings, active) = as_options(&[a, b]);

        let outcome = cluster_speakers(&embeddings, &active, None);
        assert_eq!(outcome.num_clusters, 1);
    }

    #[test]
    fn num_speakers_overrides_the_threshold_cut() {
        // The same two close bundles, but the caller pins 2 speakers.
        let a = bundle([1.0, 0.0, 0.0, 0.0], 14);
        let b = bundle([1.0, 0.035, 0.0, 0.0], 14);
        let (embeddings, active) = as_options(&[a, b]);

        let outcome = cluster_speakers(&embeddings, &active, Some(2));
        assert_eq!(outcome.num_clusters, 2);
    }

    #[test]
    fn small_cluster_is_absorbed_into_nearest_large_one() {
        // 14 + 14 large groups and a 2-member stray near group A: the stray
        // is under min_cluster_size and must be absorbed into A.
        let a = bundle([1.0, 0.0, 0.0, 0.0], 14);
        let stray = bundle([0.92, 0.39, 0.0, 0.0], 2);
        let b = bundle([0.0, 1.0, 0.0, 0.0], 14);
        let (embeddings, active) = as_options(&[a, stray, b]);

        let outcome = cluster_speakers(&embeddings, &active, None);
        assert_eq!(outcome.num_clusters, 2);
        // Stray members carry group A's label.
        assert_eq!(outcome.labels[14], outcome.labels[0]);
        assert_eq!(outcome.labels[15], outcome.labels[0]);
    }

    #[test]
    fn inactive_and_failed_slots_stay_unclustered() {
        let a = bundle([1.0, 0.0, 0.0, 0.0], 14);
        let (mut embeddings, mut active) = as_options(&[a]);
        embeddings.push(None); // extraction failed
        active.push(true);
        embeddings.push(Some(vec![1.0, 0.0, 0.0, 0.0])); // silent speaker
        active.push(false);

        let outcome = cluster_speakers(&embeddings, &active, None);
        assert_eq!(outcome.labels[14], NO_CLUSTER);
        assert_eq!(outcome.labels[15], NO_CLUSTER);
    }

    #[test]
    fn single_embedding_is_cluster_zero() {
        let outcome = cluster_speakers(&[Some(vec![1.0, 0.0])], &[true], None);
        assert_eq!(outcome.labels, vec![0]);
        assert_eq!(outcome.num_clusters, 1);
    }

    #[test]
    fn no_usable_embedding_yields_no_clusters() {
        let outcome = cluster_speakers(&[None, None], &[true, false], None);
        assert_eq!(outcome.labels, vec![NO_CLUSTER, NO_CLUSTER]);
        assert_eq!(outcome.num_clusters, 0);
    }
}
