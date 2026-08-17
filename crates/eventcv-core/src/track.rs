//! Object tracking — following connected components across slices.
//!
//! [`EventFrame::connected_components`] segments one frame into blobs but has no memory: run it on
//! consecutive slices and you get labels that mean nothing across time, since they are assigned in
//! scan order and renumber whenever anything moves. A [`Tracker`] adds the memory, matching this
//! frame's blobs to the tracks it already holds so an object keeps one identity while it is visible.
//!
//! ```no_run
//! # use eventcv_core::track::{Tracker, TrackerConfig};
//! # fn demo(frames: impl Iterator<Item = eventcv_core::representation::EventFrame>) {
//! let mut tracker = Tracker::new(TrackerConfig::default());
//! for frame in frames {
//!     for track in tracker.update(&frame).unwrap() {
//!         println!("track {} at {:?}", track.id, track.centroid);
//!     }
//! }
//! # }
//! ```
//!
//! # Association, and where it fails
//!
//! Blobs are matched to tracks greedily: closest pair first, then the next closest among what is
//! left, with a gating radius beyond which nothing matches. This is what the field actually uses at
//! this scale, and it is not optimal — a globally optimal assignment (Hungarian) can beat it when
//! several objects are close together.
//!
//! The consequence is worth stating plainly rather than discovering later: **two objects that pass
//! close to each other can swap identities.** Greedy matching commits to the closest pair before
//! considering the rest, so at the crossing point the wrong pairing can be cheaper. Nothing here
//! prevents that, and [`Tracker`] does not pretend otherwise — if identity through occlusion
//! matters, the appearance or motion model needed to resolve it is a larger piece of work than the
//! association rule.

use std::fmt;

use crate::cluster::ClusterError;
use crate::representation::{EventFrame, EventFrameData};

/// One connected component, measured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Blob {
    /// Label this blob carried in the frame it came from. Meaningful only within that frame.
    pub label: u64,
    /// Centre of mass, in pixels.
    pub centroid: (f64, f64),
    /// Bounding box as `(x_min, y_min, x_max, y_max)`, inclusive.
    pub bounds: (usize, usize, usize, usize),
    /// Number of pixels in the component.
    pub area: usize,
}

impl Blob {
    /// Longest side of the bounding box, in pixels.
    pub fn extent(&self) -> usize {
        let (x0, y0, x1, y1) = self.bounds;
        (x1 - x0).max(y1 - y0) + 1
    }
}

/// An object being followed across frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Track {
    /// Stable for the life of the track. Ids are never reused.
    pub id: u64,
    pub centroid: (f64, f64),
    /// Pixels per frame, from the last matched step. Zero until a track has been seen twice.
    pub velocity: (f64, f64),
    pub area: usize,
    /// Frames since the track was created.
    pub age: usize,
    /// Consecutive frames with no matching blob. Reset to zero on every match.
    pub missed: usize,
}

impl Track {
    /// Where this track is expected next, extrapolating its velocity.
    ///
    /// Matching against the prediction rather than the last position is what lets a fast object stay
    /// inside the gate: a blob moving 20 px per frame is 20 px away from where it was, but roughly
    /// zero from where it was going.
    fn predicted(&self) -> (f64, f64) {
        (
            self.centroid.0 + self.velocity.0,
            self.centroid.1 + self.velocity.1,
        )
    }
}

/// Association and lifetime settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackerConfig {
    /// Connectivity passed to [`EventFrame::connected_components`] — 4 or 8.
    pub connectivity: u8,
    /// Blobs smaller than this are ignored. The main noise control: a hot pixel or a stray event
    /// makes a one-pixel component, and without a floor every one of them becomes a track.
    pub min_area: usize,
    /// Maximum distance, in pixels, between a track's predicted position and a blob's centroid for
    /// the two to be matched.
    pub max_distance: f64,
    /// How many consecutive frames a track survives without a match before it is dropped. Above
    /// zero, this is what carries a track through a brief occlusion.
    pub max_missed: usize,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            connectivity: 8,
            min_area: 4,
            max_distance: 20.0,
            max_missed: 3,
        }
    }
}

/// Follows blobs across frames, keeping their identities.
pub struct Tracker {
    config: TrackerConfig,
    tracks: Vec<Track>,
    next_id: u64,
}

impl Tracker {
    pub fn new(config: TrackerConfig) -> Self {
        Self {
            config,
            tracks: Vec::new(),
            next_id: 1,
        }
    }

    /// Tracks currently alive, including any not matched in the most recent frame.
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    /// Forgets every track. Ids continue from where they left off, so a new track can never be
    /// confused with an old one in a log.
    pub fn reset(&mut self) {
        self.tracks.clear();
    }

    /// Segments `frame`, matches the blobs to existing tracks, and returns the live tracks.
    pub fn update(&mut self, frame: &EventFrame) -> Result<&[Track], ClusterError> {
        let blobs = blobs_of(frame, self.config.connectivity, self.config.min_area)?;
        self.associate(&blobs);
        Ok(&self.tracks)
    }

    /// Greedy nearest-neighbour matching between the current tracks and `blobs`.
    fn associate(&mut self, blobs: &[Blob]) {
        // Every candidate pairing inside the gate, cheapest first. Sorting once and walking it is
        // the whole algorithm: take the closest pair, retire both, continue.
        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
        for (track_index, track) in self.tracks.iter().enumerate() {
            let (px, py) = track.predicted();
            for (blob_index, blob) in blobs.iter().enumerate() {
                let distance =
                    ((blob.centroid.0 - px).powi(2) + (blob.centroid.1 - py).powi(2)).sqrt();
                if distance <= self.config.max_distance {
                    pairs.push((distance, track_index, blob_index));
                }
            }
        }
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut track_taken = vec![false; self.tracks.len()];
        let mut blob_taken = vec![false; blobs.len()];
        for (_, track_index, blob_index) in pairs {
            if track_taken[track_index] || blob_taken[blob_index] {
                continue;
            }
            track_taken[track_index] = true;
            blob_taken[blob_index] = true;

            let blob = blobs[blob_index];
            let track = &mut self.tracks[track_index];
            track.velocity = (
                blob.centroid.0 - track.centroid.0,
                blob.centroid.1 - track.centroid.1,
            );
            track.centroid = blob.centroid;
            track.area = blob.area;
            track.missed = 0;
        }

        // Unmatched tracks coast on their last velocity. Predicting through a gap is what makes
        // `max_missed` useful — a track that simply froze would fall outside its own gate by the
        // time the object reappeared.
        for (index, track) in self.tracks.iter_mut().enumerate() {
            track.age += 1;
            if !track_taken[index] {
                track.missed += 1;
                track.centroid = track.predicted();
            }
        }
        self.tracks
            .retain(|track| track.missed <= self.config.max_missed);

        for (index, blob) in blobs.iter().enumerate() {
            if !blob_taken[index] {
                self.tracks.push(Track {
                    id: self.next_id,
                    centroid: blob.centroid,
                    velocity: (0.0, 0.0),
                    area: blob.area,
                    age: 0,
                    missed: 0,
                });
                self.next_id += 1;
            }
        }
    }
}

/// Segments `frame` and measures every component at least `min_area` pixels.
pub fn blobs_of(
    frame: &EventFrame,
    connectivity: u8,
    min_area: usize,
) -> Result<Vec<Blob>, ClusterError> {
    let labels = frame.connected_components(connectivity)?;
    let (_, height, width) = labels.shape();
    let EventFrameData::U64(values) = labels.data() else {
        // `connected_components` documents a u64 label frame; anything else is a bug there, not
        // input this function should try to interpret.
        return Ok(Vec::new());
    };

    // One pass accumulating sums per label; labels are 1..=k so the index is the label itself.
    let count = values.iter().copied().max().unwrap_or(0) as usize;
    let mut sums = vec![
        (
            0.0_f64,
            0.0_f64,
            0_usize,
            usize::MAX,
            usize::MAX,
            0_usize,
            0_usize
        );
        count + 1
    ];
    for y in 0..height {
        for x in 0..width {
            let label = values[y * width + x] as usize;
            if label == 0 {
                continue;
            }
            let entry = &mut sums[label];
            entry.0 += x as f64;
            entry.1 += y as f64;
            entry.2 += 1;
            entry.3 = entry.3.min(x);
            entry.4 = entry.4.min(y);
            entry.5 = entry.5.max(x);
            entry.6 = entry.6.max(y);
        }
    }

    Ok((1..=count)
        .filter_map(|label| {
            let (sx, sy, area, x0, y0, x1, y1) = sums[label];
            if area < min_area.max(1) {
                return None;
            }
            Some(Blob {
                label: label as u64,
                centroid: (sx / area as f64, sy / area as f64),
                bounds: (x0, y0, x1, y1),
                area,
            })
        })
        .collect())
}

/// Errors specific to tracking. Segmentation failures surface as [`ClusterError`].
#[derive(Debug, PartialEq, Eq)]
pub enum TrackError {
    Cluster(ClusterError),
}

impl fmt::Display for TrackError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cluster(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for TrackError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::representation::EventFrame;

    /// A frame with filled squares at the given centres.
    fn frame_with(
        width: usize,
        height: usize,
        centres: &[(usize, usize)],
        size: usize,
    ) -> EventFrame {
        let mut data = vec![0_u8; width * height];
        for &(cx, cy) in centres {
            for dy in 0..size {
                for dx in 0..size {
                    let (x, y) = (cx + dx, cy + dy);
                    if x < width && y < height {
                        data[y * width + x] = 1;
                    }
                }
            }
        }
        EventFrame::intensity(EventFrameData::U8(data), width, height).unwrap()
    }

    #[test]
    fn blobs_measure_centroid_area_and_bounds() {
        // A 4x4 square with its corner at (10, 6) has its centre of mass at (11.5, 7.5).
        let frame = frame_with(32, 32, &[(10, 6)], 4);
        let blobs = blobs_of(&frame, 8, 1).unwrap();
        assert_eq!(blobs.len(), 1);
        assert_eq!(blobs[0].area, 16);
        assert!((blobs[0].centroid.0 - 11.5).abs() < 1e-9);
        assert!((blobs[0].centroid.1 - 7.5).abs() < 1e-9);
        assert_eq!(blobs[0].bounds, (10, 6, 13, 9));
        assert_eq!(blobs[0].extent(), 4);
    }

    #[test]
    fn separate_objects_are_separate_blobs() {
        let frame = frame_with(64, 64, &[(5, 5), (40, 40)], 4);
        assert_eq!(blobs_of(&frame, 8, 1).unwrap().len(), 2);
    }

    #[test]
    fn min_area_filters_noise() {
        // One real object and one single-pixel speck.
        let mut data = vec![0_u8; 32 * 32];
        for dy in 0..5 {
            for dx in 0..5 {
                data[(10 + dy) * 32 + 10 + dx] = 1;
            }
        }
        data[2 * 32 + 30] = 1;
        let frame = EventFrame::intensity(EventFrameData::U8(data), 32, 32).unwrap();
        assert_eq!(blobs_of(&frame, 8, 1).unwrap().len(), 2);
        assert_eq!(blobs_of(&frame, 8, 4).unwrap().len(), 1);
    }

    #[test]
    fn an_empty_frame_has_no_blobs() {
        let frame = frame_with(16, 16, &[], 0);
        assert!(blobs_of(&frame, 8, 1).unwrap().is_empty());
    }

    #[test]
    fn a_track_keeps_its_id_while_the_object_moves() {
        // The property the whole module exists for: connected-component labels renumber every
        // frame, but the track id must not.
        let mut tracker = Tracker::new(TrackerConfig::default());
        let mut ids = Vec::new();
        for step in 0..10 {
            let frame = frame_with(96, 64, &[(5 + step * 6, 20)], 5);
            let tracks = tracker.update(&frame).unwrap();
            assert_eq!(tracks.len(), 1, "step {step}");
            ids.push(tracks[0].id);
        }
        assert!(ids.windows(2).all(|w| w[0] == w[1]), "id changed: {ids:?}");
    }

    #[test]
    fn velocity_matches_the_motion() {
        let mut tracker = Tracker::new(TrackerConfig::default());
        for step in 0..5 {
            let frame = frame_with(96, 64, &[(5 + step * 6, 20)], 5);
            tracker.update(&frame).unwrap();
        }
        let track = tracker.tracks()[0];
        assert!(
            (track.velocity.0 - 6.0).abs() < 1e-6,
            "vx {}",
            track.velocity.0
        );
        assert!(track.velocity.1.abs() < 1e-6, "vy {}", track.velocity.1);
    }

    #[test]
    fn two_objects_get_two_ids() {
        let mut tracker = Tracker::new(TrackerConfig::default());
        for step in 0..6 {
            let frame = frame_with(96, 64, &[(5 + step * 4, 10), (5 + step * 4, 45)], 5);
            let tracks = tracker.update(&frame).unwrap();
            assert_eq!(tracks.len(), 2, "step {step}");
        }
        let ids: Vec<u64> = tracker.tracks().iter().map(|t| t.id).collect();
        assert_ne!(ids[0], ids[1]);
    }

    #[test]
    fn a_track_survives_a_brief_disappearance() {
        let config = TrackerConfig {
            max_missed: 3,
            ..TrackerConfig::default()
        };
        let mut tracker = Tracker::new(config);
        for step in 0..4 {
            tracker
                .update(&frame_with(96, 64, &[(5 + step * 5, 20)], 5))
                .unwrap();
        }
        let id = tracker.tracks()[0].id;

        // Two empty frames: the track coasts rather than dying.
        for _ in 0..2 {
            tracker.update(&frame_with(96, 64, &[], 0)).unwrap();
        }
        assert_eq!(tracker.tracks().len(), 1);
        assert_eq!(tracker.tracks()[0].id, id, "the id must survive the gap");

        // The object reappears roughly where its velocity predicted.
        let tracks = tracker.update(&frame_with(96, 64, &[(35, 20)], 5)).unwrap();
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].id, id, "and be recognised as the same object");
    }

    #[test]
    fn a_track_dies_after_max_missed() {
        let config = TrackerConfig {
            max_missed: 2,
            ..TrackerConfig::default()
        };
        let mut tracker = Tracker::new(config);
        tracker.update(&frame_with(64, 64, &[(20, 20)], 5)).unwrap();
        assert_eq!(tracker.tracks().len(), 1);
        for _ in 0..3 {
            tracker.update(&frame_with(64, 64, &[], 0)).unwrap();
        }
        assert!(tracker.tracks().is_empty(), "the track should have expired");
    }

    #[test]
    fn a_distant_jump_starts_a_new_track_rather_than_teleporting() {
        // The gate exists so an unrelated object across the frame is not mistaken for this one.
        let config = TrackerConfig {
            max_distance: 10.0,
            max_missed: 0,
            ..TrackerConfig::default()
        };
        let mut tracker = Tracker::new(config);
        let first = tracker.update(&frame_with(96, 96, &[(5, 5)], 5)).unwrap()[0].id;
        let tracks = tracker.update(&frame_with(96, 96, &[(80, 80)], 5)).unwrap();
        assert_eq!(tracks.len(), 1);
        assert_ne!(
            tracks[0].id, first,
            "a jump beyond the gate is a new object"
        );
    }

    #[test]
    fn ids_are_never_reused() {
        let mut tracker = Tracker::new(TrackerConfig {
            max_missed: 0,
            ..TrackerConfig::default()
        });
        let first = tracker.update(&frame_with(64, 64, &[(10, 10)], 5)).unwrap()[0].id;
        tracker.update(&frame_with(64, 64, &[], 0)).unwrap();
        let second = tracker.update(&frame_with(64, 64, &[(10, 10)], 5)).unwrap()[0].id;
        assert_ne!(first, second, "a new object must not inherit a retired id");

        tracker.reset();
        let third = tracker.update(&frame_with(64, 64, &[(10, 10)], 5)).unwrap()[0].id;
        assert!(third > second, "ids continue past a reset");
    }

    #[test]
    fn fast_motion_is_followed_by_predicting_ahead() {
        // Moving 15 px per frame with a 20 px gate: matching against the last position would still
        // fit, but matching against the prediction is what keeps this comfortable rather than
        // marginal. The test pins that fast tracks survive at all.
        let mut tracker = Tracker::new(TrackerConfig::default());
        let mut ids = Vec::new();
        for step in 0..6 {
            let frame = frame_with(160, 64, &[(5 + step * 15, 20)], 6);
            let tracks = tracker.update(&frame).unwrap();
            assert_eq!(tracks.len(), 1, "lost the object at step {step}");
            ids.push(tracks[0].id);
        }
        assert!(ids.windows(2).all(|w| w[0] == w[1]));
    }
}
