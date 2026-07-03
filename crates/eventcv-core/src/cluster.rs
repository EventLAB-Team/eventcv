//! Connected-component labelling on event frames — a simple clustering / segmentation step over a
//! dense [`EventFrame`] (e.g. a `count` or `binary` image). A pixel is **foreground** when any of
//! its channels is non-zero; a flood fill assigns each maximal 4- or 8-connected foreground blob a
//! distinct label `1..=k`, with background `0`. Returns a single-channel `u64` "labels" frame.

use std::{error::Error, fmt};

use crate::representation::{EventFrame, EventFrameData, RepresentationKind};

impl EventFrame {
    /// Labels 4- or 8-connected foreground components. `connectivity` must be `4` or `8`. The
    /// result is a single-channel `u64` frame the same size as `self`: `0` is background, and each
    /// component gets a distinct label in scan order starting at `1`.
    pub fn connected_components(&self, connectivity: u8) -> Result<EventFrame, ClusterError> {
        if connectivity != 4 && connectivity != 8 {
            return Err(ClusterError::InvalidConnectivity(connectivity));
        }
        let (channels, height, width) = self.shape();
        let plane = width
            .checked_mul(height)
            .ok_or(ClusterError::SizeOverflow)?;

        let foreground = foreground_mask(self.data(), plane, channels);
        let mut labels = vec![0_u64; plane];
        let mut next = 0_u64;
        let mut stack = Vec::new();
        let neighbours = neighbour_offsets(connectivity);

        for start in 0..plane {
            if !foreground[start] || labels[start] != 0 {
                continue;
            }
            next += 1;
            labels[start] = next;
            stack.push(start);
            while let Some(index) = stack.pop() {
                let (x, y) = (index % width, index / width);
                for &(dx, dy) in neighbours {
                    let (nx, ny) = (x as isize + dx, y as isize + dy);
                    if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                        continue;
                    }
                    let neighbour = ny as usize * width + nx as usize;
                    if foreground[neighbour] && labels[neighbour] == 0 {
                        labels[neighbour] = next;
                        stack.push(neighbour);
                    }
                }
            }
        }

        Ok(EventFrame::from_parts(
            EventFrameData::U64(labels),
            width,
            height,
            RepresentationKind::Labels,
            vec!["labels".to_owned()],
        ))
    }
}

/// A pixel is foreground when any channel is non-zero.
fn foreground_mask(data: &EventFrameData, plane: usize, channels: usize) -> Vec<bool> {
    let mut mask = vec![false; plane];
    let mut mark = |nonzero: &dyn Fn(usize) -> bool| {
        for c in 0..channels {
            let base = c * plane;
            for (i, slot) in mask.iter_mut().enumerate() {
                if nonzero(base + i) {
                    *slot = true;
                }
            }
        }
    };
    match data {
        EventFrameData::U8(v) => mark(&|k| v[k] != 0),
        EventFrameData::U16(v) => mark(&|k| v[k] != 0),
        EventFrameData::U64(v) => mark(&|k| v[k] != 0),
        EventFrameData::F32(v) => mark(&|k| v[k] != 0.0),
    }
    mask
}

fn neighbour_offsets(connectivity: u8) -> &'static [(isize, isize)] {
    const FOUR: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    const EIGHT: [(isize, isize); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];
    if connectivity == 4 {
        &FOUR
    } else {
        &EIGHT
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ClusterError {
    SizeOverflow,
    InvalidConnectivity(u8),
}

impl fmt::Display for ClusterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SizeOverflow => formatter.write_str("label field dimensions are too large"),
            Self::InvalidConnectivity(value) => {
                write!(formatter, "connectivity must be 4 or 8, got {value}")
            }
        }
    }
}

impl Error for ClusterError {}

#[cfg(test)]
mod tests {
    use super::ClusterError;
    use crate::representation::{EventFrame, EventFrameData, RepresentationKind};

    fn frame(width: usize, height: usize, pixels: &[u8]) -> EventFrame {
        EventFrame::from_parts(
            EventFrameData::U8(pixels.to_vec()),
            width,
            height,
            RepresentationKind::Count,
            vec!["count".to_owned()],
        )
    }

    fn labels(frame: &EventFrame) -> Vec<u64> {
        let EventFrameData::U64(values) = frame.data() else {
            panic!("labels are u64");
        };
        values.clone()
    }

    #[test]
    fn rejects_bad_connectivity() {
        let f = frame(2, 2, &[0, 0, 0, 0]);
        assert_eq!(
            f.connected_components(6).unwrap_err(),
            ClusterError::InvalidConnectivity(6)
        );
    }

    #[test]
    fn empty_frame_is_all_background() {
        let out = frame(3, 3, &[0; 9]).connected_components(8).unwrap();
        assert_eq!(out.shape(), (1, 3, 3));
        assert!(labels(&out).iter().all(|&l| l == 0));
    }

    #[test]
    fn two_diagonal_blobs_split_under_4_but_merge_under_8() {
        // Two pixels touching only at a corner: (0,0) and (1,1).
        let pixels = [1, 0, 0, 1];
        let four = frame(2, 2, &pixels).connected_components(4).unwrap();
        assert_eq!(labels(&four), vec![1, 0, 0, 2]);
        let eight = frame(2, 2, &pixels).connected_components(8).unwrap();
        assert_eq!(labels(&eight), vec![1, 0, 0, 1]);
    }

    #[test]
    fn foreground_is_any_nonzero_channel() {
        // Two channels; pixel 1 is foreground only via channel 1.
        let data = EventFrameData::U8(vec![0, 0, 0, 1]);
        let f = EventFrame::from_parts(
            data,
            2,
            1,
            RepresentationKind::Polarity,
            vec!["a".to_owned(), "b".to_owned()],
        );
        assert_eq!(labels(&f.connected_components(4).unwrap()), vec![0, 1]);
    }
}
