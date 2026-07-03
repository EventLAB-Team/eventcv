//! Interactive viewer. Frame-domain (image) representations are colour-mapped through
//! [`eventcv_core::viz::render_frame`] and shown as a textured quad; volumetric ones
//! (voxel / time surfaces / MCTS / point sets) become a 3-D point cloud. Both are rendered
//! on the GPU (Metal / Vulkan / DX12) by the [`gpu`] module — the old minifb CPU rasteriser
//! is gone. This file only *builds the scene*; `gpu` owns the window and draw loop.

use eventcv_core::representation::{EventFrame, EventFrameData, EventPointSet, RepresentationKind};
use eventcv_core::viz::{render_frame, Colormap, Rgb8Image};

mod gpu;

// Polarity colours shared by every cloud (positive = warm, negative = cool).
const POSITIVE: u32 = 0xff496c;
const NEGATIVE: u32 = 0x27c2ff;

/// A single splat in the normalised `[-1, 1]` view cube.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CloudPoint {
    x: f64,
    y: f64,
    z: f64,
    strength: f32,
    color: u32,
}

/// What the GPU should draw: a 2-D colour-mapped image, or a 3-D point cloud.
pub(crate) enum Scene {
    Image(Rgb8Image),
    Cloud {
        points: Vec<CloudPoint>,
        name: String,
    },
}

/// Renders `frame`: image kinds are colour-mapped (`colormap`, auto-contrast via `normalize`),
/// volumetric kinds become an orbitable point cloud.
pub(crate) fn view(frame: &EventFrame, colormap: Colormap, normalize: bool) -> Result<(), String> {
    let scene = match frame.kind() {
        RepresentationKind::Polarity
        | RepresentationKind::Binary
        | RepresentationKind::Count
        | RepresentationKind::Flow
        | RepresentationKind::Labels
        | RepresentationKind::Tencode => Scene::Image(render_frame(frame, colormap, normalize)),
        RepresentationKind::Voxel => Scene::Cloud {
            points: voxel_cloud(frame)?,
            name: "Voxel Grid".to_owned(),
        },
        RepresentationKind::TimeSurface => Scene::Cloud {
            points: time_surface_cloud(frame)?,
            name: "Time Surface".to_owned(),
        },
        RepresentationKind::AveragedTimeSurface => Scene::Cloud {
            points: time_surface_cloud(frame)?,
            name: "Averaged Time Surface".to_owned(),
        },
        RepresentationKind::Mcts => Scene::Cloud {
            points: mcts_cloud(frame)?,
            name: "MCTS".to_owned(),
        },
    };
    gpu::run(scene)
}

pub(crate) fn view_point_set(points: &EventPointSet) -> Result<(), String> {
    gpu::run(Scene::Cloud {
        points: point_set_cloud(points),
        name: "Point Set".to_owned(),
    })
}

fn voxel_cloud(frame: &EventFrame) -> Result<Vec<CloudPoint>, String> {
    let (channels, height, width) = frame.shape();
    let values = float_data(frame)?;
    let plane_len = checked_plane_len(width, height)?;
    let mut points = Vec::new();
    for channel in 0..channels {
        let z = normalize_index(channel, channels);
        for index in 0..plane_len {
            let value = values[channel * plane_len + index];
            if value != 0.0 {
                points.push(spatial_point(
                    index,
                    width,
                    height,
                    z,
                    value.abs(),
                    if value > 0.0 { POSITIVE } else { NEGATIVE },
                ));
            }
        }
    }
    Ok(points)
}

fn time_surface_cloud(frame: &EventFrame) -> Result<Vec<CloudPoint>, String> {
    let (channels, height, width) = frame.shape();
    if channels != 2 {
        return Err("tsurf frame must have two channels".to_owned());
    }
    let values = float_data(frame)?;
    let plane_len = checked_plane_len(width, height)?;
    let mut points = Vec::new();
    for channel in 0..channels {
        for index in 0..plane_len {
            let value = values[channel * plane_len + index];
            if value > 0.0 {
                points.push(spatial_point(
                    index,
                    width,
                    height,
                    f64::from(value) * 2.0 - 1.0,
                    value,
                    if channel == 0 { POSITIVE } else { NEGATIVE },
                ));
            }
        }
    }
    Ok(points)
}

fn mcts_cloud(frame: &EventFrame) -> Result<Vec<CloudPoint>, String> {
    let (channels, height, width) = frame.shape();
    if channels != 10 {
        return Err("mcts frame must have ten channels".to_owned());
    }
    let values = float_data(frame)?;
    let plane_len = checked_plane_len(width, height)?;
    let windows = channels / 2;
    let mut points = Vec::new();
    for channel in 0..channels {
        let z = normalize_index(channel % windows, windows);
        for index in 0..plane_len {
            let value = values[channel * plane_len + index];
            if value > 0.0 {
                points.push(spatial_point(
                    index,
                    width,
                    height,
                    z,
                    value,
                    if channel < windows {
                        NEGATIVE
                    } else {
                        POSITIVE
                    },
                ));
            }
        }
    }
    Ok(points)
}

fn point_set_cloud(points: &EventPointSet) -> Vec<CloudPoint> {
    points
        .data()
        .chunks_exact(4)
        .map(|point| CloudPoint {
            x: f64::from(point[0]) * 2.0 - 1.0,
            y: 1.0 - f64::from(point[1]) * 2.0,
            z: f64::from(point[2]) * 2.0 - 1.0,
            strength: 1.0,
            color: if point[3] > 0.0 { POSITIVE } else { NEGATIVE },
        })
        .collect()
}

fn spatial_point(
    index: usize,
    width: usize,
    height: usize,
    z: f64,
    strength: f32,
    color: u32,
) -> CloudPoint {
    CloudPoint {
        x: normalize_index(index % width, width),
        y: -normalize_index(index / width, height),
        z,
        strength,
        color,
    }
}

fn normalize_index(index: usize, length: usize) -> f64 {
    if length <= 1 {
        0.0
    } else {
        index as f64 / (length - 1) as f64 * 2.0 - 1.0
    }
}

fn float_data(frame: &EventFrame) -> Result<&[f32], String> {
    let EventFrameData::F32(values) = frame.data() else {
        return Err(format!(
            "{} frame must use float32 data",
            frame.kind().as_str()
        ));
    };
    Ok(values)
}

fn checked_plane_len(width: usize, height: usize) -> Result<usize, String> {
    width
        .checked_mul(height)
        .ok_or_else(|| "frame dimensions are too large".to_owned())
}

#[cfg(test)]
mod tests {
    use eventcv_core::representation::{
        AveragedTimeSurface, Binary, EventCount, Mcts, PointSet, Polarity, Representation, Tencode,
        TimeSurface, VoxelGrid,
    };
    use eventcv_core::viz::Colormap;

    use super::{point_set_cloud, time_surface_cloud, voxel_cloud, Scene};

    fn scene_of(frame: &eventcv_core::representation::EventFrame) -> Scene {
        // Mirrors `view` without opening a window: image vs cloud dispatch.
        use eventcv_core::representation::RepresentationKind::*;
        match frame.kind() {
            Polarity | Binary | Count | Flow | Labels | Tencode => Scene::Image(
                eventcv_core::viz::render_frame(frame, Colormap::Viridis, true),
            ),
            Voxel => Scene::Cloud {
                points: voxel_cloud(frame).unwrap(),
                name: String::new(),
            },
            TimeSurface | AveragedTimeSurface => Scene::Cloud {
                points: time_surface_cloud(frame).unwrap(),
                name: String::new(),
            },
            Mcts => Scene::Cloud {
                points: super::mcts_cloud(frame).unwrap(),
                name: String::new(),
            },
        }
    }

    #[test]
    fn every_representation_produces_a_scene() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/test/example.npz");
        let stream =
            eventcv_core::io::load(path, eventcv_core::io::LoadOptions::default()).unwrap();

        for frame in [
            Binary.generate(&stream).unwrap(),
            EventCount::new(true).generate(&stream).unwrap(),
            Polarity::default().generate(&stream).unwrap(),
            Tencode::default().generate(&stream).unwrap(),
        ] {
            assert!(matches!(scene_of(&frame), Scene::Image(_)));
        }
        for frame in [
            VoxelGrid::default().generate(&stream).unwrap(),
            TimeSurface::default().generate(&stream).unwrap(),
            AveragedTimeSurface::default().generate(&stream).unwrap(),
            Mcts::default().generate(&stream).unwrap(),
        ] {
            assert!(matches!(scene_of(&frame), Scene::Cloud { .. }));
        }
        assert!(!point_set_cloud(&PointSet.generate(&stream).unwrap()).is_empty());
    }
}
