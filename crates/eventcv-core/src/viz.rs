//! Frame → pixels rendering: the shared 2-D visualisation path used by PNG export and the
//! interactive viewer's image representations. Most [`EventFrame`]s reduce to a per-pixel
//! scalar field (signed for polarity/voxel/time-surface reprs, unsigned for count/binary),
//! which a [`Colormap`] turns into RGB. Two kinds have their own RGB path: Tencode (already an RGB
//! encoding) and Flow (Middlebury colour coding — direction → hue, speed → saturation).

use crate::representation::{EventFrame, EventFrameData, RepresentationKind};

/// A packed 8-bit RGB image (`pixels.len() == width * height * 3`, row-major).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgb8Image {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

/// Colour mapping applied to a frame's scalar field. Sequential maps (`Grayscale`,
/// `Viridis`, `Turbo`) suit unsigned reprs; `RedBlue` is a diverging map for signed
/// reprs (negative → blue, zero → black, positive → red), matching the viewer's polarity colours.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Colormap {
    Grayscale,
    #[default]
    Viridis,
    Turbo,
    RedBlue,
}

impl Colormap {
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "grayscale" | "gray" | "grey" => Self::Grayscale,
            "viridis" => Self::Viridis,
            "turbo" => Self::Turbo,
            "redblue" | "diverging" => Self::RedBlue,
            _ => return None,
        })
    }
}

/// Renders `frame` to an RGB image. `normalize` stretches the field to its own data range
/// (auto-contrast); when `false`, values are read at their natural scale (`f32` reprs in
/// `[0, 1]` / `[-1, 1]`, integer counts divided by 255). Diverging kinds ignore `colormap`
/// and use `RedBlue`; Tencode ignores it entirely (it is already an RGB encoding).
pub fn render_frame(frame: &EventFrame, colormap: Colormap, normalize: bool) -> Rgb8Image {
    let (_, height, width) = frame.shape();
    let plane_len = width * height;

    if frame.kind() == RepresentationKind::Tencode {
        return Rgb8Image {
            width,
            height,
            pixels: render_tencode(frame, plane_len, normalize),
        };
    }
    // Flow has its own RGB path: Middlebury colour coding (direction → hue, speed → saturation).
    if frame.kind() == RepresentationKind::Flow {
        return Rgb8Image {
            width,
            height,
            pixels: render_flow(frame, plane_len, normalize),
        };
    }

    let (field, signed) = scalar_field(frame, plane_len);
    let scale = field_scale(&field, signed, is_float(frame.data()), normalize);
    let colormap = if signed { Colormap::RedBlue } else { colormap };

    let mut pixels = Vec::with_capacity(plane_len * 3);
    for &value in &field {
        let normalized = value * scale;
        let [r, g, b] = if signed {
            colormap.sample_signed(normalized.clamp(-1.0, 1.0))
        } else {
            colormap.sample(normalized.clamp(0.0, 1.0))
        };
        pixels.extend_from_slice(&[r, g, b]);
    }
    Rgb8Image {
        width,
        height,
        pixels,
    }
}

/// Collapses a frame's channels to one scalar per pixel and reports whether it is signed.
fn scalar_field(frame: &EventFrame, plane_len: usize) -> (Vec<f64>, bool) {
    let (channels, _, _) = frame.shape();
    let data = frame.data();
    match frame.kind() {
        // One channel — the value itself, unsigned.
        RepresentationKind::Binary | RepresentationKind::Count | RepresentationKind::Labels => {
            ((0..plane_len).map(|i| value_at(data, i)).collect(), false)
        }
        // Flow is handled by render_flow before this call (Middlebury colour coding).
        RepresentationKind::Flow => (vec![0.0; plane_len], false),
        // Positive/negative planes — their difference (signed).
        RepresentationKind::Polarity
        | RepresentationKind::TimeSurface
        | RepresentationKind::AveragedTimeSurface => (
            (0..plane_len)
                .map(|i| value_at(data, i) - value_at(data, plane_len + i))
                .collect(),
            true,
        ),
        // MCTS stores negative windows then positive windows — sum each half, then their difference.
        RepresentationKind::Mcts => {
            let half = channels / 2;
            (
                (0..plane_len)
                    .map(|i| {
                        let neg: f64 = (0..half).map(|c| value_at(data, c * plane_len + i)).sum();
                        let pos: f64 = (half..channels)
                            .map(|c| value_at(data, c * plane_len + i))
                            .sum();
                        pos - neg
                    })
                    .collect(),
                true,
            )
        }
        // Voxel bins are signed contributions — sum them.
        RepresentationKind::Voxel => (
            (0..plane_len)
                .map(|i| {
                    (0..channels)
                        .map(|c| value_at(data, c * plane_len + i))
                        .sum()
                })
                .collect(),
            true,
        ),
        // Tencode is handled before this call.
        RepresentationKind::Tencode => (vec![0.0; plane_len], false),
    }
}

/// The reciprocal of the field's display extent, so `value * scale` lands in `[0, 1]` (unsigned)
/// or `[-1, 1]` (signed). The extent is a **high percentile** of the non-zero magnitudes rather
/// than the raw max, so a handful of outliers (e.g. spurious large flow vectors) don't crush the
/// rest of the field to black; values above the percentile clamp to the top colour.
fn field_scale(field: &[f64], signed: bool, is_float: bool, normalize: bool) -> f64 {
    if normalize {
        let extent = robust_extent(field, signed);
        if extent > 0.0 {
            1.0 / extent
        } else {
            0.0
        }
    } else if is_float {
        1.0 // f32 reprs already live in [0, 1] / [-1, 1]
    } else {
        1.0 / 255.0 // integer counts read as 8-bit intensities
    }
}

/// The 99th percentile of the field's non-zero magnitudes — a display max robust to a few
/// outliers. Falls back to the raw max when there are too few non-zero samples for a percentile
/// to be meaningful (e.g. a sparse count image), preserving the "busiest pixel → top" behaviour.
fn robust_extent(field: &[f64], signed: bool) -> f64 {
    let mut mags: Vec<f64> = field
        .iter()
        .map(|&v| if signed { v.abs() } else { v })
        .filter(|&v| v > 0.0)
        .collect();
    if mags.len() < 100 {
        return mags.iter().copied().fold(0.0_f64, f64::max);
    }
    let index = (((mags.len() as f64) * 0.99).ceil() as usize - 1).min(mags.len() - 1);
    mags.select_nth_unstable_by(index, f64::total_cmp);
    mags[index]
}

/// Renders a two-channel flow frame with the **Middlebury** colour coding (Baker et al.): the flow
/// *direction* selects a hue from a fixed colour wheel, and the *speed* sets saturation — zero flow
/// is white, faster flow is more vivid. Speed is normalised by a robust percentile of the field
/// (or `1.0` when `normalize` is false, treating values as already in px/ms ≈ [0, 1]).
/// Gamma applied to normalised flow speed before colour-coding, to lift the skewed low end.
const FLOW_GAMMA: f64 = 0.5;

fn render_flow(frame: &EventFrame, plane_len: usize, normalize: bool) -> Vec<u8> {
    let data = frame.data();
    let magnitude = |i: usize| value_at(data, i).hypot(value_at(data, plane_len + i));
    let scale = if normalize {
        let mags: Vec<f64> = (0..plane_len).map(magnitude).collect();
        let extent = robust_extent(&mags, false);
        if extent > 0.0 {
            1.0 / extent
        } else {
            0.0
        }
    } else {
        1.0
    };

    let wheel = flow_color_wheel();
    let ncols = wheel.len();
    let mut pixels = Vec::with_capacity(plane_len * 3);
    for i in 0..plane_len {
        let (fx, fy) = (value_at(data, i), value_at(data, plane_len + i));
        // Perceptual gamma on the normalised speed: event flow is heavily skewed toward small
        // magnitudes, so a plain linear scale leaves almost everything near white. `√` expands the
        // low end so slow flow is still visible while the top stays vivid.
        let rad = (magnitude(i) * scale).powf(FLOW_GAMMA);
        // Direction → position on the colour wheel (Baker et al. use atan2(-v, -u)).
        let angle = (-fy).atan2(-fx) / std::f64::consts::PI; // [-1, 1]
        let fk = (angle + 1.0) / 2.0 * (ncols as f64 - 1.0);
        let k0 = fk.floor() as usize;
        let k1 = (k0 + 1) % ncols;
        let f = fk - k0 as f64;
        let mut rgb = [0_u8; 3];
        for (channel, slot) in rgb.iter_mut().enumerate() {
            let base = (1.0 - f) * wheel[k0][channel] + f * wheel[k1][channel];
            // Saturate with speed: rad=0 → white, rad=1 → full colour, rad>1 → darkened.
            let col = if rad <= 1.0 {
                1.0 - rad * (1.0 - base)
            } else {
                base * 0.75
            };
            *slot = (255.0 * col).round().clamp(0.0, 255.0) as u8;
        }
        pixels.extend_from_slice(&rgb);
    }
    pixels
}

/// The canonical 55-entry Middlebury flow colour wheel (values in `[0, 1]`), stepping through
/// red→yellow→green→cyan→blue→magenta→red so that each flow direction maps to a distinct hue.
fn flow_color_wheel() -> Vec<[f64; 3]> {
    const SEGMENTS: [(usize, [f64; 3], [f64; 3]); 6] = [
        (15, [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]), // red → yellow
        (6, [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]),  // yellow → green
        (4, [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]),  // green → cyan
        (11, [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]), // cyan → blue
        (13, [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]), // blue → magenta
        (6, [1.0, 0.0, 1.0], [1.0, 0.0, 0.0]),  // magenta → red
    ];
    let mut wheel = Vec::with_capacity(55);
    for (count, from, to) in SEGMENTS {
        for step in 0..count {
            let t = step as f64 / count as f64;
            wheel.push([
                from[0] + t * (to[0] - from[0]),
                from[1] + t * (to[1] - from[1]),
                from[2] + t * (to[2] - from[2]),
            ]);
        }
    }
    wheel
}

fn render_tencode(frame: &EventFrame, plane_len: usize, normalize: bool) -> Vec<u8> {
    let data = frame.data();
    let scale = if normalize {
        let max = (0..plane_len * 3)
            .map(|i| value_at(data, i))
            .fold(0.0, f64::max);
        if max > 0.0 {
            255.0 / max
        } else {
            0.0
        }
    } else {
        1.0
    };
    let channel = |plane: usize, i: usize| {
        (value_at(data, plane * plane_len + i) * scale)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    let mut pixels = Vec::with_capacity(plane_len * 3);
    for i in 0..plane_len {
        pixels.extend_from_slice(&[channel(0, i), channel(1, i), channel(2, i)]);
    }
    pixels
}

fn value_at(data: &EventFrameData, index: usize) -> f64 {
    match data {
        EventFrameData::U8(values) => f64::from(values[index]),
        EventFrameData::U16(values) => f64::from(values[index]),
        EventFrameData::U64(values) => values[index] as f64,
        EventFrameData::F32(values) => f64::from(values[index]),
    }
}

fn is_float(data: &EventFrameData) -> bool {
    matches!(data, EventFrameData::F32(_))
}

impl Colormap {
    /// Maps `t ∈ [0, 1]` to RGB for a sequential colormap.
    fn sample(self, t: f64) -> [u8; 3] {
        match self {
            Self::Grayscale => {
                let v = (t * 255.0).round() as u8;
                [v, v, v]
            }
            Self::Viridis => interpolate(&VIRIDIS, t),
            Self::Turbo => interpolate(&TURBO, t),
            // A diverging map used on unsigned data folds to its warm half.
            Self::RedBlue => self.sample_signed(t),
        }
    }

    /// Maps `s ∈ [-1, 1]` to RGB for the diverging red/blue map (negative → blue,
    /// positive → red) on a black background.
    fn sample_signed(self, s: f64) -> [u8; 3] {
        match self {
            Self::RedBlue => {
                let positive = s.max(0.0);
                let negative = (-s).max(0.0);
                [
                    (positive * 255.0).round() as u8,
                    ((positive.min(negative)) * 40.0).round() as u8,
                    (negative * 255.0).round() as u8,
                ]
            }
            // Sequential maps fold the signed field onto their magnitude.
            other => other.sample(s.abs()),
        }
    }
}

/// Linear interpolation over an anchor table of RGB control points.
fn interpolate(anchors: &[[u8; 3]], t: f64) -> [u8; 3] {
    let last = anchors.len() - 1;
    let position = t.clamp(0.0, 1.0) * last as f64;
    let lower = position.floor() as usize;
    if lower >= last {
        return anchors[last];
    }
    let frac = position - lower as f64;
    let a = anchors[lower];
    let b = anchors[lower + 1];
    std::array::from_fn(|c| {
        (f64::from(a[c]) + (f64::from(b[c]) - f64::from(a[c])) * frac).round() as u8
    })
}

// Compact anchor tables (interpolated) — close enough to matplotlib's for previews.
const VIRIDIS: [[u8; 3]; 9] = [
    [68, 1, 84],
    [72, 40, 120],
    [62, 74, 137],
    [49, 104, 142],
    [38, 130, 142],
    [31, 158, 137],
    [53, 183, 121],
    [110, 206, 88],
    [253, 231, 37],
];

const TURBO: [[u8; 3]; 11] = [
    [48, 18, 59],
    [61, 79, 195],
    [54, 138, 247],
    [33, 192, 225],
    [39, 232, 166],
    [127, 251, 86],
    [191, 235, 49],
    [240, 190, 50],
    [251, 128, 44],
    [225, 58, 20],
    [122, 4, 3],
];

#[cfg(test)]
mod tests {
    use super::{render_frame, Colormap, Rgb8Image};
    use crate::representation::{
        AveragedTimeSurface, Binary, EventCount, EventFrame, EventFrameData, Representation,
        RepresentationKind, Tencode,
    };
    use crate::EventStream;
    use ndarray::array;

    fn pixel(image: &Rgb8Image, x: usize, y: usize) -> [u8; 3] {
        let i = (y * image.width + x) * 3;
        [image.pixels[i], image.pixels[i + 1], image.pixels[i + 2]]
    }

    #[test]
    fn count_frame_maps_the_busiest_pixel_to_the_colormap_top() {
        let stream = EventStream::from_array2(
            array![[0, 0, 1, 1], [0, 0, 2, 0], [1, 0, 3, 1]],
            2,
            1,
            0.001,
        );
        let frame = EventCount::default().generate(&stream).unwrap();

        let image = render_frame(&frame, Colormap::Grayscale, true);

        assert_eq!(image.width, 2);
        assert_eq!(image.height, 1);
        // Pixel (0,0) has the max count (2) → white; (1,0) has 1 → mid grey.
        assert_eq!(pixel(&image, 0, 0), [255, 255, 255]);
        assert_eq!(pixel(&image, 1, 0), [128, 128, 128]);
    }

    #[test]
    fn a_single_outlier_does_not_black_out_the_rest_of_the_field() {
        // A 12×12 count frame: 143 pixels at 10, one spurious outlier at 1000. With
        // max-normalisation the typical pixels would render near-black; the robust (p99) extent
        // keeps them bright.
        let plane = 12 * 12;
        let mut data = vec![10_u64; plane];
        data[0] = 1000; // the outlier
        let frame = EventFrame::from_parts(
            EventFrameData::U64(data),
            12,
            12,
            RepresentationKind::Count,
            vec!["count".to_owned()],
        );

        let image = render_frame(&frame, Colormap::Grayscale, true);

        // A typical (10) pixel maps to the top of the range, not the floor.
        assert!(
            pixel(&image, 5, 5)[0] > 200,
            "typical value must stay visible"
        );
    }

    #[test]
    fn flow_middlebury_encodes_direction_as_hue_and_zero_as_white() {
        // A 12×12 flow frame: left half flows +x, right half flows -x, plus one zero pixel.
        let plane = 12 * 12;
        let mut data = vec![0.0_f32; plane * 2];
        for y in 0..12 {
            for x in 0..12 {
                data[y * 12 + x] = if x < 6 { 1.0 } else { -1.0 }; // flow_x; flow_y stays 0
            }
        }
        data[0] = 0.0; // a zero-flow pixel at the top-left
        let frame = EventFrame::from_parts(
            EventFrameData::F32(data),
            12,
            12,
            RepresentationKind::Flow,
            vec!["flow_x".to_owned(), "flow_y".to_owned()],
        );

        let image = render_frame(&frame, Colormap::Viridis, true);

        // Opposite directions get different colours; zero flow renders white.
        assert_ne!(
            pixel(&image, 3, 5),
            pixel(&image, 9, 5),
            "opposite flow directions must differ in colour"
        );
        assert_eq!(pixel(&image, 0, 0), [255, 255, 255], "zero flow is white");
    }

    #[test]
    fn signed_reprs_use_the_diverging_red_blue_map() {
        // One positive event at (0,0), one negative at (1,0): the averaged time surface
        // is signed, so (0,0) reads red and (1,0) reads blue regardless of colormap arg.
        let stream = EventStream::from_array2(array![[0, 0, 10, 1], [1, 0, 10, 0]], 2, 1, 0.001);
        let frame = AveragedTimeSurface::default().generate(&stream).unwrap();

        let image = render_frame(&frame, Colormap::Viridis, true);

        let [r0, _, b0] = pixel(&image, 0, 0);
        let [r1, _, b1] = pixel(&image, 1, 0);
        assert!(r0 > b0, "positive pixel should be red-dominant");
        assert!(b1 > r1, "negative pixel should be blue-dominant");
    }

    #[test]
    fn tencode_passes_through_as_rgb() {
        let stream = EventStream::from_array2(array![[0, 0, 10, 1]], 1, 1, 0.001);
        let frame = Tencode::default().generate(&stream).unwrap();

        let image = render_frame(&frame, Colormap::Turbo, false);

        assert_eq!(image.pixels.len(), 3);
    }

    #[test]
    fn empty_frame_renders_uniformly_at_the_colormap_floor() {
        let stream = EventStream::from_array2(ndarray::Array2::zeros((0, 4)), 3, 2, 0.001);
        let frame = Binary.generate(&stream).unwrap();

        let image = render_frame(&frame, Colormap::Viridis, true);

        // Full-size, and every pixel is the colormap's zero anchor (Viridis floor).
        assert_eq!(image.pixels.len(), 3 * 2 * 3);
        assert_eq!(pixel(&image, 0, 0), [68, 1, 84]);
        assert!(image
            .pixels
            .chunks_exact(3)
            .all(|rgb| rgb == pixel(&image, 0, 0)));
    }
}
