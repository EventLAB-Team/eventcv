use eventcv_core::representation::{EventFrame, EventFrameData, RepresentationKind};
use minifb::{Key, Scale, Window, WindowOptions};
use plotters::backend::{BitMapBackend, DrawingBackend};

pub(crate) fn view(frame: &EventFrame, normalize: bool) -> Result<(), String> {
    if frame.kind() != RepresentationKind::Polarity {
        return Err("EventFrame does not have a viewer".to_owned());
    }

    let (_, height, width) = frame.shape();
    let pixels = match frame.data() {
        EventFrameData::U8(data) => render_values(data, width, height, true)?,
        EventFrameData::U16(data) => render_counts(data, width, height, normalize)?,
    };
    let mut window = Window::new(
        "eventcv - Polarity",
        width,
        height,
        WindowOptions {
            resize: false,
            scale: Scale::X1,
            ..WindowOptions::default()
        },
    )
    .map_err(|error| error.to_string())?;
    window.set_background_color(255, 255, 255);
    window.set_target_fps(60);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        window
            .update_with_buffer(&pixels, width, height)
            .map_err(|error| error.to_string())?;
    }

    Ok(())
}

fn render_counts(
    data: &[u16],
    width: usize,
    height: usize,
    normalize: bool,
) -> Result<Vec<u32>, String> {
    let values = if normalize {
        let maximum = data.iter().copied().max().unwrap_or(0);
        if maximum == 0 {
            vec![0; data.len()]
        } else {
            data.iter()
                .map(|&count| {
                    let scaled = u32::from(count) * u32::from(u8::MAX);
                    ((scaled + u32::from(maximum) / 2) / u32::from(maximum)) as u8
                })
                .collect()
        }
    } else {
        data.iter()
            .map(|&count| count.min(u16::from(u8::MAX)) as u8)
            .collect()
    };
    render_values(&values, width, height, normalize)
}

fn render_values(
    data: &[u8],
    width: usize,
    height: usize,
    enhance: bool,
) -> Result<Vec<u32>, String> {
    let plane_len = width
        .checked_mul(height)
        .ok_or_else(|| "frame dimensions are too large".to_owned())?;
    let data_len = plane_len
        .checked_mul(2)
        .ok_or_else(|| "frame dimensions are too large".to_owned())?;
    if data.len() != data_len {
        return Err("polarity frame must have two channels".to_owned());
    }

    let source_len = plane_len
        .checked_mul(3)
        .ok_or_else(|| "frame dimensions are too large".to_owned())?;
    let display_curve: [u8; 256] = std::array::from_fn(|value| {
        ((value as f32 / f32::from(u8::MAX)).sqrt() * f32::from(u8::MAX)).round() as u8
    });
    let mut source = vec![255; source_len];
    for index in 0..plane_len {
        let positive = if enhance {
            display_curve[usize::from(data[index])]
        } else {
            data[index]
        };
        let negative = if enhance {
            display_curve[usize::from(data[plane_len + index])]
        } else {
            data[plane_len + index]
        };
        let intensity = positive.max(negative);
        source[index * 3] = u8::MAX - intensity + positive;
        source[index * 3 + 1] = u8::MAX - intensity;
        source[index * 3 + 2] = u8::MAX - intensity + negative;
    }

    let dimensions = (
        u32::try_from(width).map_err(|error| error.to_string())?,
        u32::try_from(height).map_err(|error| error.to_string())?,
    );
    let mut rendered = vec![0; source.len()];
    {
        let mut backend = BitMapBackend::with_buffer(&mut rendered, dimensions);
        backend
            .blit_bitmap((0, 0), dimensions, &source)
            .map_err(|error| error.to_string())?;
        backend.present().map_err(|error| error.to_string())?;
    }

    Ok(rendered
        .chunks_exact(3)
        .map(|pixel| {
            (u32::from(pixel[0]) << 16) | (u32::from(pixel[1]) << 8) | u32::from(pixel[2])
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use std::{hint::black_box, time::Instant};

    use super::render_counts;

    #[test]
    fn renders_polarities_with_a_shared_linear_scale() {
        let pixels = render_counts(&[2, 1, 0, 2], 2, 1, true).unwrap();

        assert_eq!(pixels, [0xff0000, 0xb500ff]);
    }

    #[test]
    fn renders_an_empty_frame_as_white() {
        let pixels = render_counts(&[0; 4], 2, 1, true).unwrap();

        assert_eq!(pixels, [0xffffff, 0xffffff]);
    }

    #[test]
    fn fades_lower_counts_towards_white() {
        let pixels = render_counts(&[2, 1, 0, 0], 2, 1, true).unwrap();

        assert_eq!(pixels, [0xff0000, 0xff4a4a]);
    }

    #[test]
    fn clips_raw_counts_into_uint8_space() {
        let pixels = render_counts(&[300, 0, 0, 128], 2, 1, false).unwrap();

        assert_eq!(pixels, [0xff0000, 0x7f7fff]);
    }

    #[test]
    fn rejects_an_invalid_polarity_frame() {
        let error = render_counts(&[0; 3], 2, 1, true).unwrap_err();

        assert_eq!(error, "polarity frame must have two channels");
    }

    #[test]
    #[ignore = "manual performance benchmark"]
    fn benchmark_polarity_colorization() {
        let data = vec![1_u16; 2 * 640 * 480];
        let iterations = 20;
        let start = Instant::now();

        for _ in 0..iterations {
            black_box(render_counts(black_box(&data), 640, 480, true).unwrap());
        }

        eprintln!(
            "polarity colorization: {:?} per frame",
            start.elapsed() / iterations
        );
    }
}
