use eventcv_core::representation::{
    EventFrame, EventFrameData, EventPointSet, RepresentationKind,
};
use minifb::{Key, MouseButton, MouseMode, Scale, Window, WindowOptions};

const VIEW_WIDTH: usize = 960;
const VIEW_HEIGHT: usize = 720;
const BACKGROUND: u32 = 0x0b1020;
const POSITIVE: u32 = 0xff496c;
const NEGATIVE: u32 = 0x27c2ff;
const AXIS_X: u32 = 0xffca3a;
const AXIS_Y: u32 = 0x8ac926;
const AXIS_Z: u32 = 0xc77dff;
const DEFAULT_ANGLE: f32 = -0.55;
const RESET_BOUNDS: (usize, usize, usize, usize) = (16, 16, 96, 48);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DragAxis {
    Horizontal,
    Vertical,
}

pub(crate) fn view(frame: &EventFrame, normalize: bool) -> Result<(), String> {
    match prepare_frame(frame, normalize)? {
        PreparedView::Image {
            pixels,
            width,
            height,
            name,
        } => show_image(&pixels, width, height, name),
        PreparedView::Cloud {
            points,
            z_label,
            name,
        } => show_cloud(&points, z_label, name),
    }
}

pub(crate) fn view_point_set(points: &EventPointSet) -> Result<(), String> {
    show_cloud(&point_set_cloud(points), "TIME", "Point Set")
}

enum PreparedView {
    Image {
        pixels: Vec<u32>,
        width: usize,
        height: usize,
        name: &'static str,
    },
    Cloud {
        points: Vec<CloudPoint>,
        z_label: &'static str,
        name: &'static str,
    },
}

fn prepare_frame(frame: &EventFrame, normalize: bool) -> Result<PreparedView, String> {
    let (_, height, width) = frame.shape();
    match frame.kind() {
        RepresentationKind::Polarity => Ok(PreparedView::Image {
            pixels: render_polarity(frame.data(), width, height, normalize)?,
            width,
            height,
            name: "Polarity",
        }),
        RepresentationKind::Binary => Ok(PreparedView::Image {
            pixels: render_binary(frame.data(), width, height)?,
            width,
            height,
            name: "Binary",
        }),
        RepresentationKind::Tencode => Ok(PreparedView::Image {
            pixels: render_tencode(frame.data(), width, height, normalize)?,
            width,
            height,
            name: "Tencode",
        }),
        RepresentationKind::Voxel => Ok(PreparedView::Cloud {
            points: voxel_cloud(frame)?,
            z_label: "TIME BIN",
            name: "Voxel Grid",
        }),
        RepresentationKind::TimeSurface => Ok(PreparedView::Cloud {
            points: time_surface_cloud(frame)?,
            z_label: "RESPONSE",
            name: "Time Surface",
        }),
        RepresentationKind::Mcts => Ok(PreparedView::Cloud {
            points: mcts_cloud(frame)?,
            z_label: "WINDOW MS",
            name: "MCTS",
        }),
    }
}

fn show_image(
    pixels: &[u32],
    width: usize,
    height: usize,
    name: &str,
) -> Result<(), String> {
    let mut window = Window::new(
        &format!("eventcv - {name}"),
        width,
        height,
        WindowOptions {
            resize: false,
            scale: Scale::X1,
            ..WindowOptions::default()
        },
    )
    .map_err(|error| error.to_string())?;
    window.set_target_fps(60);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        window
            .update_with_buffer(pixels, width, height)
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn show_cloud(points: &[CloudPoint], z_label: &str, name: &str) -> Result<(), String> {
    let mut window = Window::new(
        &format!("eventcv - {name}"),
        VIEW_WIDTH,
        VIEW_HEIGHT,
        WindowOptions {
            resize: false,
            ..WindowOptions::default()
        },
    )
    .map_err(|error| error.to_string())?;

    window.set_target_fps(60);

    let mut pitch = DEFAULT_ANGLE;
    let mut yaw = 0.0_f32;
    let mut last_x = 0.0_f32;
    let mut last_y = 0.0_f32;

    let mut dragging = false;
    let mut drag_axis = None;
    let mut previous_down = false;

    let mut buffer = render_cloud(points, z_label, VIEW_WIDTH, VIEW_HEIGHT, pitch, yaw);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let mouse = window.get_mouse_pos(MouseMode::Discard);
        let down = window.get_mouse_down(MouseButton::Left);
        let mut redraw = false;

        if down && !previous_down {
            if mouse.is_some_and(|(x, y)| reset_contains(x, y)) {
                pitch = DEFAULT_ANGLE;
                yaw = 0.0;
                redraw = true;
            } else if let Some((x, y)) = mouse {
                dragging = true;
                last_x = x;
                last_y = y;
                drag_axis = None;
            }
        }

        if down && dragging {
            if let Some((x, y)) = mouse {
                let delta_x = x - last_x;
                let delta_y = y - last_y;
                drag_axis = drag_axis.or_else(|| dominant_axis(delta_x, delta_y));
                match drag_axis {
                    Some(DragAxis::Horizontal) if delta_x != 0.0 => {
                        yaw += delta_x * 0.008;
                        redraw = true;
                    }
                    Some(DragAxis::Vertical) if delta_y != 0.0 => {
                        let next_pitch = (pitch + delta_y * 0.008).clamp(-1.45, 1.45);
                        redraw = next_pitch != pitch;
                        pitch = next_pitch;
                    }
                    _ => {}
                }
                last_x = x;
                last_y = y;
            }
        } else if !down {
            dragging = false;
            drag_axis = None;
        }

        if redraw {
            buffer = render_cloud(points, z_label, VIEW_WIDTH, VIEW_HEIGHT, pitch, yaw);
        }

        window
            .update_with_buffer(&buffer, VIEW_WIDTH, VIEW_HEIGHT)
            .map_err(|error| error.to_string())?;

        previous_down = down;
    }

    Ok(())
}

fn dominant_axis(delta_x: f32, delta_y: f32) -> Option<DragAxis> {
    if delta_x == 0.0 && delta_y == 0.0 {
        None
    } else if delta_x.abs() >= delta_y.abs() {
        Some(DragAxis::Horizontal)
    } else {
        Some(DragAxis::Vertical)
    }
}

fn render_polarity(
    data: &EventFrameData,
    width: usize,
    height: usize,
    normalize: bool,
) -> Result<Vec<u32>, String> {
    match data {
        EventFrameData::U8(values) => render_polarity_values(values, width, height, true),
        EventFrameData::U16(values) => {
            let values = scale_counts(values, normalize);
            render_polarity_values(&values, width, height, normalize)
        }
        EventFrameData::U64(values) => {
            let values = scale_counts(values, normalize);
            render_polarity_values(&values, width, height, normalize)
        }
        EventFrameData::F32(_) => Err("polarity frame cannot use float32 data".to_owned()),
    }
}

fn scale_counts<T>(data: &[T], normalize: bool) -> Vec<u8>
where
    T: Copy + Into<u64>,
{
    if !normalize {
        return data
            .iter()
            .map(|&value| value.into().min(u64::from(u8::MAX)) as u8)
            .collect();
    }
    let maximum = data.iter().copied().map(Into::into).max().unwrap_or(0);
    if maximum == 0 {
        return vec![0; data.len()];
    }
    data.iter()
        .map(|&value| {
            let value = u128::from(value.into());
            ((value * u128::from(u8::MAX) + u128::from(maximum) / 2)
                / u128::from(maximum)) as u8
        })
        .collect()
}

fn render_polarity_values(
    data: &[u8],
    width: usize,
    height: usize,
    enhance: bool,
) -> Result<Vec<u32>, String> {
    let plane_len = checked_plane_len(width, height)?;
    if data.len() != plane_len * 2 {
        return Err("polarity frame must have two channels".to_owned());
    }
    let curve: [u8; 256] = std::array::from_fn(|value| {
        ((value as f32 / f32::from(u8::MAX)).sqrt() * f32::from(u8::MAX)).round() as u8
    });
    Ok((0..plane_len)
        .map(|index| {
            let positive = if enhance {
                curve[usize::from(data[index])]
            } else {
                data[index]
            };
            let negative = if enhance {
                curve[usize::from(data[plane_len + index])]
            } else {
                data[plane_len + index]
            };
            let intensity = positive.max(negative);
            rgb(
                u8::MAX - intensity + positive,
                u8::MAX - intensity,
                u8::MAX - intensity + negative,
            )
        })
        .collect())
}

fn render_binary(
    data: &EventFrameData,
    width: usize,
    height: usize,
) -> Result<Vec<u32>, String> {
    let plane_len = checked_plane_len(width, height)?;
    if frame_data_len(data) != plane_len {
        return Err("binary frame must have one channel".to_owned());
    }
    Ok((0..plane_len)
        .map(|index| {
            if frame_value(data, index).unwrap_or(0.0) > 0.0 {
                0x20d9c5
            } else {
                0x071018
            }
        })
        .collect())
}

fn render_tencode(
    data: &EventFrameData,
    width: usize,
    height: usize,
    normalize: bool,
) -> Result<Vec<u32>, String> {
    let plane_len = checked_plane_len(width, height)?;
    if frame_data_len(data) != plane_len * 3 {
        return Err("tencode frame must have three channels".to_owned());
    }
    let maximum = if normalize {
        (0..frame_data_len(data))
            .map(|index| frame_value(data, index).unwrap_or(0.0))
            .fold(0.0_f64, f64::max)
    } else {
        255.0
    };
    let scale = if maximum > 255.0 { 255.0 / maximum } else { 1.0 };

    Ok((0..plane_len)
        .map(|index| {
            let positive = (frame_value(data, index).unwrap_or(0.0) * scale)
                .round()
                .clamp(0.0, 255.0) as u8;
            let age = (frame_value(data, plane_len + index).unwrap_or(0.0) * scale)
                .round()
                .clamp(0.0, 255.0) as u8;
            let negative = (frame_value(data, 2 * plane_len + index).unwrap_or(0.0) * scale)
                .round()
                .clamp(0.0, 255.0) as u8;
            rgb(positive, age, negative)
        })
        .collect())
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
                    if channel < windows { NEGATIVE } else { POSITIVE },
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

fn render_cloud(
    points: &[CloudPoint],
    z_label: &str,
    width: usize,
    height: usize,
    pitch: f32,
    yaw: f32,
) -> Vec<u32> {
    let mut buffer = vec![BACKGROUND; width * height];
    let mut depth_buffer = vec![f32::INFINITY; width * height];
    let maximum = points
        .iter()
        .map(|point| point.strength)
        .fold(0.0_f32, f32::max);

    for point in points {
        if let Some(projected) = project(*point, pitch, yaw, width, height) {
            let relative = if maximum > 0.0 {
                (point.strength / maximum).sqrt()
            } else {
                0.0
            };
            let radius = 1 + relative.round() as i32;
            let color = scale_color(point.color, 0.45 + relative * 0.55);
            draw_depth_point(
                &mut buffer,
                &mut depth_buffer,
                width,
                height,
                projected,
                radius,
                color,
            );
        }
    }

    draw_axes(&mut buffer, width, height, pitch, yaw, z_label);
    draw_reset(&mut buffer, width, height);
    draw_legend(&mut buffer, width, height);
    buffer
}

fn project(
    point: CloudPoint,
    pitch: f32,
    yaw: f32,
    width: usize,
    height: usize,
) -> Option<ProjectedPoint> {
    let (x, y, z) = rotate_y(point.x as f32, point.y as f32, point.z as f32, yaw);
    let (x, y, z) = rotate_x(x, y, z, pitch);
    let depth = 3.4 - z;
    if depth <= 0.1 {
        return None;
    }
    let scale = width.min(height) as f32 * 0.75 / depth;
    let screen_x = width as f32 * 0.5 + x * scale;
    let screen_y = height as f32 * 0.52 - y * scale;
    if screen_x < 0.0 || screen_x >= width as f32 || screen_y < 0.0 || screen_y >= height as f32 {
        return None;
    }
    Some(ProjectedPoint {
        x: screen_x.round() as i32,
        y: screen_y.round() as i32,
        depth,
    })
}

fn rotate_x(x: f32, y: f32, z: f32, angle: f32) -> (f32, f32, f32) {
    let cosine = angle.cos();
    let sine = angle.sin();
    (x, y * cosine - z * sine, y * sine + z * cosine)
}

fn rotate_y(x: f32, y: f32, z: f32, angle: f32) -> (f32, f32, f32) {
    let cosine = angle.cos();
    let sine = angle.sin();
    (x * cosine + z * sine, y, -x * sine + z * cosine)
}

fn draw_depth_point(
    buffer: &mut [u32],
    depth_buffer: &mut [f32],
    width: usize,
    height: usize,
    point: ProjectedPoint,
    radius: i32,
    color: u32,
) {
    for y in point.y - radius..=point.y + radius {
        for x in point.x - radius..=point.x + radius {
            if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
                continue;
            }
            let index = y as usize * width + x as usize;
            if point.depth < depth_buffer[index] {
                depth_buffer[index] = point.depth;
                buffer[index] = color;
            }
        }
    }
}

fn draw_axes(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    pitch: f32,
    yaw: f32,
    z_label: &str,
) {
    let origin = axis_project(-1.0, -1.0, -1.0, pitch, yaw, width, height);
    let x_end = axis_project(1.0, -1.0, -1.0, pitch, yaw, width, height);
    let y_end = axis_project(-1.0, 1.0, -1.0, pitch, yaw, width, height);
    let z_end = axis_project(-1.0, -1.0, 1.0, pitch, yaw, width, height);
    if let (Some(origin), Some(x_end)) = (origin, x_end) {
        draw_line(buffer, width, height, origin, x_end, AXIS_X);
        draw_label_at(buffer, width, height, x_end, "X", AXIS_X);
    }
    if let (Some(origin), Some(y_end)) = (origin, y_end) {
        draw_line(buffer, width, height, origin, y_end, AXIS_Y);
        draw_label_at(
            buffer,
            width,
            height,
            (y_end.0 - 18, y_end.1 - 10),
            "Y",
            AXIS_Y,
        );
    }
    if let (Some(origin), Some(z_end)) = (origin, z_end) {
        draw_line(buffer, width, height, origin, z_end, AXIS_Z);
        draw_label_at(
            buffer,
            width,
            height,
            (z_end.0 - (z_label.len() * 12 + 14) as i32, z_end.1),
            z_label,
            AXIS_Z,
        );
    }
}

fn axis_project(
    x: f64,
    y: f64,
    z: f64,
    pitch: f32,
    yaw: f32,
    width: usize,
    height: usize,
) -> Option<(i32, i32)> {
    project(
        CloudPoint {
            x,
            y,
            z,
            strength: 1.0,
            color: 0,
        },
        pitch,
        yaw,
        width,
        height,
    )
    .map(|point| (point.x, point.y))
}

fn draw_reset(buffer: &mut [u32], width: usize, height: usize) {
    let (left, top, right, bottom) = RESET_BOUNDS;
    fill_rect(buffer, width, height, left, top, right, bottom, 0x26324d);
    stroke_rect(buffer, width, height, left, top, right, bottom, 0xd7e3ff);
    draw_text(buffer, width, height, left + 12, top + 9, "RESET", 0xffffff, 2);
}

fn draw_legend(buffer: &mut [u32], width: usize, height: usize) {
    let x = width.saturating_sub(132);
    fill_rect(buffer, width, height, x, 17, x + 10, 27, POSITIVE);
    draw_text(buffer, width, height, x + 16, 16, "POSITIVE", 0xe8edf7, 1);
    fill_rect(buffer, width, height, x, 35, x + 10, 45, NEGATIVE);
    draw_text(buffer, width, height, x + 16, 34, "NEGATIVE", 0xe8edf7, 1);
}

fn reset_contains(x: f32, y: f32) -> bool {
    let (left, top, right, bottom) = RESET_BOUNDS;
    x >= left as f32 && x <= right as f32 && y >= top as f32 && y <= bottom as f32
}

fn draw_label_at(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    point: (i32, i32),
    label: &str,
    color: u32,
) {
    let label_width = label.len() * 12;
    let x = (point.0 + 7).clamp(2, width.saturating_sub(label_width + 2) as i32) as usize;
    let y = (point.1 - 8).clamp(2, height.saturating_sub(16) as i32) as usize;
    draw_text(buffer, width, height, x, y, label, color, 2);
}

fn draw_line(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    start: (i32, i32),
    end: (i32, i32),
    color: u32,
) {
    let (mut x, mut y) = start;
    let dx = (end.0 - x).abs();
    let sx = if x < end.0 { 1 } else { -1 };
    let dy = -(end.1 - y).abs();
    let sy = if y < end.1 { 1 } else { -1 };
    let mut error = dx + dy;
    loop {
        set_pixel(buffer, width, height, x, y, color);
        if x == end.0 && y == end.1 {
            break;
        }
        let doubled = 2 * error;
        if doubled >= dy {
            error += dy;
            x += sx;
        }
        if doubled <= dx {
            error += dx;
            y += sy;
        }
    }
}

fn fill_rect(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    left: usize,
    top: usize,
    right: usize,
    bottom: usize,
    color: u32,
) {
    for y in top..bottom.min(height) {
        for x in left..right.min(width) {
            buffer[y * width + x] = color;
        }
    }
}

fn stroke_rect(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    left: usize,
    top: usize,
    right: usize,
    bottom: usize,
    color: u32,
) {
    draw_line(
        buffer,
        width,
        height,
        (left as i32, top as i32),
        (right as i32, top as i32),
        color,
    );
    draw_line(
        buffer,
        width,
        height,
        (right as i32, top as i32),
        (right as i32, bottom as i32),
        color,
    );
    draw_line(
        buffer,
        width,
        height,
        (right as i32, bottom as i32),
        (left as i32, bottom as i32),
        color,
    );
    draw_line(
        buffer,
        width,
        height,
        (left as i32, bottom as i32),
        (left as i32, top as i32),
        color,
    );
}

fn draw_text(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    text: &str,
    color: u32,
    scale: usize,
) {
    for (character_index, character) in text.chars().enumerate() {
        let glyph = glyph(character.to_ascii_uppercase());
        for (row, bits) in glyph.into_iter().enumerate() {
            for column in 0..5 {
                if bits & (1 << (4 - column)) == 0 {
                    continue;
                }
                fill_rect(
                    buffer,
                    width,
                    height,
                    x + character_index * 6 * scale + column * scale,
                    y + row * scale,
                    x + character_index * 6 * scale + (column + 1) * scale,
                    y + (row + 1) * scale,
                    color,
                );
            }
        }
    }
}

fn glyph(character: char) -> [u8; 7] {
    match character {
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
        'I' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        ' ' => [0; 7],
        _ => [0b11111, 0b10001, 0b00010, 0b00100, 0b00100, 0b00000, 0b00100],
    }
}

fn set_pixel(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    color: u32,
) {
    if x >= 0 && y >= 0 && x < width as i32 && y < height as i32 {
        buffer[y as usize * width + x as usize] = color;
    }
}

fn scale_color(color: u32, brightness: f32) -> u32 {
    let red = (((color >> 16) & 0xff) as f32 * brightness).round() as u8;
    let green = (((color >> 8) & 0xff) as f32 * brightness).round() as u8;
    let blue = ((color & 0xff) as f32 * brightness).round() as u8;
    rgb(red, green, blue)
}

fn rgb(red: u8, green: u8, blue: u8) -> u32 {
    (u32::from(red) << 16) | (u32::from(green) << 8) | u32::from(blue)
}

fn frame_value(data: &EventFrameData, index: usize) -> Option<f64> {
    match data {
        EventFrameData::U8(values) => values.get(index).map(|&value| f64::from(value)),
        EventFrameData::U16(values) => values.get(index).map(|&value| f64::from(value)),
        EventFrameData::U64(values) => values.get(index).map(|&value| value as f64),
        EventFrameData::F32(values) => values.get(index).map(|&value| f64::from(value)),
    }
}

fn float_data(frame: &EventFrame) -> Result<&[f32], String> {
    let EventFrameData::F32(values) = frame.data() else {
        return Err(format!("{} frame must use float32 data", frame.kind().as_str()));
    };
    Ok(values)
}

fn checked_plane_len(width: usize, height: usize) -> Result<usize, String> {
    width
        .checked_mul(height)
        .ok_or_else(|| "frame dimensions are too large".to_owned())
}

fn frame_data_len(data: &EventFrameData) -> usize {
    match data {
        EventFrameData::U8(values) => values.len(),
        EventFrameData::U16(values) => values.len(),
        EventFrameData::U64(values) => values.len(),
        EventFrameData::F32(values) => values.len(),
    }
}

#[derive(Clone, Copy, Debug)]
struct CloudPoint {
    x: f64,
    y: f64,
    z: f64,
    strength: f32,
    color: u32,
}

#[derive(Clone, Copy, Debug)]
struct ProjectedPoint {
    x: i32,
    y: i32,
    depth: f32,
}

#[cfg(test)]
mod tests {
    use eventcv_core::representation::{
        Binary, Mcts, PointSet, Polarity, Representation, Tencode, TimeSurface, VoxelGrid,
    };

    use super::{
        dominant_axis, point_set_cloud, prepare_frame, project, render_cloud,
        render_polarity_values, render_tencode, reset_contains, rgb, rotate_x, rotate_y,
        scale_counts, voxel_cloud, CloudPoint, DragAxis, PreparedView, DEFAULT_ANGLE, VIEW_HEIGHT,
        VIEW_WIDTH,
    };

    #[test]
    fn preserves_the_original_polarity_colors() {
        let pixels = render_polarity_values(&[255, 128, 0, 255], 2, 1, true).unwrap();

        assert_eq!(pixels, [0xff0000, 0xb500ff]);
    }

    #[test]
    fn scales_large_counts_without_overflow() {
        assert_eq!(scale_counts(&[u64::MAX, u64::MAX / 2], true), [255, 127]);
    }

    #[test]
    fn rotation_is_strictly_about_x() {
        let rotated = rotate_x(0.75, -0.25, 0.5, DEFAULT_ANGLE);

        assert_eq!(rotated.0, 0.75);
        assert_ne!(rotated.1, -0.25);
        assert_ne!(rotated.2, 0.5);
    }

    #[test]
    fn rotation_is_strictly_about_y() {
        let rotated = rotate_y(0.75, -0.25, 0.5, DEFAULT_ANGLE);

        assert_ne!(rotated.0, 0.75);
        assert_eq!(rotated.1, -0.25);
        assert_ne!(rotated.2, 0.5);
    }

    #[test]
    fn locks_viewpoint_movement_to_the_dominant_drag_axis() {
        assert_eq!(dominant_axis(8.0, 3.0), Some(DragAxis::Horizontal));
        assert_eq!(dominant_axis(2.0, -9.0), Some(DragAxis::Vertical));
        assert_eq!(dominant_axis(0.0, 0.0), None);
    }

    #[test]
    fn viewpoint_changes_keep_the_volume_origin_fixed() {
        let point = CloudPoint {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            strength: 1.0,
            color: 0,
        };
        let original = project(point, DEFAULT_ANGLE, 0.0, VIEW_WIDTH, VIEW_HEIGHT).unwrap();
        let moved = project(point, 0.75, -0.8, VIEW_WIDTH, VIEW_HEIGHT).unwrap();

        assert_eq!(moved.x, original.x);
        assert_eq!(moved.y, original.y);
        assert_eq!(moved.depth, original.depth);
    }

    #[test]
    fn reset_button_has_a_bounded_hit_area() {
        assert!(reset_contains(20.0, 20.0));
        assert!(!reset_contains(120.0, 20.0));
    }

    #[test]
    fn packs_rgb_channels_in_display_order() {
        assert_eq!(rgb(255, 128, 64), 0xff8040);
    }

    #[test]
    fn renders_tencode_as_a_three_channel_image() {
        let data = eventcv_core::representation::EventFrameData::U8(vec![
            255, 0, 170, 0, 0, 255,
        ]);

        let pixels = render_tencode(&data, 2, 1, true).unwrap();

        assert_eq!(pixels, [0xffaa00, 0x0000ff]);
    }

    #[test]
    fn prepares_every_representation_without_opening_a_window() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/test/example.npz");
        let stream = eventcv_core::load(path).unwrap();

        for frame in [
            Binary.generate(&stream).unwrap(),
            Polarity::default().generate(&stream).unwrap(),
            Tencode::default().generate(&stream).unwrap(),
        ] {
            assert!(matches!(
                prepare_frame(&frame, true).unwrap(),
                PreparedView::Image { .. }
            ));
        }
        for frame in [
            VoxelGrid::default().generate(&stream).unwrap(),
            TimeSurface::default().generate(&stream).unwrap(),
            Mcts::default().generate(&stream).unwrap(),
        ] {
            assert!(matches!(
                prepare_frame(&frame, true).unwrap(),
                PreparedView::Cloud { .. }
            ));
        }
        assert!(!point_set_cloud(&PointSet.generate(&stream).unwrap()).is_empty());
    }

    #[test]
    #[ignore = "manual performance benchmark"]
    fn benchmark_cpu_cloud_rendering() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/test/example.npz");
        let stream = eventcv_core::load(path).unwrap();
        let frame = VoxelGrid::default().generate(&stream).unwrap();
        let points = voxel_cloud(&frame).unwrap();
        let start = std::time::Instant::now();

        let pixels = render_cloud(
            &points,
            "TIME BIN",
            VIEW_WIDTH,
            VIEW_HEIGHT,
            DEFAULT_ANGLE,
            0.0,
        );

        assert_eq!(pixels.len(), VIEW_WIDTH * VIEW_HEIGHT);
        eprintln!(
            "CPU cloud rendering: {} points in {:?}",
            points.len(),
            start.elapsed()
        );
    }
}
