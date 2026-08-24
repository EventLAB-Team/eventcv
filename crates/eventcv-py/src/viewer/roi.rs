//! Interactive ROI drawing — the state behind `draw_mask()`.
//!
//! The editor owns a mask (row-major `width·height`, `true` = keep) and turns pointer drags into
//! the shapes of [`eventcv_core::mask`]. It knows nothing about winit or wgpu: [`gpu`](super::gpu)
//! translates window events into these calls and paints the result into each displayed frame, so
//! the same editor drives the live camera view and a still frame from a file.
//!
//! The mask starts empty, so everything is dimmed until the first shape is drawn — what you see
//! undimmed is exactly what survives.

use eventcv_core::mask;
use eventcv_core::viz::Rgb8Image;

/// The key legend, shown in the window title (the viewer draws no text of its own).
pub(crate) const LEGEND: &str =
    "draw ROI - e/r/f ellipse|rect|freehand, drag keeps, shift+drag drops, \
     a all, c clear, z undo, Enter accept, Esc cancel";

/// Which shape the next drag draws.
#[derive(Clone, Copy, PartialEq)]
enum Tool {
    Ellipse,
    Rect,
    Freehand,
}

/// A drag in progress, in sensor coordinates.
struct Drag {
    anchor: (f64, f64),
    /// The cursor track, for the freehand tool (closed into a polygon on release).
    track: Vec<(f64, f64)>,
    /// Shift was held when the drag started: the shape is removed from the mask, not added.
    subtract: bool,
}

pub(crate) struct MaskEditor {
    width: usize,
    height: usize,
    mask: Vec<bool>,
    /// Masks as they were before each committed shape, for `z`.
    history: Vec<Vec<bool>>,
    tool: Tool,
    drag: Option<Drag>,
    cursor: (f64, f64),
    shift: bool,
    accepted: bool,
}

impl MaskEditor {
    pub(crate) fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            mask: vec![false; width * height],
            history: Vec::new(),
            tool: Tool::Ellipse,
            drag: None,
            cursor: (0.0, 0.0),
            shift: false,
            accepted: false,
        }
    }

    /// Moves the cursor. `(x, y)` are fractions of the window, which maps straight onto the sensor
    /// because the viewer stretches the frame across the whole window.
    pub(crate) fn cursor_moved(&mut self, x: f64, y: f64) {
        let cursor = (x * self.width as f64, y * self.height as f64);
        self.cursor = cursor;
        if let Some(drag) = &mut self.drag {
            // Record a freehand point only once the cursor reaches another pixel: the polygon fill
            // costs a pass per point, and pointer jitter would otherwise grow the track without
            // changing the shape it traces.
            let moved = drag.track.last().is_none_or(|&(px, py)| {
                (px - cursor.0).abs() >= 1.0 || (py - cursor.1).abs() >= 1.0
            });
            if moved {
                drag.track.push(cursor);
            }
        }
    }

    pub(crate) fn shift_held(&mut self, held: bool) {
        self.shift = held;
    }

    pub(crate) fn press(&mut self) {
        self.drag = Some(Drag {
            anchor: self.cursor,
            track: vec![self.cursor],
            subtract: self.shift,
        });
    }

    /// Ends the drag, folding its shape into the mask (or out of it, when shift was held).
    pub(crate) fn release(&mut self) {
        let Some(drag) = self.drag.take() else {
            return;
        };
        let shape = self.shape(&drag);
        self.history.push(self.mask.clone());
        for (keep, in_shape) in self.mask.iter_mut().zip(shape) {
            *keep = if drag.subtract {
                *keep && !in_shape
            } else {
                *keep || in_shape
            };
        }
    }

    /// Handles a key press. Returns whether the editor is finished: `Enter` accepts (the mask is
    /// available from [`into_mask`](Self::into_mask)), `Esc` is left to the viewer to cancel.
    pub(crate) fn key(&mut self, key: char) -> bool {
        match key {
            'e' => self.tool = Tool::Ellipse,
            'r' => self.tool = Tool::Rect,
            'f' => self.tool = Tool::Freehand,
            'a' => self.replace(vec![true; self.width * self.height]),
            'c' => self.replace(vec![false; self.width * self.height]),
            'z' => {
                if let Some(previous) = self.history.pop() {
                    self.mask = previous;
                }
            }
            '\r' => {
                self.accepted = true;
                return true;
            }
            _ => {}
        }
        false
    }

    /// The drawn mask, or `None` if the editor was closed without accepting.
    pub(crate) fn into_mask(self) -> Option<Vec<bool>> {
        self.accepted.then_some(self.mask)
    }

    /// Dims everything the mask drops, including the shape being dragged — so the bright region is
    /// always exactly what a `draw_mask()` would return right now.
    pub(crate) fn paint(&self, image: &mut Rgb8Image) {
        if image.width != self.width || image.height != self.height {
            return;
        }
        let preview = self.drag.as_ref().map(|drag| (self.shape(drag), drag.subtract));
        let (pixels, _) = image.pixels.as_chunks_mut::<3>();
        for (index, pixel) in pixels.iter_mut().enumerate() {
            let keep = match &preview {
                Some((shape, true)) => self.mask[index] && !shape[index],
                Some((shape, false)) => self.mask[index] || shape[index],
                None => self.mask[index],
            };
            if !keep {
                // Halve the brightness and cast it red, so an excluded region still shows its
                // events (you can see what you are cutting out) but never reads as kept.
                *pixel = [pixel[0] / 2 + 48, pixel[1] / 3, pixel[2] / 3];
            }
        }
    }

    fn replace(&mut self, mask: Vec<bool>) {
        self.history.push(std::mem::replace(&mut self.mask, mask));
    }

    /// Rasterises the drag: rect and ellipse are fitted to the box from the anchor to the cursor,
    /// freehand closes the cursor track into a polygon.
    fn shape(&self, drag: &Drag) -> Vec<bool> {
        let (x0, y0) = drag.anchor;
        let (x1, y1) = self.cursor;
        let (w, h) = ((x1 - x0).abs(), (y1 - y0).abs());
        match self.tool {
            Tool::Rect => mask::rect(self.width, self.height, x0.min(x1), y0.min(y1), w, h),
            Tool::Ellipse => mask::ellipse(
                self.width,
                self.height,
                (x0 + x1) / 2.0,
                (y0 + y1) / 2.0,
                w / 2.0,
                h / 2.0,
            ),
            Tool::Freehand => mask::polygon(self.width, self.height, &drag.track),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MaskEditor;
    use eventcv_core::viz::Rgb8Image;

    fn drag(editor: &mut MaskEditor, from: (f64, f64), to: (f64, f64)) {
        editor.cursor_moved(from.0, from.1);
        editor.press();
        editor.cursor_moved(to.0, to.1);
        editor.release();
    }

    fn kept(editor: &MaskEditor) -> usize {
        editor.mask.iter().filter(|&&keep| keep).count()
    }

    #[test]
    fn a_rect_drag_keeps_the_box_it_covers() {
        let mut editor = MaskEditor::new(100, 100);
        editor.key('r');
        drag(&mut editor, (0.25, 0.25), (0.75, 0.75)); // the middle half of each axis
        assert_eq!(kept(&editor), 50 * 50);
        assert!(editor.mask[50 * 100 + 50]);
        assert!(!editor.mask[0]);
    }

    #[test]
    fn shift_drag_subtracts_and_undo_puts_it_back() {
        let mut editor = MaskEditor::new(100, 100);
        editor.key('a'); // start from everything
        assert_eq!(kept(&editor), 100 * 100);

        editor.key('r');
        editor.shift_held(true);
        drag(&mut editor, (0.0, 0.0), (0.5, 0.5));
        assert_eq!(kept(&editor), 100 * 100 - 50 * 50);

        editor.key('z');
        assert_eq!(kept(&editor), 100 * 100);
    }

    #[test]
    fn a_freehand_track_closes_into_a_polygon() {
        // Trace three sides of the top-left quadrant; the fourth closes automatically.
        let mut editor = MaskEditor::new(100, 100);
        editor.key('f');
        editor.cursor_moved(0.0, 0.0);
        editor.press();
        for point in [(0.5, 0.0), (0.5, 0.5), (0.0, 0.5)] {
            editor.cursor_moved(point.0, point.1);
        }
        editor.release();
        assert_eq!(kept(&editor), 50 * 50);
        assert!(editor.mask[25 * 100 + 25] && !editor.mask[75 * 100 + 75]);
    }

    #[test]
    fn drags_accumulate_and_only_accepting_yields_a_mask() {
        let mut editor = MaskEditor::new(100, 100);
        editor.key('r');
        drag(&mut editor, (0.0, 0.0), (0.2, 1.0));
        drag(&mut editor, (0.8, 0.0), (1.0, 1.0));
        assert_eq!(kept(&editor), 2 * 20 * 100);

        // Closing without Enter discards the drawing; Enter hands the mask over.
        assert!(MaskEditor::new(4, 4).into_mask().is_none());
        assert!(editor.key('\r'));
        assert_eq!(editor.into_mask().unwrap().len(), 100 * 100);
    }

    #[test]
    fn painting_dims_only_what_the_mask_drops() {
        let mut editor = MaskEditor::new(4, 1);
        editor.key('r');
        drag(&mut editor, (0.0, 0.0), (0.5, 1.0)); // keep the left half
        let mut image = Rgb8Image {
            width: 4,
            height: 1,
            pixels: vec![200; 4 * 3],
        };
        editor.paint(&mut image);
        assert_eq!(&image.pixels[..6], &[200, 200, 200, 200, 200, 200]);
        assert_eq!(&image.pixels[6..], &[148, 66, 66, 148, 66, 66]);
    }
}
