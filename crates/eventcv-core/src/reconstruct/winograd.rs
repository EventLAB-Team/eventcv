//! 3x3 stride-1 convolution by Winograd's minimal filtering, F(4x4, 3x3), for the CPU.
//!
//! Each 4x4 output tile comes from one 6x6 input tile: transformed, multiplied element-wise by the
//! transformed kernel, and transformed back (Lavin & Gray, *Fast Algorithms for Convolutional Neural
//! Networks*, 2016). Summed over input channels, the element-wise products become 36 independent
//! matrix products, `[out, in] x [in, tiles]` — so a 3x3 convolution costs 36 multiplies per 16
//! outputs instead of 144. E2VID's ConvLSTM gates and residual blocks are all 3x3 stride-1 and were
//! most of a CPU frame.
//!
//! Tiles are processed in blocks, one block per task, so the transformed buffers stay a few
//! megabytes per thread whatever the sensor size.

use rayon::prelude::*;

use super::Output;

/// Input tile side (`m + r - 1` for `m = 4` outputs and an `r = 3` kernel).
const TILE: usize = 6;
/// Output tile side.
const OUT: usize = 4;
/// Points per transformed tile.
const POINTS: usize = TILE * TILE;
/// Output tiles per task: large enough that each of the 36 matrix products has some width, small
/// enough that a block's buffers stay in cache-sized pieces.
const BLOCK: usize = 64;

/// `B^T`, `G` and `A^T` for interpolation points 0, +-1, +-2 and infinity.
const BT: [[f32; TILE]; TILE] = [
    [4.0, 0.0, -5.0, 0.0, 1.0, 0.0],
    [0.0, -4.0, -4.0, 1.0, 1.0, 0.0],
    [0.0, 4.0, -4.0, -1.0, 1.0, 0.0],
    [0.0, -2.0, -1.0, 2.0, 1.0, 0.0],
    [0.0, 2.0, -1.0, -2.0, 1.0, 0.0],
    [0.0, 4.0, 0.0, -5.0, 0.0, 1.0],
];
const G: [[f32; 3]; TILE] = [
    [1.0 / 4.0, 0.0, 0.0],
    [-1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0],
    [-1.0 / 6.0, 1.0 / 6.0, -1.0 / 6.0],
    [1.0 / 24.0, 1.0 / 12.0, 1.0 / 6.0],
    [1.0 / 24.0, -1.0 / 12.0, 1.0 / 6.0],
    [0.0, 0.0, 1.0],
];
const AT: [[f32; TILE]; OUT] = [
    [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    [0.0, 1.0, -1.0, 2.0, -2.0, 0.0],
    [0.0, 1.0, 1.0, 4.0, 4.0, 0.0],
    [0.0, 1.0, -1.0, 8.0, -8.0, 1.0],
];

/// A 3x3 kernel bank, transformed once (`[36][out][in]`), and its bias.
pub(super) struct Winograd {
    kernel: Vec<f32>,
    bias: Vec<f32>,
    out_channels: usize,
    in_channels: usize,
}

impl Winograd {
    /// From a `[out, in, 3, 3]` weight, row-major, and its `[out]` bias.
    pub(super) fn new(
        weight: &[f32],
        bias: &[f32],
        out_channels: usize,
        in_channels: usize,
    ) -> Self {
        let mut kernel = vec![0.0; POINTS * out_channels * in_channels];
        for o in 0..out_channels {
            for i in 0..in_channels {
                let g = &weight[(o * in_channels + i) * 9..][..9];
                // G g G^T, 6x6.
                let mut gg = [[0.0_f32; 3]; TILE];
                for (r, row) in G.iter().enumerate() {
                    for c in 0..3 {
                        gg[r][c] = (0..3).map(|k| row[k] * g[k * 3 + c]).sum();
                    }
                }
                for r in 0..TILE {
                    for c in 0..TILE {
                        let value: f32 = (0..3).map(|k| gg[r][k] * G[c][k]).sum();
                        kernel[((r * TILE + c) * out_channels + o) * in_channels + i] = value;
                    }
                }
            }
        }
        Self {
            kernel,
            bias: bias.to_vec(),
            out_channels,
            in_channels,
        }
    }

    /// Convolves a `[in, height, width]` image with padding 1 and adds the bias, returning
    /// `[out, height, width]`.
    pub(super) fn forward(&self, x: &[f32], height: usize, width: usize) -> Vec<f32> {
        let (out_channels, in_channels) = (self.out_channels, self.in_channels);
        let (tiles_y, tiles_x) = (height.div_ceil(OUT), width.div_ceil(OUT));
        let tiles = tiles_y * tiles_x;
        let plane = height * width;

        let mut out = vec![0.0_f32; out_channels * plane];
        let target = Output(out.as_mut_ptr());
        (0..tiles.div_ceil(BLOCK))
            .into_par_iter()
            .for_each(|block| {
                let first = block * BLOCK;
                let count = BLOCK.min(tiles - first);
                // Within a block, too: at the network's deepest level a whole image is one block,
                // and a single task there was the slowest layer of the frame.
                //
                // Transformed input, [in][36][count], each channel's slab written by one task.
                let mut v = vec![0.0_f32; in_channels * POINTS * count];
                v.par_chunks_mut(POINTS * count)
                    .enumerate()
                    .for_each(|(c, slab)| {
                        let image = &x[c * plane..][..plane];
                        for t in 0..count {
                            let (ty, tx) = ((first + t) / tiles_x, (first + t) % tiles_x);
                            // The 6x6 input tile, rows/cols offset by the padding of 1.
                            let mut d = [[0.0_f32; TILE]; TILE];
                            for (r, row) in d.iter_mut().enumerate() {
                                let y = (ty * OUT + r) as isize - 1;
                                if y < 0 || y >= height as isize {
                                    continue;
                                }
                                for (col, cell) in row.iter_mut().enumerate() {
                                    let xx = (tx * OUT + col) as isize - 1;
                                    if xx >= 0 && xx < width as isize {
                                        *cell = image[y as usize * width + xx as usize];
                                    }
                                }
                            }
                            for (p, value) in transform_input(&d).iter().enumerate() {
                                slab[p * count + t] = *value;
                            }
                        }
                    });

                // The 36 products, one task each: m[p] = kernel[p] ([out, in]) x v[.., p, ..]
                // ([in, count], rows POINTS * count apart).
                let mut m = vec![0.0_f32; POINTS * out_channels * count];
                m.par_chunks_mut(out_channels * count)
                    .enumerate()
                    .for_each(|(p, product)| {
                        // SAFETY: `product` is the row-major [out, count] result, the kernel slice
                        // the row-major [out, in] matrix for point p, and `v` holds point p's
                        // [in, count] matrix at offset p * count with rows POINTS * count apart.
                        unsafe {
                            gemm::gemm(
                                out_channels,
                                count,
                                in_channels,
                                product.as_mut_ptr(),
                                1,
                                count as isize,
                                false,
                                self.kernel[p * out_channels * in_channels..].as_ptr(),
                                1,
                                in_channels as isize,
                                v[p * count..].as_ptr(),
                                1,
                                (POINTS * count) as isize,
                                0.0,
                                1.0,
                                false,
                                false,
                                false,
                                gemm::Parallelism::None,
                            );
                        }
                    });

                // Back to 4x4 tiles, straight into the image: one task per output channel.
                (0..out_channels).into_par_iter().for_each(|o| {
                    for t in 0..count {
                        let (ty, tx) = ((first + t) / tiles_x, (first + t) % tiles_x);
                        let mut tile = [0.0_f32; POINTS];
                        for (p, value) in tile.iter_mut().enumerate() {
                            *value = m[(p * out_channels + o) * count + t];
                        }
                        let (tile, bias) = (transform_output(&tile), self.bias[o]);
                        for r in 0..OUT.min(height - ty * OUT) {
                            let row = (o * height + ty * OUT + r) * width + tx * OUT;
                            for c in 0..OUT.min(width - tx * OUT) {
                                target.write(row + c, tile[r * OUT + c] + bias);
                            }
                        }
                    }
                });
            });
        out
    }
}

/// `B^T d B` for one 6x6 tile.
fn transform_input(d: &[[f32; TILE]; TILE]) -> [f32; POINTS] {
    let mut bd = [[0.0_f32; TILE]; TILE];
    for (r, row) in BT.iter().enumerate() {
        for c in 0..TILE {
            bd[r][c] = (0..TILE).map(|k| row[k] * d[k][c]).sum();
        }
    }
    let mut out = [0.0_f32; POINTS];
    for r in 0..TILE {
        for c in 0..TILE {
            out[r * TILE + c] = (0..TILE).map(|k| bd[r][k] * BT[c][k]).sum();
        }
    }
    out
}

/// `A^T m A` for one transformed tile, 6x6 to 4x4.
fn transform_output(m: &[f32; POINTS]) -> [f32; OUT * OUT] {
    let mut am = [[0.0_f32; TILE]; OUT];
    for (r, row) in AT.iter().enumerate() {
        for c in 0..TILE {
            am[r][c] = (0..TILE).map(|k| row[k] * m[k * TILE + c]).sum();
        }
    }
    let mut out = [0.0_f32; OUT * OUT];
    for r in 0..OUT {
        for c in 0..OUT {
            out[r * OUT + c] = (0..TILE).map(|k| am[r][k] * AT[c][k]).sum();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use burn::tensor::module::conv2d;
    use burn::tensor::ops::ConvOptions;
    use burn::tensor::{Device, Distribution, Tensor};

    use super::Winograd;

    #[test]
    fn matches_a_direct_convolution() {
        // Sizes that are not multiples of the 4x4 tile, so the edge tiles are exercised.
        let (out_channels, in_channels, height, width) = (5, 3, 11, 9);
        let device = Device::from(burn::backend::FlexDevice);
        let x = Tensor::<4>::random(
            [1, in_channels, height, width],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let w = Tensor::<4>::random(
            [out_channels, in_channels, 3, 3],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let b = Tensor::<1>::random([out_channels], Distribution::Normal(0.0, 1.0), &device);
        let expected = conv2d(
            x.clone(),
            w.clone(),
            Some(b.clone()),
            ConvOptions::new([1, 1], [1, 1], [1, 1], 1),
        );
        let expected = expected.into_data().try_to_vec::<f32>().unwrap();
        let x = x.into_data().try_to_vec::<f32>().unwrap();
        let w = w.into_data().try_to_vec::<f32>().unwrap();
        let b = b.into_data().try_to_vec::<f32>().unwrap();
        let got = Winograd::new(&w, &b, out_channels, in_channels).forward(&x, height, width);
        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(&expected) {
            assert!((g - e).abs() < 1e-4 * (1.0 + e.abs()), "{g} != {e}");
        }
    }
}
