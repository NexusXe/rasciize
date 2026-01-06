use std::{f32::consts::PI, simd::prelude::*};

use super::*;
use std::arch::x86_64::*;

#[derive(Debug)]
struct PlanarBuffer {
    width: u32,
    height: u32,
    red: Vec<__m512>,   // Contiguous Reds
    green: Vec<__m512>, // Contiguous Greens
    blue: Vec<__m512>,  // Contiguous Blues
}

// this intrinsic doesn't exist, so i'm implementing it myself
// converts 64 bytes to 64 single-precision floats
fn _mm512_cvtepu8_ps(input: __m512i) -> [__m512; 4] {
    // convert each byte to a 32-bit unsigned integer accross 4 different 512-bit destination vectors

    // split __m512i into four __m128i
    let byte_lanes: [__m128i; 4] = unsafe {
        [
            _mm512_castsi512_si128(input),
            _mm512_extracti32x4_epi32(input, 1),
            _mm512_extracti32x4_epi32(input, 2),
            _mm512_extracti32x4_epi32(input, 3),
        ]
    };

    unsafe {
        [
            _mm512_cvtepu32_ps(_mm512_cvtepu8_epi32(byte_lanes[0])),
            _mm512_cvtepu32_ps(_mm512_cvtepu8_epi32(byte_lanes[1])),
            _mm512_cvtepu32_ps(_mm512_cvtepu8_epi32(byte_lanes[2])),
            _mm512_cvtepu32_ps(_mm512_cvtepu8_epi32(byte_lanes[3])),
        ]
    }
}

impl PlanarBuffer {
    fn rgb8_to_planarbuffer(rgb8: &DynamicImage) -> Result<Self, Box<dyn std::error::Error>> {
        let frame = rgb8.to_rgb8();
        // consume all reds, greens, and blues individually, 64 at a time
        // get 64 pixels

        let mut all_reds: Vec<__m512> = Vec::with_capacity(frame.pixels().len().div_ceil(64));
        let mut all_greens: Vec<__m512> = Vec::with_capacity(frame.pixels().len().div_ceil(64));
        let mut all_blues: Vec<__m512> = Vec::with_capacity(frame.pixels().len().div_ceil(64));

        let mut pixels = frame.pixels();

        let mut reds = [0u8; 64];
        let mut greens = [0u8; 64];
        let mut blues = [0u8; 64];

        for _ in 0..(pixels.len().div_floor(64)) {
            while let Ok(chunk) = pixels.next_chunk::<64>() {
                chunk.iter().enumerate().for_each(|(idx, p)| {
                    reds[idx] = p[0];
                    greens[idx] = p[1];
                    blues[idx] = p[2];
                });
                let reds_vec: __m512i = __m512i::from(u8x64::from_array(reds));
                let greens_vec: __m512i = __m512i::from(u8x64::from_array(greens));
                let blues_vec: __m512i = __m512i::from(u8x64::from_array(blues));

                all_reds.extend_from_slice(&_mm512_cvtepu8_ps(reds_vec));
                all_greens.extend_from_slice(&_mm512_cvtepu8_ps(greens_vec));
                all_blues.extend_from_slice(&_mm512_cvtepu8_ps(blues_vec));
            }
        }

        let last_chunk = pixels.next_chunk::<64>();

        debug_assert!(last_chunk.is_err());

        // reinitialize the arrays since this chunk is not full
        reds = [0u8; 64];
        greens = [0u8; 64];
        blues = [0u8; 64];

        last_chunk.unwrap_err().enumerate().for_each(|(idx, p)| {
            reds[idx] = p[0];
            greens[idx] = p[1];
            blues[idx] = p[2];
        });
        let reds_vec: __m512i = __m512i::from(u8x64::from_array(reds));
        let greens_vec: __m512i = __m512i::from(u8x64::from_array(greens));
        let blues_vec: __m512i = __m512i::from(u8x64::from_array(blues));

        all_reds.extend_from_slice(&_mm512_cvtepu8_ps(reds_vec));
        all_greens.extend_from_slice(&_mm512_cvtepu8_ps(greens_vec));
        all_blues.extend_from_slice(&_mm512_cvtepu8_ps(blues_vec));

        Ok(Self {
            width: frame.width(),
            height: frame.height(),
            red: all_reds,
            green: all_greens,
            blue: all_blues,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ResizeWeight {
    /// The index of the first source row/col to read
    pub start_index: usize,
    /// The list of normalized weights (length depends on the scale)
    pub values: Vec<f32>,
}

pub type FilterBank = Vec<ResizeWeight>;

fn sinc(x: f32) -> f32 {
    if x.abs() < f32::EPSILON {
        return 1.0;
    }
    let a = unsafe { fmul_fast(x, PI) };
    unsafe { fdiv_fast(a.sin(), a) }
}

fn lanczos3(x: f32) -> f32 {
    if x.abs() >= 3.0 {
        0.0
    } else {
        unsafe { fmul_fast(sinc(x), sinc(fdiv_fast(x, 3.0))) }
    }
}

pub fn compute_lanczos_weights(src_len: u32, dst_len: u32) -> FilterBank {
    let scale = dst_len as f32 / src_len as f32;
    let is_downscaling = scale < 1.0;

    // 1. Determine Kernel Support (Radius)
    // If downscaling, we widen the kernel to prevent aliasing.
    // If upscaling, the kernel is fixed at radius 3.0.
    let support = if is_downscaling { 3.0 / scale } else { 3.0 };

    // Scale factor for the kernel evaluation (only used when downscaling)
    let kernel_scale = if is_downscaling { scale } else { 1.0 };

    let mut bank = Vec::with_capacity(dst_len as usize);

    for i in 0..dst_len {
        // 2. Map Destination Pixel to Source Coordinate Space
        // The center of the pixel 'i' in destination space maps to 'center' in source space.
        let center = (i as f32 + 0.5) / scale - 0.5;

        // 3. Determine the Window (Integers)
        // Which source pixels fall under the kernel?
        let start = (center - support).ceil() as isize;
        let end = (center + support).floor() as isize;

        let mut weights = Vec::new();
        let mut total_weight = 0.0;

        // We clamp the start index to valid image bounds for the storage index,
        // but we keep the math correct based on the virtual 'j'.
        // Note: For high-perf SIMD, it is often better to PAD the source image
        // with ghost pixels so you don't need clamping logic in the hot loop.
        // This implementation assumes you will handle the 'start' index carefully.
        let clamped_start = start.max(0).min(src_len as isize - 1) as usize;

        for j in start..=end {
            // Distance from center
            let distance = j as f32 - center;

            // 4. Sample the Kernel
            // If downscaling, we multiply distance by scale to 'stretch' the kernel input
            let w = lanczos3(distance * kernel_scale);

            // Only add if it contributes significantly (avoid denormals)
            // and apply edge handling (conceptually, we just weight the valid pixels)
            if j >= 0 && j < src_len as isize {
                weights.push(w);
                total_weight += w;
            }
        }

        // 5. Normalize Weights
        // The sum of weights must equal 1.0 to preserve brightness.
        if total_weight > 0.0 {
            for w in &mut weights {
                *w /= total_weight;
            }
        }

        bank.push(ResizeWeight {
            start_index: clamped_start,
            values: weights,
        });
    }

    bank
}

fn lanczos_resize(
    input_image: &PlanarBuffer,
    dst_width: u32,
    dst_height: u32,
) -> DynamicImage {
    let input_width = input_image.width;
    let input_height = input_image.height;

    let vertical_filters = compute_lanczos_weights(input_height, dst_height);
    let horizontal_filters = compute_lanczos_weights(input_width, dst_width);
    unsafe {
        // Vertical pass
        for y_out in 0..dst_height {
            let filter = vertical_filters[y_out as usize];

            // Iterate over the width in chunks of 16 pixels (AVX-512 width)
            for x_chunk in input_image {
                let mut accumulator = _mm512_setzero_ps();

                // Convolve: Iterate over the contributing source rows
                for (i, weight) in filter.values.iter().enumerate() {
                    let y_in = filter.start_index + i;

                    // Load 16 pixels from the source row
                    // Note: Since you have Vec<__m512>, this is just a direct slice access
                    let src_vec = input_image.get_chunk(y_in, x_chunk);

                    // Broadcast the weight to all 16 lanes
                    let w_vec = _mm512_set1_ps(*weight);

                    // Fused Multiply Add: acc = acc + (src * weight)
                    accumulator = _mm512_fmadd_ps(src_vec, w_vec, accumulator);
                }

                // Store the result in the Intermediate Buffer
                intermediate_image.set_chunk(y_out, x_chunk, accumulator);
            }
        }
    }
    todo!()
}
