use std::arch::x86_64::*;

use image::DynamicImage;
use std::intrinsics::{fadd_fast, fdiv_fast, fmul_fast, fsub_fast};
use std::simd::prelude::*;

const LANCZOS_RADIUS: f32 = 3.0;
const ROUNDING_MODE: i32 = _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC;

#[derive(Debug)]
struct PlanarBuffer {
    width: u32,
    height: u32,
    red: Vec<__m512>,   // Contiguous Reds
    green: Vec<__m512>, // Contiguous Greens
    blue: Vec<__m512>,  // Contiguous Blues
}

impl PlanarBuffer {
    fn new(width: u32, height: u32) -> Self {
        let size_in_floats = (width * height) as usize;
        let size_in_vectors = size_in_floats.div_ceil(16);
        Self {
            width,
            height,
            red: vec![unsafe { _mm512_setzero_ps() }; size_in_vectors],
            green: vec![unsafe { _mm512_setzero_ps() }; size_in_vectors],
            blue: vec![unsafe { _mm512_setzero_ps() }; size_in_vectors],
        }
    }

    fn from_rgb8(rgb8: &DynamicImage) -> Self {
        let frame = rgb8.to_rgb8();
        let width = frame.width();
        let height = frame.height();
        let pixel_count = (width * height) as usize;

        // Calculate exact vector capacity needed
        let vec_capacity = pixel_count.div_ceil(16);
        let mut all_reds: Vec<__m512> = Vec::with_capacity(vec_capacity);
        let mut all_greens: Vec<__m512> = Vec::with_capacity(vec_capacity);
        let mut all_blues: Vec<__m512> = Vec::with_capacity(vec_capacity);

        let mut pixels = frame.pixels();

        loop {
            let mut reds = [0u8; 64];
            let mut greens = [0u8; 64];
            let mut blues = [0u8; 64];

            match pixels.next_chunk::<64>() {
                Ok(chunk) => {
                    for (i, p) in chunk.iter().enumerate() {
                        reds[i] = p[0];
                        greens[i] = p[1];
                        blues[i] = p[2];
                    }

                    let r_vec = _mm512_cvtepu8_ps(__m512i::from(u8x64::from_array(reds)));
                    let g_vec = _mm512_cvtepu8_ps(__m512i::from(u8x64::from_array(greens)));
                    let b_vec = _mm512_cvtepu8_ps(__m512i::from(u8x64::from_array(blues)));

                    all_reds.extend_from_slice(&r_vec);
                    all_greens.extend_from_slice(&g_vec);
                    all_blues.extend_from_slice(&b_vec);
                }
                Err(remainder) => {
                    let mut count = 0;
                    for p in remainder {
                        reds[count] = p[0];
                        greens[count] = p[1];
                        blues[count] = p[2];
                        count += 1;
                    }

                    if count > 0 {
                        let r_vec = _mm512_cvtepu8_ps(__m512i::from(u8x64::from_array(reds)));
                        let g_vec = _mm512_cvtepu8_ps(__m512i::from(u8x64::from_array(greens)));
                        let b_vec = _mm512_cvtepu8_ps(__m512i::from(u8x64::from_array(blues)));

                        // Only take the vectors that contain valid data
                        let vecs_needed = count.div_ceil(16);
                        all_reds.extend_from_slice(&r_vec[0..vecs_needed]);
                        all_greens.extend_from_slice(&g_vec[0..vecs_needed]);
                        all_blues.extend_from_slice(&b_vec[0..vecs_needed]);
                    }
                    break;
                }
            }
        }

        Self {
            width,
            height,
            red: all_reds,
            green: all_greens,
            blue: all_blues,
        }
    }

    fn to_rgb8(&self) -> DynamicImage {
        let mut output = image::RgbImage::new(self.width, self.height);

        let red_slice = as_f32(&self.red);
        let green_slice = as_f32(&self.green);
        let blue_slice = as_f32(&self.blue);

        for (i, pixel) in output.pixels_mut().enumerate() {
            // PlanarBuffer is guaranteed to have capacity for all pixels (padded to 16 floats)
            // Clamping is necessary as Lanczos resampling can produce values outside [0, 255]
            let r = red_slice[i].round().clamp(0.0, 255.0) as u8;
            let g = green_slice[i].round().clamp(0.0, 255.0) as u8;
            let b = blue_slice[i].round().clamp(0.0, 255.0) as u8;

            *pixel = image::Rgb([r, g, b]);
        }

        image::DynamicImage::ImageRgb8(output)
    }
}

// this intrinsic doesn't exist, so i'm implementing it myself
// converts 64 bytes to 64 single-precision floats
#[inline(always)]
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

#[derive(Debug, Clone)]
pub struct FilterBounds {
    /// The index of the first source row/col to read
    pub start_index: usize,
    /// The list of normalized weights (length depends on the scale)
    pub values: Vec<f32>,
}

pub type FilterBank = Vec<FilterBounds>;

fn sinc(x: f32) -> f32 {
    if x == 0.0 {
        1.0
    } else {
        let pi_x = unsafe { fmul_fast(x, std::f32::consts::PI) };
        unsafe { fdiv_fast(pi_x.sin(), pi_x) }
    }
}

fn lanczos3_kernel(x: f32) -> f32 {
    let x = x.abs();
    if x < LANCZOS_RADIUS {
        unsafe { fmul_fast(sinc(x), sinc(fdiv_fast(x, LANCZOS_RADIUS))) }
    } else {
        0.0
    }
}

/// Function to precompute the lanczos3 kernel weights for a given scale
pub fn precompute_weights(src_size: u32, dst_size: u32) -> FilterBank {
    let scale = unsafe { fdiv_fast(src_size as f32, dst_size as f32) };
    let mut bounds = Vec::with_capacity(dst_size as usize);

    for i in 0..dst_size {
        unsafe {
            // Find the center of the destination pixel in source coordinates
            let center = fmul_fast(fadd_fast(i as f32, 0.5), scale);
            let start = fsub_fast(center, LANCZOS_RADIUS).ceil() as isize;
            let end = fadd_fast(center, LANCZOS_RADIUS).floor() as isize;

            let mut row_weights = Vec::new();
            let mut sum = 0.0;

            for j in start..=end {
                // Distance from center of source pixel to sampling point
                let weight = lanczos3_kernel(fsub_fast(fadd_fast(j as f32, 0.5), center));
                row_weights.push(weight);
                sum += weight;
            }

            // Normalization: Ensure the weights sum to 1.0 to preserve brightness
            for w in &mut row_weights {
                *w = fdiv_fast(*w, sum);
            }

            bounds.push(FilterBounds {
                start_index: start.max(0) as usize,
                values: row_weights,
            });
        }
    }
    bounds
}

#[inline(always)]
unsafe fn lanczos3_horizontal_pass_avx512(
    src: &[f32],
    dst: &mut [f32],
    src_width: u32,
    dst_width: u32,
    filters: &FilterBank,
) {
    debug_assert!(
        filters.len() == dst_width as usize,
        "Filter bank length must match destination width"
    );

    // Determine max filter size for padding
    let max_taps = filters.iter().map(|f| f.values.len()).max().unwrap_or(0);

    // Precompute transposed weights and indices for SIMD processing
    // We process 16 destination pixels at a time.
    // Transposed weights layout: [block][tap][lane_in_block]
    let num_blocks = dst_width.div_ceil(16) as usize;
    let mut indices_store = vec![0i32; num_blocks * 16];
    let mut weights_store = vec![0.0f32; num_blocks * max_taps * 16];

    for (i, f) in filters.iter().enumerate() {
        let block = i / 16;
        let lane = i % 16;

        // might wrap on 32-bit systems, but that's fine
        #[allow(clippy::cast_possible_wrap)]
        {
            indices_store[block * 16 + lane] = f.start_index as i32;
        }

        for (t, &w) in f.values.iter().enumerate() {
            // Linear index for the weight: block * (stride) + tap * 16 + lane
            let idx = block * (max_taps * 16) + t * 16 + lane;
            weights_store[idx] = w;
        }
    }

    let height = ((src.len() as u32) / src_width) as usize;

    unsafe {
        let src_limit = _mm512_set1_epi32((src_width.saturating_sub(1)).cast_signed());

        for y in 0..height {
            let src_row = src.as_ptr().add(y * src_width as usize);
            let dst_row = dst.as_mut_ptr().add(y * dst_width as usize);

            for b in 0..num_blocks {
                // Handle edge case where width is not a multiple of 16
                let pixels_remaining = dst_width.saturating_sub(b as u32 * 16);
                let mask = if pixels_remaining < 16 {
                    ((1u32 << pixels_remaining) - 1) as u16
                } else {
                    0xFFFF
                };

                // Load base starting indices for this block of 16 pixels
                let mut v_indices = _mm512_loadu_si512(indices_store.as_ptr().add(b * 16).cast());
                let mut v_acc = _mm512_setzero_ps();

                for t in 0..max_taps {
                    // Load weights for tap `t` across 16 pixels
                    let w_ptr = weights_store.as_ptr().add(b * max_taps * 16 + t * 16);
                    let v_w = _mm512_loadu_ps(w_ptr);

                    // Clamp indices to image bounds to ensure safe gather
                    let v_idx_clamped = _mm512_min_epi32(v_indices, src_limit);

                    // Gather source pixels: dst[i] uses src[indices[i]]
                    let v_src = _mm512_i32gather_ps(v_idx_clamped, src_row.cast(), 4);

                    // Accumulate: acc += weight * src
                    v_acc = _mm512_fmadd_round_ps(v_w, v_src, v_acc, ROUNDING_MODE);

                    // Increment indices to point to the next source pixel for the next tap
                    v_indices = _mm512_add_epi32(v_indices, _mm512_set1_epi32(1));
                }

                // Store results
                _mm512_mask_storeu_ps(dst_row.add(b * 16).cast(), mask, v_acc);
            }
        }
    }
}

/// Helper to cast &[__m512] -> &[f32]
#[inline(always)]
const fn as_f32(v: &[__m512]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<f32>(), v.len() * 16) }
}

/// Helper to cast &mut [__m512] -> &mut [f32]
#[inline(always)]
const fn as_f32_mut(v: &mut [__m512]) -> &mut [f32] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<f32>(), v.len() * 16) }
}

#[inline(always)]
fn lanczos3_horizontal(src: &PlanarBuffer, dst_width: u32, filters: &FilterBank) -> PlanarBuffer {
    let src_width = src.width;
    let src_height = src.height;
    // Horizontal pass changes width, keeps height
    let dst_height = src_height;

    let mut dst = PlanarBuffer::new(dst_width, dst_height);

    unsafe {
        lanczos3_horizontal_pass_avx512(
            as_f32(&src.red),
            as_f32_mut(&mut dst.red),
            src_width,
            dst_width,
            filters,
        );
        lanczos3_horizontal_pass_avx512(
            as_f32(&src.green),
            as_f32_mut(&mut dst.green),
            src_width,
            dst_width,
            filters,
        );
        lanczos3_horizontal_pass_avx512(
            as_f32(&src.blue),
            as_f32_mut(&mut dst.blue),
            src_width,
            dst_width,
            filters,
        );
    }
    dst
}

#[inline(always)]
unsafe fn lanczos3_vertical_pass_avx512(
    src_intermediate: &[f32], // Result of the horizontal pass
    dst: &mut [f32],
    width: u32, // Current width (after horizontal resize)
    height: u32,
    v_bounds: &FilterBank,
) {
    // Process the image in chunks of 16 columns
    for x_chunk in (0..width).step_by(16) {
        let rem = width - x_chunk;
        let mask = unsafe { _cvtu32_mask16((1u32 << rem.min(16)) - 1) };

        unsafe {
            for (y_dst, b) in v_bounds.iter().enumerate() {
                let mut acc = _mm512_setzero_ps();

                // Lanczos3 window of 6 vertical taps
                for (i, &weight) in b.values.iter().enumerate() {
                    let weight_vec = _mm512_set1_ps(weight);

                    // Calculate the pointer to the start of the 16-pixel chunk in the source row
                    let src_y = b.start_index + i;
                    // Clamp src_y to the last row to prevent OOB reads
                    let clamped_y = if src_y >= height as usize {
                        height - 1
                    } else {
                        src_y as u32
                    };

                    let src_ptr = src_intermediate
                        .as_ptr()
                        .add((clamped_y * width + x_chunk) as usize);

                    // Load 16 horizontal pixels from the source row
                    let pixels = _mm512_maskz_loadu_ps(mask, src_ptr);

                    // acc += pixels * weight
                    acc = _mm512_fmadd_round_ps(pixels, weight_vec, acc, ROUNDING_MODE);
                }

                // Store the 16 results into the destination row
                let dst_ptr = dst
                    .as_mut_ptr()
                    .add(y_dst * width as usize + x_chunk as usize);
                _mm512_mask_storeu_ps(dst_ptr, mask, acc);
            }
        }
    }
}

#[inline(always)]
fn lanczos3_vertical(src: &PlanarBuffer, dst_height: u32, filters: &FilterBank) -> PlanarBuffer {
    let mut dst = PlanarBuffer::new(src.width, dst_height);

    unsafe {
        lanczos3_vertical_pass_avx512(
            as_f32(&src.red),
            as_f32_mut(&mut dst.red),
            src.width,
            src.height,
            filters,
        );
        lanczos3_vertical_pass_avx512(
            as_f32(&src.green),
            as_f32_mut(&mut dst.green),
            src.width,
            src.height,
            filters,
        );
        lanczos3_vertical_pass_avx512(
            as_f32(&src.blue),
            as_f32_mut(&mut dst.blue),
            src.width,
            src.height,
            filters,
        );
    }
    dst
}

pub fn lanczos3_resize(
    input: &DynamicImage,
    dst_width: u32,
    dst_height: u32,
    horizontal_filters: &FilterBank,
    vertical_filters: &FilterBank,
) -> DynamicImage {
    let src = PlanarBuffer::from_rgb8(input);
    let dst = lanczos3_vertical(
        &lanczos3_horizontal(&src, dst_width, horizontal_filters),
        dst_height,
        vertical_filters,
    );
    dst.to_rgb8()
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_mm512_cvtepu8_ps() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512bw") {
            println!("AVX-512 not supported, skipping test");
            return;
        }

        let test_cases = vec![
            (0..64).collect::<Vec<u8>>().try_into().unwrap(),
            [255u8; 64],
            [0u8; 64],
        ];

        for input_bytes in test_cases {
            unsafe {
                // clippy false positive, loadu doesn't need alignment
                #[allow(clippy::cast_ptr_alignment)]
                let input_vec = _mm512_loadu_si512(input_bytes.as_ptr().cast::<__m512i>());
                let results = _mm512_cvtepu8_ps(input_vec);

                for i in 0..4 {
                    let mut output_floats = [0.0f32; 16];
                    _mm512_storeu_ps(output_floats.as_mut_ptr(), results[i]);

                    for j in 0..16 {
                        let expected = f32::from(input_bytes[i * 16 + j]);
                        assert_eq!(
                            output_floats[j],
                            expected,
                            "Mismatch at index {} for case {:?}",
                            i * 16 + j,
                            input_bytes
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_sinc() {
        assert_eq!(sinc(0.0), 1.0);
        assert!(sinc(1.0).abs() < 1e-6);
        assert!(sinc(2.0).abs() < 1e-6);
        assert!((sinc(0.5) - 0.636_619_75).abs() < 1e-6);
    }

    #[test]
    fn test_lanczos3_kernel() {
        if LANCZOS_RADIUS == 3.0 {
            assert_eq!(lanczos3_kernel(0.0), 1.0);
            assert!(lanczos3_kernel(3.0).abs() < 1e-6);
            assert!(lanczos3_kernel(-3.0).abs() < 1e-6);
            assert_eq!(lanczos3_kernel(4.0), 0.0);
            assert!(lanczos3_kernel(1.0).abs() < 1e-6); // Lanczos3 at 1.0 is 0}
        } else {
            eprintln!("LANCZOS_RADIUS is not 3.0, skipping test_lanczos3_kernel");
        }
    }

    #[test]
    fn test_precompute_weights() {
        if LANCZOS_RADIUS == 3.0 {
            let src_size = 100;
            let dst_size = 50;
            let weights = precompute_weights(src_size, dst_size);

            assert_eq!(weights.len(), dst_size as usize);

            for weight in &weights {
                let sum: f32 = weight.values.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-5,
                    "Weight sum should be 1.0, got {sum}"
                );
                assert!(weight.start_index < src_size as usize);
            }
        } else {
            eprintln!("LANCZOS_RADIUS is not 3.0, skipping test_precompute_weights");
        }
    }

    #[test]
    fn test_precompute_weights_identity() {
        let size = 10;
        let weights = precompute_weights(size, size);
        assert_eq!(weights.len(), size as usize);

        // For 1:1 scaling, Lanczos3 should behave like an identity for integer-aligned pixels
        let w5 = &weights[5];
        // center = 5.5, start = ceil(5.5 - 3) = 3, end = floor(5.5 + 3) = 8
        // j=3: dist -2, j=4: dist -1, j=5: dist 0, j=6: dist 1, j=7: dist 2, j=8: dist 3
        // All weights should be 0 except for j=5 which should be 1.0

        let mut found_one = false;
        for (i, &v) in w5.values.iter().enumerate() {
            let source_j = 3 + i; // start is 3
            if source_j == 5 {
                assert!((v - 1.0).abs() < 1e-6);
                found_one = true;
            } else {
                assert!(v.abs() < 1e-6);
            }
        }
        assert!(found_one);
    }

    #[test]
    fn test_lanczos3_horizontal_pass_avx512_correctness() {
        if !is_x86_feature_detected!("avx512f") {
            println!("AVX-512 not supported, skipping test");
            return;
        }

        let width_height_cases: Vec<(u32, u32, u32)> = vec![
            (32, 16, 1), // 2:1 downscale, aligned
            (16, 32, 1), // 1:2 upscale, aligned
            (35, 17, 2), // odd sizes, multi-row
            (10, 5, 1),  // small
        ];

        for (src_w, dst_w, h) in width_height_cases {
            let src: Vec<f32> = (0..src_w * h).map(|i| i as f32).collect();
            let mut dst = vec![0.0f32; (dst_w * h) as usize];
            let filters = precompute_weights(src_w, dst_w);

            unsafe {
                lanczos3_horizontal_pass_avx512(&src, &mut dst, src_w, dst_w, &filters);
            }

            // Reference check
            for y in 0..h {
                for x in 0..dst_w {
                    let filter = &filters[x as usize];
                    let mut sum = 0.0;
                    for (t, &w) in filter.values.iter().enumerate() {
                        let idx = filter.start_index + t;
                        // The SIMD implementation clamps the index to src_width - 1
                        let offset_idx = if idx >= src_w as usize {
                            src_w - 1
                        } else {
                            idx as u32
                        };
                        sum += src[(y * src_w + offset_idx) as usize] * w;
                    }

                    let computed = dst[(y * dst_w + x) as usize];
                    let diff = (computed - sum).abs();
                    assert!(
                        diff < 1e-4,
                        "Case {src_w}->{dst_w}: Mismatch at ({x},{y}) want {sum}, got {computed}, diff {diff}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_lanczos3_vertical_pass_avx512_correctness() {
        if !is_x86_feature_detected!("avx512f") {
            println!("AVX-512 not supported, skipping test");
            return;
        }

        let cases: Vec<(u32, u32, u32)> = vec![
            (16, 32, 16), // 1:2 Upscale, aligned
            (32, 16, 16), // 2:1 Downscale, aligned
            (17, 35, 12), // Odd sizes
            (5, 10, 8),   // Small
        ];

        for (src_h, dst_h, width) in cases {
            // Create source image (linear buffer)
            // Pattern: val = y * width + x
            let src: Vec<f32> = (0..width * src_h).map(|i| i as f32).collect();

            let mut dst = vec![0.0f32; (width * dst_h) as usize];
            let filters = precompute_weights(src_h, dst_h);

            unsafe {
                lanczos3_vertical_pass_avx512(&src, &mut dst, width, src_h, &filters);
            }

            // Reference check
            // For vertical pass, we iterate over columns (x), then rows (y_dst)
            for x in 0..width {
                for y_dst in 0..dst_h {
                    let filter = &filters[y_dst as usize];
                    let mut sum = 0.0;

                    for (t, &w) in filter.values.iter().enumerate() {
                        let src_y = filter.start_index + t;
                        // Clamp src_y to height - 1
                        let clamped_y = if src_y >= src_h as usize {
                            src_h as usize - 1
                        } else {
                            src_y
                        };

                        let val = src[clamped_y * width as usize + x as usize];
                        sum += val * w;
                    }

                    let computed = dst[(y_dst * width + x) as usize];
                    let diff = (computed - sum).abs();
                    assert!(
                        diff < 2e-4,
                        "Vertical Case {src_h}->{dst_h} (w={width}): Mismatch at ({x},{y_dst}) want {sum}, got {computed}, diff {diff}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_lanczos3_vertical_integration() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }

        let width = 32;
        let src_height = 32;
        let dst_height = 16;

        // This relies on PlanarBuffer::new being fixed to allocate enough!
        let mut src = PlanarBuffer::new(width, src_height);

        // Fill with pattern
        let src_floats = as_f32_mut(&mut src.red);
        // If allocation is still wrong (it shouldn't be), this protects from panic somewhat or fails strictly
        assert!(
            src_floats.len() >= (width * src_height) as usize,
            "PlanarBuffer allocation too small: {} < {}",
            src_floats.len(),
            width * src_height
        );

        for i in 0..(width * src_height) as usize {
            if i < src_floats.len() {
                src_floats[i] = i as f32;
            }
        }

        let filters = precompute_weights(src_height, dst_height);

        let dst = lanczos3_vertical(&src, dst_height, &filters);

        assert_eq!(dst.width, width);
        assert_eq!(dst.height, dst_height);

        // Check output is not all zeros
        let dst_floats = as_f32(&dst.red);
        assert!(dst_floats.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_horizontal_vertical_symmetry() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }

        // Test parameters
        // Use a width that fills a full SIMD vector (16) to properly test the vertical pass's full-width processing
        let size_in = 16;
        let size_out = 32;

        // Generate synthetic data (gradient)
        // Treat this data as:
        // 1. A Row of dimensions (size_in x 1) for the Horizontal Pass
        // 2. A Column of dimensions (1 x size_in) for the Vertical Pass
        //
        // NOTE: For the vertical pass, we need width=16 to use full SIMD lanes,
        // effectively running the "same" column operation 16 times in parallel.
        // So we will replicate the column to be 16x(size_in) for the vertical pass
        // to strictly test the 16-lane path vs the horizontal gather path.

        let src_len = size_in;
        let src_data: Vec<f32> = (0..src_len).map(|i| (i as f32) * 0.5).collect();
        let filters = precompute_weights(size_in, size_out);

        // Path A: Horizontal Pass
        // sizes: src: size_in x 1, dst: size_out x 1
        let mut dst_h = vec![0.0f32; size_out as usize];
        unsafe {
            lanczos3_horizontal_pass_avx512(&src_data, &mut dst_h, size_in, size_out, &filters);
        }

        // Path B: Vertical Pass (Single column case)
        let mut dst_v = vec![0.0f32; size_out as usize];
        unsafe {
            lanczos3_vertical_pass_avx512(
                &src_data, &mut dst_v, 1,       // width = 1
                size_in, // height = size_in
                &filters,
            );
        }

        // Compare results
        for i in 0..size_out as usize {
            let h_val = dst_h[i];
            let v_val = dst_v[i];
            let diff = (h_val - v_val).abs();

            assert!(
                diff < 1e-5,
                "Symmetry mismatch at index {i}: Horizontal={h_val} vs Vertical={v_val}, diff={diff}"
            );
        }

        // Path C: Full Width Vertical Pass Symmetry
        // Let's create a 16x(size_in) image where every column is identical.
        // After vertical pass, every column should be identical to dst_h.
        let width_v = 16;
        let mut src_full = vec![0.0f32; (width_v * size_in) as usize];
        for y in 0..size_in {
            for x in 0..width_v {
                src_full[(y * width_v + x) as usize] = src_data[y as usize];
            }
        }

        let mut dst_full = vec![0.0f32; (width_v * size_out) as usize];
        unsafe {
            lanczos3_vertical_pass_avx512(&src_full, &mut dst_full, width_v, size_in, &filters);
        }

        // Check that column 0 (or any column) matches the horizontal result
        for y in 0..size_out {
            let val_v = dst_full[(y * width_v) as usize];
            let val_h = dst_h[y as usize];
            let diff = (val_h - val_v).abs();
            assert!(
                diff < 1e-5,
                "Full-Width Symmetry mismatch at y={y}: H={val_h} vs V={val_v}, diff={diff}"
            );
        }
    }

    #[test]
    fn test_rgb8_to_planarbuffer() {
        if !is_x86_feature_detected!("avx512f") {
            println!("AVX-512 not supported, skipping test");
            return;
        }

        // 4x4 image = 16 pixels.
        // 16 pixels -> 1 vector.
        let width = 4u32;
        let height = 4u32;
        let mut img = image::RgbImage::new(width, height);

        // Fill with pattern
        // (x, y) = R: x, G: y, B: x+y
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = image::Rgb([x as u8, y as u8, (x + y) as u8]);
        }

        // Convert
        let dynamic_img = image::DynamicImage::ImageRgb8(img.clone());
        let pb = PlanarBuffer::from_rgb8(&dynamic_img);

        assert_eq!(pb.width, width);
        assert_eq!(pb.height, height);
        assert_eq!(pb.red.len(), 1); // 16 floats

        // helpers
        let r_floats = as_f32(&pb.red);
        let g_floats = as_f32(&pb.green);
        let b_floats = as_f32(&pb.blue);

        // Check values
        // Iterator order is row by row.
        let mut idx = 0;
        for y in 0..height {
            for x in 0..width {
                let expect_r = x as f32;
                let expect_g = y as f32;
                let expect_b = (x + y) as f32;

                assert_eq!(r_floats[idx], expect_r, "Red at {x},{y}");
                assert_eq!(g_floats[idx], expect_g, "Green at {x},{y}");
                assert_eq!(b_floats[idx], expect_b, "Blue at {x},{y}");
                idx += 1;
            }
        }
    }

    #[test]
    fn test_rgb8_to_planarbuffer_remainder() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }

        // Test logic for remainder loop
        // We trigger remainder if pixel count is not multiple of 64.
        // Let's use 70 pixels (64 + 6).
        // 70 pixels:
        // - 1 full chunk of 64 -> 4 vectors.
        // - 1 remainder of 6 -> 6 float -> 1 vector.
        // Total vectors: 5.
        // Total allocated capacity: 70/16 ceil = 5.

        let width = 10;
        let height = 7; // 70 pixels
        let mut img = image::RgbImage::new(width, height);

        // Set last pixel
        img.put_pixel(9, 6, image::Rgb([100, 200, 50]));

        let dynamic_img = image::DynamicImage::ImageRgb8(img);
        let pb = PlanarBuffer::from_rgb8(&dynamic_img);

        assert_eq!(pb.red.len(), 5);

        let r_floats = as_f32(&pb.red);
        let g_floats = as_f32(&pb.green);
        let b_floats = as_f32(&pb.blue);

        // Index 69 (last pixel)
        assert_eq!(r_floats[69], 100.0);
        assert_eq!(g_floats[69], 200.0);
        assert_eq!(b_floats[69], 50.0);
    }

    #[test]
    fn test_to_rgb8_basic() {
        if !is_x86_feature_detected!("avx512f") {
            println!("AVX-512 not supported, skipping test");
            return;
        }

        let width = 4;
        let height = 2; // 8 pixels total, fits in 1 vector (16 floats)
        let mut pb = PlanarBuffer::new(width, height);

        {
            let r = as_f32_mut(&mut pb.red);
            let g = as_f32_mut(&mut pb.green);
            let b = as_f32_mut(&mut pb.blue);

            // Set specific pixel values
            // Pixel (0,0) -> index 0. Standard black.
            r[0] = 0.0;
            g[0] = 0.0;
            b[0] = 0.0;
            // Pixel (1,0) -> index 1. Standard white.
            r[1] = 255.0;
            g[1] = 255.0;
            b[1] = 255.0;
            // Pixel (2,0) -> index 2. Test Rounding.
            // 127.6 -> 128, 128.4 -> 128, 50.5 -> 51
            r[2] = 127.6;
            g[2] = 128.4;
            b[2] = 50.5;
            // Pixel (3,0) -> index 3. Test Clamping.
            // -10 -> 0, 300 -> 255
            r[3] = -10.0;
            g[3] = 300.0;
            b[3] = 128.0;
        }

        let out_img = pb.to_rgb8();
        let rgb = out_img.to_rgb8();

        assert_eq!(rgb.width(), width);
        assert_eq!(rgb.height(), height);

        assert_eq!(rgb.get_pixel(0, 0), &image::Rgb([0, 0, 0]));
        assert_eq!(rgb.get_pixel(1, 0), &image::Rgb([255, 255, 255]));
        assert_eq!(rgb.get_pixel(2, 0), &image::Rgb([128, 128, 51]));
        assert_eq!(rgb.get_pixel(3, 0), &image::Rgb([0, 255, 128]));
    }
}
