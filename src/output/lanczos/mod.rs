use std::arch::x86_64::*;
use std::ptr::{read_unaligned, write_unaligned};

use image::DynamicImage;
use std::intrinsics::{fadd_fast, fdiv_fast, fmul_fast, fsub_fast};

use std::mem;
use std::simd::{StdFloat, prelude::*};

const LANCZOS_RADIUS: f32 = 3.0;
//const ROUNDING_MODE: i32 = _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC;

#[derive(Debug)]
struct PlanarBuffer {
    width: u32,
    height: u32,
    red: Vec<f32x16>,   // Contiguous Reds
    green: Vec<f32x16>, // Contiguous Greens
    blue: Vec<f32x16>,  // Contiguous Blues
}

impl PlanarBuffer {
    #[inline(always)]
    fn new(width: u32, height: u32) -> Self {
        let size_in_floats = (width * height) as usize;
        let size_in_vectors = size_in_floats.div_ceil(16);
        Self {
            width,
            height,
            red: vec![unsafe { mem::zeroed() }; size_in_vectors],
            green: vec![unsafe { mem::zeroed() }; size_in_vectors],
            blue: vec![unsafe { mem::zeroed() }; size_in_vectors],
        }
    }

    fn from_rgb8(rgb8: &DynamicImage) -> Self {
        let frame = rgb8.to_rgb8();
        let width = frame.width();
        let height = frame.height();
        let pixel_count = (width * height) as usize;

        let vec_capacity = pixel_count.div_ceil(16);
        let mut all_reds = Vec::with_capacity(vec_capacity);
        let mut all_greens = Vec::with_capacity(vec_capacity);
        let mut all_blues = Vec::with_capacity(vec_capacity);

        let pixels = frame.as_flat_samples().samples;

        // VBMI De-interleaving Indices for 64 pixels (3 x 64 bytes = 192 bytes)
        // We load 3 ZMM registers (V0, V1, V2) and produce 3 ZMM registers (R, G, B)
        // R indices for V0/V1 and V1/V2
        let mut r_idx_a = [0u8; 64];
        let mut r_idx_b = [0u8; 64];
        let mut g_idx_a = [0u8; 64];
        let mut g_idx_b = [0u8; 64];
        let mut b_idx_a = [0u8; 64];
        let mut b_idx_b = [0u8; 64];

        for i in 0..64 {
            // R: 0, 3, 6 ... 189
            // G: 1, 4, 7 ... 190
            // B: 2, 5, 8 ... 191
            if i < 32 {
                r_idx_a[i] = (i * 3) as u8;
                g_idx_a[i] = (i * 3 + 1) as u8;
                b_idx_a[i] = (i * 3 + 2) as u8;
            } else {
                r_idx_b[i - 32] = (i * 3) as u8;
                g_idx_b[i - 32] = (i * 3 + 1) as u8;
                b_idx_b[i - 32] = (i * 3 + 2) as u8;
            }
        }

        let mut chunks = pixels.chunks_exact(192);
        for chunk in &mut chunks {
            let v0: u8x64 = u8x64::from_slice(chunk);
            let v1: u8x64 = u8x64::from_slice(&chunk[64..]);
            let v2: u8x64 = u8x64::from_slice(&chunk[128..]);

            let r_bytes = Self::_mm512_permutex3var_epi8(v0, v1, v2, 0);
            let g_bytes = Self::_mm512_permutex3var_epi8(v0, v1, v2, 1);
            let b_bytes = Self::_mm512_permutex3var_epi8(v0, v1, v2, 2);

            all_reds.extend_from_slice(&_mm512_cvtepu8_ps(r_bytes));
            all_greens.extend_from_slice(&_mm512_cvtepu8_ps(g_bytes));
            all_blues.extend_from_slice(&_mm512_cvtepu8_ps(b_bytes));
        }

        // Remainder
        let remainder = chunks.remainder();
        if !remainder.is_empty() {
            // Scalar fallback for the very last few pixels is fine
            let mut reds = [0u8; 64];
            let mut greens = [0u8; 64];
            let mut blues = [0u8; 64];
            let mut count = 0;
            for p in remainder.chunks_exact(3) {
                reds[count] = p[0];
                greens[count] = p[1];
                blues[count] = p[2];
                count += 1;
            }
            if count > 0 {
                let r_batch = _mm512_cvtepu8_ps(u8x64::from_slice(&reds));
                let g_batch = _mm512_cvtepu8_ps(u8x64::from_slice(&greens));
                let b_batch = _mm512_cvtepu8_ps(u8x64::from_slice(&blues));
                let needed = count.div_ceil(16);

                all_reds.extend_from_slice(&r_batch[..needed]);
                all_greens.extend_from_slice(&g_batch[..needed]);
                all_blues.extend_from_slice(&b_batch[..needed]);
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

    #[inline(always)]
    fn permutex2var_epi8(a: u8x64, index: u8x64, b: u8x64) -> u8x64 {
        // low 6 bits = byte index
        let idx = index & u8x64::splat(0x3F);

        // swizzle both sources
        let from_a = a.swizzle_dyn(idx);
        let from_b = b.swizzle_dyn(idx);

        // bit 6 selects source
        let select_b: mask8x64 = (index & u8x64::splat(0x40)).simd_ne(u8x64::splat(0));
        select_b.select(from_b, from_a)
    }

    #[inline(always)]
    fn _mm512_permutex3var_epi8(v0: u8x64, v1: u8x64, v2: u8x64, channel: u8) -> u8x64 {
        let mut idx_low = [0u8; 64];
        let mut idx_high = [0u8; 64];
        let mut mask_val = 0u64;

        for i in 0..64 {
            let src = i as u32 * 3 + u32::from(channel);
            if src < 128 {
                idx_low[i] = src as u8;
            } else {
                idx_high[i] = (src - 64) as u8;
                mask_val |= 1 << i;
            }
        }

        if is_x86_feature_detected!("avx512vbmi") && is_x86_feature_detected!("avx512bw") {
            unsafe {
                let v0 = __m512i::from(v0);
                let v1 = __m512i::from(v1);
                let v2 = __m512i::from(v2);
                let m = _cvtu64_mask64(mask_val);
                let res_low =
                    _mm512_permutex2var_epi8(v0, _mm512_loadu_si512(idx_low.as_ptr().cast()), v1);
                let res_high =
                    _mm512_permutex2var_epi8(v1, _mm512_loadu_si512(idx_high.as_ptr().cast()), v2);
                u8x64::from(_mm512_mask_blend_epi8(m, res_low, res_high))
            }
        } else {
            let m: mask8x64 = mask8x64::from_bitmask(mask_val);
            let low_vector: u8x64 = u8x64::from_slice(&idx_low);
            let high_vector: u8x64 = u8x64::from_slice(&idx_high);
            let res_low = Self::permutex2var_epi8(v0, low_vector, v1);
            let res_high = Self::permutex2var_epi8(v1, high_vector, v2);
            m.select(res_high, res_low)
        }
    }

    fn to_rgb8(&self) -> DynamicImage {
        let mut output = image::RgbImage::new(self.width, self.height);
        let pixel_count = (self.width * self.height) as usize;

        unsafe {
            let red_ptr = self.red.as_ptr().cast::<__m512>();
            let green_ptr = self.green.as_ptr().cast::<__m512>();
            let blue_ptr = self.blue.as_ptr().cast::<__m512>();

            let output_ptr = output.as_flat_samples_mut().samples.as_mut_ptr();

            let mut pix = 0;
            while pix + 64 <= pixel_count {
                let r_batch = Self::pack_floats_to_u8(
                    f32x16::from(*red_ptr.add(pix / 16)),
                    f32x16::from(*red_ptr.add(pix / 16 + 1)),
                    f32x16::from(*red_ptr.add(pix / 16 + 2)),
                    f32x16::from(*red_ptr.add(pix / 16 + 3)),
                );
                let g_batch = Self::pack_floats_to_u8(
                    f32x16::from(*green_ptr.add(pix / 16)),
                    f32x16::from(*green_ptr.add(pix / 16 + 1)),
                    f32x16::from(*green_ptr.add(pix / 16 + 2)),
                    f32x16::from(*green_ptr.add(pix / 16 + 3)),
                );
                let b_batch = Self::pack_floats_to_u8(
                    f32x16::from(*blue_ptr.add(pix / 16)),
                    f32x16::from(*blue_ptr.add(pix / 16 + 1)),
                    f32x16::from(*blue_ptr.add(pix / 16 + 2)),
                    f32x16::from(*blue_ptr.add(pix / 16 + 3)),
                );

                let (v0, v1, v2) = Self::interleave_rgb(r_batch, g_batch, b_batch);

                write_unaligned(output_ptr.add(pix * 3).cast(), v0);
                write_unaligned(output_ptr.add(pix * 3 + 64).cast(), v1);
                write_unaligned(output_ptr.add(pix * 3 + 128).cast(), v2);

                pix += 64;
            }

            // Remainder scalar fallback
            let r_slice = as_f32(&self.red);
            let g_slice = as_f32(&self.green);
            let b_slice = as_f32(&self.blue);

            for i in pix..pixel_count {
                let r_val = r_slice[i].round().clamp(0.0, 255.0) as u8;
                let g_val = g_slice[i].round().clamp(0.0, 255.0) as u8;
                let b_val = b_slice[i].round().clamp(0.0, 255.0) as u8;
                *output.get_pixel_mut(i as u32 % self.width, i as u32 / self.width) =
                    image::Rgb([r_val, g_val, b_val]);
            }
        }

        image::DynamicImage::ImageRgb8(output)
    }

    // Helper: Pack 64 floats to 64 bytes with clamping
    #[inline(always)]
    fn pack_floats_to_u8(v0: f32x16, v1: f32x16, v2: f32x16, v3: f32x16) -> u8x64 {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                // Round to nearest integer (cvteps_epi32)
                let i0 = _mm512_cvttps_epi32(v0.into());
                let i1 = _mm512_cvttps_epi32(v1.into());
                let i2 = _mm512_cvttps_epi32(v2.into());
                let i3 = _mm512_cvttps_epi32(v3.into());

                // Pack with unsigned saturation to u8 (AVX-512F/BW/DQ)
                let b0 = _mm512_cvtusepi32_epi8(i0); // 16 bytes in __m128i
                let b1 = _mm512_cvtusepi32_epi8(i1);
                let b2 = _mm512_cvtusepi32_epi8(i2);
                let b3 = _mm512_cvtusepi32_epi8(i3);

                // Combine 4 XMMs into one ZMM
                u8x64::from(_mm512_inserti32x4(
                    _mm512_inserti32x4(
                        _mm512_inserti32x4(_mm512_castsi128_si512(b0), b1, 1),
                        b2,
                        2,
                    ),
                    b3,
                    3,
                ))
            }
        } else {
            // no built-in saturating cast, but luckily as-casting float to int saturates so that's what portable_simd cast() does
            // relevant issue: https://github.com/rust-lang/portable-simd/issues/369
            let b0: u8x16 = v0.cast();
            let b1: u8x16 = v1.cast();
            let b2: u8x16 = v2.cast();
            let b3: u8x16 = v3.cast();

            unsafe { mem::transmute::<[u8x16; 4], u8x64>([b0, b1, b2, b3]) }
        }
    }

    // compiles into a disgusting straight-through of instructions. how
    #[inline(always)]
    fn interleave(r: u8x64, g: u8x64, b: u8x64, raw_idx: &[u8; 64]) -> u8x64 {
        let mut idx_low = [0u8; 64];
        let mut idx_high = [0u8; 64];
        let mut mask_val = 0u64;

        for (i, &src) in raw_idx.iter().enumerate() {
            let src_val = u32::from(src);
            if src_val < 128 {
                idx_low[i] = src_val as u8;
            } else {
                idx_high[i] = (src_val - 64) as u8;
                mask_val |= 1 << i;
            }
        }

        if is_x86_feature_detected!("avx512vbmi") && is_x86_feature_detected!("avx512bw") {
            unsafe {
                let m = _cvtu64_mask64(mask_val);
                let res_low = _mm512_permutex2var_epi8(
                    __m512i::from(r),
                    _mm512_loadu_si512(idx_low.as_ptr().cast()),
                    __m512i::from(g),
                );
                let res_high = _mm512_permutex2var_epi8(
                    __m512i::from(g),
                    _mm512_loadu_si512(idx_high.as_ptr().cast()),
                    __m512i::from(b),
                );

                u8x64::from(_mm512_mask_blend_epi8(m, res_low, res_high))
            }
        } else {
            let m: mask8x64 = mask8x64::from_bitmask(mask_val);
            let low_vector: u8x64 = u8x64::from_slice(&idx_low);
            let high_vector: u8x64 = u8x64::from_slice(&idx_high);
            let res_low = Self::permutex2var_epi8(r, low_vector, g);
            let res_high = Self::permutex2var_epi8(g, high_vector, b);
            m.select(res_high, res_low)
        }
    }

    // Helper: Interleave planar R, G, B into packed RGB (VBMI)
    #[inline(always)]
    fn interleave_rgb(r: u8x64, g: u8x64, b: u8x64) -> (u8x64, u8x64, u8x64) {
        // Define indices for the 3 output registers
        let mut idx0 = [0u8; 64];
        let mut idx1 = [0u8; 64];
        let mut idx2 = [0u8; 64];

        for i in 0..64 {
            let pixel_idx = i / 3;
            let channel = i % 3;
            let val = if channel == 0 {
                pixel_idx
            } else if channel == 1 {
                pixel_idx + 64
            } else {
                pixel_idx + 128
            };
            idx0[i] = val as u8;

            let pixel_idx = (i + 64) / 3;
            let channel = (i + 64) % 3;
            let val = if channel == 0 {
                pixel_idx
            } else if channel == 1 {
                pixel_idx + 64
            } else {
                pixel_idx + 128
            };
            idx1[i] = val as u8;

            let pixel_idx = (i + 128) / 3;
            let channel = (i + 128) % 3;
            let val = if channel == 0 {
                pixel_idx
            } else if channel == 1 {
                pixel_idx + 64
            } else {
                pixel_idx + 128
            };
            idx2[i] = val as u8;
        }

        let v0 = Self::interleave(r, g, b, &idx0);
        let v1 = Self::interleave(r, g, b, &idx1);
        let v2 = Self::interleave(r, g, b, &idx2);

        (v0, v1, v2)
    }

    #[inline(always)]
    pub unsafe fn simd_i32gather_ps<const SCALE: i32>(offsets: i32x16, base: *const f32) -> f32x16 {
        // SCALE must be 1, 2, 4, or 8 per x86 gather rules
        //debug_assert!(matches!(SCALE, 1 | 2 | 4 | 8));
        const {
            assert!(
                !(SCALE != 1 && SCALE != 2 && SCALE != 4 && SCALE != 8),
                "SCALE must be 1, 2, 4, or 8"
            );
        }

        let mut output = [0.0f32; 16];

        for (offset, out_f) in offsets.as_array().iter().zip(output.iter_mut()) {
            // Compute byte offset exactly like the intrinsic
            let byte_offset = *offset as isize * SCALE as isize;

            // Convert base pointer to byte pointer
            let addr = unsafe { base.byte_offset(byte_offset) };

            // Unchecked load — identical UB semantics to AVX-512 gather
            *out_f = unsafe { read_unaligned(addr) };
        }

        f32x16::from_array(output)
    }
}

// this intrinsic doesn't exist, so i'm implementing it myself
// converts 64 bytes to 64 single-precision floats
#[inline(always)]
fn _mm512_cvtepu8_ps(input: u8x64) -> [f32x16; 4] {
    let output: f32x64 = if is_x86_feature_detected!("avx512f") {
        unsafe {
            let input = __m512i::from(input);
            mem::transmute::<[__m512; 4], f32x64>([
                _mm512_cvtepu32_ps(_mm512_cvtepu8_epi32(_mm512_castsi512_si128(input))),
                _mm512_cvtepu32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(input, 1))),
                _mm512_cvtepu32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(input, 2))),
                _mm512_cvtepu32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(input, 3))),
            ])
        }
    } else {
        input.cast()
    };

    let output = output.as_array();
    [
        f32x16::from_slice(&output[0..16]),
        f32x16::from_slice(&output[16..32]),
        f32x16::from_slice(&output[32..48]),
        f32x16::from_slice(&output[48..64]),
    ]
}

#[derive(Debug, Clone)]
pub struct FilterBounds {
    /// The index of the first source row/col to read
    pub start_index: usize,
    /// The list of normalized weights (length depends on the scale)
    pub values: Vec<f32>,
}

pub type FilterBank = Vec<FilterBounds>;

#[inline(always)]
#[allow(unused)]
fn sinc(x: f32) -> f32 {
    if x == 0.0 || x.is_subnormal() {
        1.0
    } else {
        let pi_x = unsafe { fmul_fast(x, std::f32::consts::PI) };
        unsafe { fdiv_fast(pi_x.sin(), pi_x) }
    }
}

#[inline(always)]
#[allow(clippy::unreadable_literal)]
pub fn sinc_approx(x: f32) -> f32 {
    // Minimax coefficients optimized for [0, 1]
    const C0: f32 = 0.999996;
    const C1: f32 = -1.6448622;
    const C2: f32 = 0.8114296;
    const C3: f32 = -0.19018897;
    const C4: f32 = 0.025642106;
    const C5: f32 = -0.0021039767;
    const C6: f32 = 8.781045e-05;

    if x == 0.0 || x.is_subnormal() {
        1.0
    } else {
        unsafe {
            use std::intrinsics::{fadd_fast, fmul_fast};

            // 1. Pre-calculate powers of u (u = x^2)
            let u = fmul_fast(x, x);
            let u2 = fmul_fast(u, u);
            let u4 = fmul_fast(u2, u2);

            // 2. Layer 1: Independent pairs (C_n + C_{n+1}*u)
            // Equivalent to: C1 * u + C0
            let pair_0_1 = fadd_fast(fmul_fast(C1, u), C0);
            let pair_2_3 = fadd_fast(fmul_fast(C3, u), C2);
            let pair_4_5 = fadd_fast(fmul_fast(C5, u), C4);

            // 3. Layer 2: Combine pairs using u^2
            // Equivalent to: pair_0_1 + pair_2_3 * u^2
            let quad_0_3 = fadd_fast(pair_0_1, fmul_fast(pair_2_3, u2));

            // Equivalent to: pair_4_5 + C6 * u^2
            let quad_4_6 = fadd_fast(pair_4_5, fmul_fast(C6, u2));

            // 4. Layer 3: Final Combination using u^4
            // Equivalent to: quad_0_3 + quad_4_6 * u^4
            fadd_fast(quad_0_3, fmul_fast(quad_4_6, u4))
        }
    }
}

#[inline(always)]
fn lanczos3_kernel(x: f32) -> f32 {
    let x = x.abs();
    if x < LANCZOS_RADIUS {
        unsafe { fmul_fast(sinc_approx(x), sinc_approx(fdiv_fast(x, LANCZOS_RADIUS))) }
    } else {
        0.0
    }
}

/// Precompute the lanczos3 kernel weights for a given scale for use with `lanczos3_resize()`
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
        let src_limit = i32x16::splat((src_width.saturating_sub(1)).cast_signed());

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
                let mut v_indices: i32x16 =
                    read_unaligned(indices_store.as_ptr().add(b * 16).cast());

                // Unrolling factor of 4 to hide FMA latency (Zen 5 FMA latency ~4 cycles)
                let mut v_acc0: f32x16 = mem::zeroed();
                let mut v_acc1: f32x16 = mem::zeroed();
                let mut v_acc2: f32x16 = mem::zeroed();
                let mut v_acc3: f32x16 = mem::zeroed();

                let mut t = 0;
                let v_one = i32x16::splat(1);
                let v_two = i32x16::splat(2);
                let v_three = i32x16::splat(3);
                let v_four = i32x16::splat(4);

                if is_x86_feature_detected!("avx512f") {
                    let mut v_indices = __m512i::from(v_indices);
                    // Main unrolled loop
                    while t + 4 <= max_taps {
                        let w_ptr0 = weights_store.as_ptr().add(b * max_taps * 16 + t * 16);
                        let w_ptr1 = weights_store.as_ptr().add(b * max_taps * 16 + (t + 1) * 16);
                        let w_ptr2 = weights_store.as_ptr().add(b * max_taps * 16 + (t + 2) * 16);
                        let w_ptr3 = weights_store.as_ptr().add(b * max_taps * 16 + (t + 3) * 16);

                        let src_limit = __m512i::from(src_limit);
                        let v_one = __m512i::from(v_one);
                        let v_two = __m512i::from(v_two);
                        let v_three = __m512i::from(v_three);
                        let v_four = __m512i::from(v_four);
                        // Tap 0
                        let v_w0 = _mm512_loadu_ps(w_ptr0);
                        // v_indices is already correct for t
                        let v_idx0 = _mm512_min_epi32(v_indices, src_limit);
                        let v_src0 = _mm512_i32gather_ps(v_idx0, src_row.cast(), 4);
                        v_acc0 = f32x16::from(_mm512_fmadd_ps(v_w0, v_src0, __m512::from(v_acc0)));

                        // Tap 1
                        let v_w1 = _mm512_loadu_ps(w_ptr1);
                        let v_idx1_raw = _mm512_add_epi32(v_indices, v_one);
                        let v_idx1 = _mm512_min_epi32(v_idx1_raw, src_limit);
                        let v_src1 = _mm512_i32gather_ps(v_idx1, src_row.cast(), 4);
                        v_acc1 = f32x16::from(_mm512_fmadd_ps(v_w1, v_src1, __m512::from(v_acc1)));

                        // Tap 2
                        let v_w2 = _mm512_loadu_ps(w_ptr2);
                        let v_idx2_raw = _mm512_add_epi32(v_indices, v_two);
                        let v_idx2 = _mm512_min_epi32(v_idx2_raw, src_limit);
                        let v_src2 = _mm512_i32gather_ps(v_idx2, src_row.cast(), 4);
                        v_acc2 = f32x16::from(_mm512_fmadd_ps(v_w2, v_src2, __m512::from(v_acc2)));

                        // Tap 3
                        let v_w3 = _mm512_loadu_ps(w_ptr3);
                        let v_idx3_raw = _mm512_add_epi32(v_indices, v_three);
                        let v_idx3 = _mm512_min_epi32(v_idx3_raw, src_limit);
                        let v_src3 = _mm512_i32gather_ps(v_idx3, src_row.cast(), 4);
                        v_acc3 = f32x16::from(_mm512_fmadd_ps(v_w3, v_src3, __m512::from(v_acc3)));

                        t += 4;
                        v_indices = _mm512_add_epi32(v_indices, v_four);
                    }

                    // Handle remaining taps
                    while t < max_taps {
                        let w_ptr = weights_store.as_ptr().add(b * max_taps * 16 + t * 16);

                        let v_w = _mm512_loadu_ps(w_ptr);
                        let v_idx_clamped = _mm512_min_epi32(v_indices, __m512i::from(src_limit));
                        let v_src = _mm512_i32gather_ps(v_idx_clamped, src_row.cast(), 4);

                        // Use acc0 for remainder
                        v_acc0 = f32x16::from(_mm512_fmadd_ps(v_w, v_src, __m512::from(v_acc0)));

                        t += 1;
                        v_indices = _mm512_add_epi32(v_indices, __m512i::from(v_one));
                    }
                } else {
                    // Main unrolled loop
                    while t + 4 <= max_taps {
                        let w_ptr0 = weights_store.as_ptr().add(b * max_taps * 16 + t * 16);
                        let w_ptr1 = weights_store.as_ptr().add(b * max_taps * 16 + (t + 1) * 16);
                        let w_ptr2 = weights_store.as_ptr().add(b * max_taps * 16 + (t + 2) * 16);
                        let w_ptr3 = weights_store.as_ptr().add(b * max_taps * 16 + (t + 3) * 16);

                        // Tap 0
                        let v_w0: f32x16 = read_unaligned(w_ptr0.cast());
                        // v_indices is already correct for t
                        let v_idx0 = v_indices.simd_min(src_limit);
                        let v_src0 = PlanarBuffer::simd_i32gather_ps::<4>(v_idx0, src_row);
                        v_acc0 = v_w0.mul_add(v_src0, v_acc0);

                        // Tap 1
                        let v_w1: f32x16 = read_unaligned(w_ptr1.cast());
                        let v_idx1_raw = v_indices + v_one;
                        let v_idx1 = v_idx1_raw.simd_min(src_limit);
                        let v_src1 = PlanarBuffer::simd_i32gather_ps::<4>(v_idx1, src_row);
                        v_acc1 = v_w1.mul_add(v_src1, v_acc1);

                        // Tap 2
                        let v_w2: f32x16 = read_unaligned(w_ptr2.cast());
                        let v_idx2_raw = v_indices + v_two;
                        let v_idx2 = v_idx2_raw.simd_min(src_limit);
                        let v_src2 = PlanarBuffer::simd_i32gather_ps::<4>(v_idx2, src_row);
                        v_acc2 = v_w2.mul_add(v_src2, v_acc2);

                        // Tap 3
                        let v_w3: f32x16 = read_unaligned(w_ptr3.cast());
                        let v_idx3_raw = v_indices + v_three;
                        let v_idx3 = v_idx3_raw.simd_min(src_limit);
                        let v_src3 = PlanarBuffer::simd_i32gather_ps::<4>(v_idx3, src_row);
                        v_acc3 = v_w3.mul_add(v_src3, v_acc3);

                        t += 4;
                        v_indices += v_four;
                    }

                    // Handle remaining taps
                    while t < max_taps {
                        let w_ptr = weights_store.as_ptr().add(b * max_taps * 16 + t * 16);

                        let v_w: f32x16 = read_unaligned(w_ptr.cast());
                        let v_idx_clamped = v_indices.simd_min(src_limit);
                        let v_src = PlanarBuffer::simd_i32gather_ps::<4>(v_idx_clamped, src_row);

                        // Use acc0 for remainder
                        v_acc0 = v_w.mul_add(v_src, v_acc0);

                        t += 1;
                        v_indices += v_one;
                    }
                }

                // Sum accumulators
                let v_acc01 = v_acc0 + v_acc1;
                let v_acc23 = v_acc2 + v_acc3;
                let v_acc = v_acc01 + v_acc23;

                // Store results
                v_acc.store_select_ptr(
                    dst_row.add(b * 16).cast(),
                    mask32x16::from_bitmask(u64::from(mask)),
                );
            }
        }
    }
}

/// Helper to cast &[f32x16] -> &[f32]
#[inline(always)]
const fn as_f32(v: &[f32x16]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<f32>(), v.len() * 16) }
}

/// Helper to cast &mut [f32x16] -> &mut [f32]
#[inline(always)]
const fn as_f32_mut(v: &mut [f32x16]) -> &mut [f32] {
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
        let mask = ((1u32 << rem.min(16)) - 1) as u16;

        unsafe {
            for (y_dst, b) in v_bounds.iter().enumerate() {
                let mut acc0: f32x16 = mem::zeroed();
                let mut acc1: f32x16 = mem::zeroed();
                let mut acc2: f32x16 = mem::zeroed();
                let mut acc3: f32x16 = mem::zeroed();

                let taps = b.values.len();
                let mut i = 0;

                // Optimization: Hoist boundary check.
                // If the entire filter footprint is within bounds, skip clamping in the inner loop.
                let is_safe = (b.start_index + taps) <= height as usize;

                if is_safe {
                    while i + 4 <= taps {
                        let start_y = b.start_index + i;

                        if is_x86_feature_detected!("avx512f") {
                            // Tap 0
                            let w0 = _mm512_set1_ps(b.values[i]);
                            let ptr0 = src_intermediate
                                .as_ptr()
                                .add(start_y * width as usize + x_chunk as usize);
                            let p0 = _mm512_maskz_loadu_ps(mask, ptr0);
                            acc0 = f32x16::from(_mm512_fmadd_ps(p0, w0, __m512::from(acc0)));

                            // Tap 1
                            let w1 = _mm512_set1_ps(b.values[i + 1]);
                            let ptr1 = src_intermediate
                                .as_ptr()
                                .add((start_y + 1) * width as usize + x_chunk as usize);
                            let p1 = _mm512_maskz_loadu_ps(mask, ptr1);
                            acc1 = f32x16::from(_mm512_fmadd_ps(p1, w1, __m512::from(acc1)));

                            // Tap 2
                            let w2 = _mm512_set1_ps(b.values[i + 2]);
                            let ptr2 = src_intermediate
                                .as_ptr()
                                .add((start_y + 2) * width as usize + x_chunk as usize);
                            let p2 = _mm512_maskz_loadu_ps(mask, ptr2);
                            acc2 = f32x16::from(_mm512_fmadd_ps(p2, w2, __m512::from(acc2)));

                            // Tap 3
                            let w3 = _mm512_set1_ps(b.values[i + 3]);
                            let ptr3 = src_intermediate
                                .as_ptr()
                                .add((start_y + 3) * width as usize + x_chunk as usize);
                            let p3 = _mm512_maskz_loadu_ps(mask, ptr3);
                            acc3 = f32x16::from(_mm512_fmadd_ps(p3, w3, __m512::from(acc3)));
                        } else {
                            // Tap 0
                            let w0 = f32x16::splat(b.values[i]);
                            let ptr0 = src_intermediate
                                .as_ptr()
                                .add(start_y * width as usize + x_chunk as usize);
                            let p0 = f32x16::load_select_ptr(
                                ptr0,
                                mask32x16::from_bitmask(u64::from(mask)),
                                f32x16::splat(0.0),
                            );
                            acc0 = p0.mul_add(w0, acc0);

                            // Tap 1
                            let w1 = f32x16::splat(b.values[i + 1]);
                            let ptr1 = src_intermediate
                                .as_ptr()
                                .add((start_y + 1) * width as usize + x_chunk as usize);
                            let p1 = f32x16::load_select_ptr(
                                ptr1,
                                mask32x16::from_bitmask(u64::from(mask)),
                                f32x16::splat(0.0),
                            );
                            acc1 = p1.mul_add(w1, acc1);

                            // Tap 2
                            let w2 = f32x16::splat(b.values[i + 2]);
                            let ptr2 = src_intermediate
                                .as_ptr()
                                .add((start_y + 2) * width as usize + x_chunk as usize);
                            let p2 = f32x16::load_select_ptr(
                                ptr2,
                                mask32x16::from_bitmask(u64::from(mask)),
                                f32x16::splat(0.0),
                            );
                            acc2 = p2.mul_add(w2, acc2);

                            // Tap 3
                            let w3 = f32x16::splat(b.values[i + 3]);
                            let ptr3 = src_intermediate
                                .as_ptr()
                                .add((start_y + 3) * width as usize + x_chunk as usize);
                            let p3 = f32x16::load_select_ptr(
                                ptr3,
                                mask32x16::from_bitmask(u64::from(mask)),
                                f32x16::splat(0.0),
                            );
                            acc3 = p3.mul_add(w3, acc3);
                        }

                        i += 4;
                    }
                }

                #[cfg(not(test))]
                #[allow(deprecated)]
                {
                    debug_assert_eq!(_MM_GET_FLUSH_ZERO_MODE(), _MM_FLUSH_ZERO_ON);
                }

                // Fallback loop for remainder and unsafe rows (edges)
                while i < taps {
                    let weight = b.values[i];
                    let weight_vec = f32x16::splat(weight);

                    let src_y = b.start_index + i;
                    let clamped_y = if src_y >= height as usize {
                        height - 1
                    } else {
                        src_y as u32
                    };

                    let src_ptr = src_intermediate
                        .as_ptr()
                        .add((clamped_y * width + x_chunk) as usize);

                    let pixels = f32x16::load_select_ptr(
                        src_ptr,
                        mask32x16::from_bitmask(u64::from(mask)),
                        f32x16::splat(0.0),
                    );

                    // Accumulate into acc0
                    acc0 = pixels.mul_add(weight_vec, acc0);
                    i += 1;
                }

                let acc01 = acc0 + acc1;
                let acc23 = acc2 + acc3;
                let acc = acc01 + acc23;

                // Store the 16 results into the destination row
                let dst_ptr = dst
                    .as_mut_ptr()
                    .add(y_dst * width as usize + x_chunk as usize);
                acc.store_select_ptr(dst_ptr, mask32x16::from_bitmask(u64::from(mask)));
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
    // set this again for good measure
    #[allow(deprecated)] // too damn bad
    unsafe {
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    }

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
mod tests;
