use std::simd::prelude::*;

const DOMAIN_CONST: u32 = 0x045d_9f3b;

#[inline(never)]
fn murmur3_128to32_seeded<const SEED: u32>(data: u32x4) -> u32 {
    let data = data.as_array();
    let mut h: u32 = data[0] ^ data[1] ^ data[2] ^ data[3];

    h ^= SEED;

    h ^= h >> 16;
    h *= 0x7feb_352d;
    h ^= h >> 15;
    h *= 0x846c_a68b;
    h ^= h >> 16;

    h
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "sha")]
unsafe fn x86_sha256_128to32(data: u32x4) -> u32 {
    use std::arch::x86_64::*;
    let hash = u32x4::from(_mm_sha256msg1_epu32(
        _mm_setzero_si128(),
        __m128i::from(data),
    ));
    hash.as_array()[0]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn x86_crc32_128to32<const SEED: u32>(data: u32x4) -> u32 {
    use std::arch::x86_64::*;

    let data = unsafe { std::mem::transmute::<u32x4, u64x2>(data) };
    let h: u64 = data[0] ^ data[1];
    _mm_crc32_u64(u64::from(SEED), h) as u32
}
    

#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")] // ARM crc32
unsafe fn arm_crc32_128to32<const SEED: u32>(data: u32x4) -> u32 {
    use std::arch::aarch64::*;

    let data = unsafe { std::mem::transmute::<u32x4, u64x2>(data) };
    let h: u64 = data[0] ^ data[1];
    __crc32cd(SEED, h)
}

#[inline(never)]
pub fn hash(input: u32x4) -> u32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sha") {
        unsafe { x86_sha256_128to32(input) }
    } else if is_x86_feature_detected!("sse4.2") {
        unsafe { x86_crc32_128to32::<DOMAIN_CONST>(input) }
    } else {
        murmur3_128to32_seeded::<DOMAIN_CONST>(input)
    }
    
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("crc") {
        unsafe { arm_crc32_128to32::<DOMAIN_CONST>(input) }
    } else {
        murmur3_128to32_seeded::<DOMAIN_CONST>(input)
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    murmur3_128to32_seeded::<DOMAIN_CONST>(input)
}
