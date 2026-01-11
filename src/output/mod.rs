use crate::*;
use ab_glyph::{Font, FontRef};
use image::{ImageReader, Rgb};
use std::arch::x86_64::*;
use std::intrinsics::prefetch_read_data;
use std::io::{self, Write};
use std::thread::sleep;

use std::time::{Duration, Instant};

mod lanczos;

const IMG_SIZE_MAX: u32 = 256;
const ESCAPE: char = '\x1b';

pub fn text(
    font: &FontRef<'static>,
    intensity_lookup: &IntensityMap<char>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::new();
    print!("Enter text to render >> ");
    io::stdout().flush()?;
    io::stdin().read_line(&mut input)?;
    let input = input.trim_end(); // remove trailing newline
    println!();

    let mut output: Vec<Vec<Vec<char>>> = Vec::new();
    for c in input.chars() {
        let Some(coverage_array) = get_coverage_array(font, RASTER_SIZE, font.glyph_id(c)) else {
            continue;
        };

        let mut char_array: Vec<Vec<char>> = Vec::new();
        for row in coverage_array {
            let mut char_row: Vec<char> = Vec::new();
            for coverage in row {
                let ch = find_nearest_optimized(intensity_lookup, FloatPrecision::from(coverage))
                    .unwrap()
                    .1;
                char_row.push(*ch);
            }
            char_array.push(char_row);
        }
        // pad each row to have the same width for this character
        let max_width = char_array.iter().map(std::vec::Vec::len).max().unwrap_or(0);
        for row in &mut char_array {
            while row.len() < max_width {
                row.push(' ');
            }
        }

        output.push(char_array);
    }

    // Set the text color to a warm orange
    print!("{ESCAPE}[38;2;243;123;33m");

    // Now print the output row by row. Some characters won't have as many rows as others, so use checked indexing.
    let max_height = output.iter().map(std::vec::Vec::len).max().unwrap_or(0);
    for row_idx in 0..max_height {
        for (char_idx, char_array) in output.iter().enumerate() {
            if let Some(char_row) = char_array.get(row_idx) {
                for ch in char_row {
                    print!("{ch}");
                }
            } else {
                // Print spaces for missing rows
                for _ in 0..char_array[0].len() {
                    print!(" ");
                }
            }
            // space between characters except for the last one
            if char_idx < output.len() - 1 {
                print!(" ");
            }
        }
        println!();
    }

    print!("{ESCAPE}[0m"); // Reset terminal formatting
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(())
}

#[inline]
fn luminance(r: u8, g: u8, b: u8) -> FloatPrecision {
    let r = u16::from(r);
    let g = u16::from(g);
    let b = u16::from(b);
    // Percieved luminance based on HSP (https://alienryderflex.com/hsp.html)
    // luminance = sqrt( 0.299*R^2 + 0.587*G^2 + 0.114*B^2 )
    FloatPrecision::sqrt(unsafe {
        fadd_fast(
            fadd_fast(
                fmul_fast(0.299, FloatPrecision::from(r * r)),
                fmul_fast(0.587, FloatPrecision::from(g * g)),
            ),
            fmul_fast(0.114, FloatPrecision::from(b * b)),
        )
    })
}

#[inline]
fn maximize(color: Rgb<u8>) -> Rgb<u8> {
    let max_component = FloatPrecision::from(*[color[0], color[1], color[2]].iter().max().unwrap());
    if max_component == 0.0 {
        return Rgb([0, 0, 0]);
    }
    let scale = unsafe { fdiv_fast(255.0, max_component) };
    let maximum_r = unsafe { fmul_fast(FloatPrecision::from(color[0]), scale) };
    let maximum_g = unsafe { fmul_fast(FloatPrecision::from(color[1]), scale) };
    let maximum_b = unsafe { fmul_fast(FloatPrecision::from(color[2]), scale) };
    //Rgb([maximum_r as u8, maximum_g as u8, maximum_b as u8])
    let half_r = unsafe { fdiv_fast(fadd_fast(maximum_r, FloatPrecision::from(color[0])), 2.0) };
    let half_g = unsafe { fdiv_fast(fadd_fast(maximum_g, FloatPrecision::from(color[1])), 2.0) };
    let half_b = unsafe { fdiv_fast(fadd_fast(maximum_b, FloatPrecision::from(color[2])), 2.0) };

    Rgb([half_r as u8, half_g as u8, half_b as u8])
}

// too many lines? too bad!
#[allow(clippy::too_many_lines)]
#[inline(always)]
pub fn image(
    intensity_lookup: &IntensityMap<char>,
    filename: &str,
    spicy: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fmt::Write;

    #[cfg(feature = "progress")]
    {
        eprintln!("\nRanked {:} characters", intensity_lookup.len());
    }

    // nuclear option: set mxcsr to flush all denormals to zero
    // using embedded rounding modes causes extra load instructions to be generated
    // since you can't use embedded rounding modes with memory sources
    #[allow(deprecated)] // too damn bad
    unsafe {
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    }

    let input = ImageReader::open(filename)?;

    let _start_time: Instant;

    #[cfg(feature = "progress")]
    {
        _start_time = std::time::Instant::now();
    }

    let mut frames = Vec::new();
    if let Some(image::ImageFormat::Gif) = input.format() {
        // crazy shuffling between DynamicImage is fine because it's done once
        let gif_decoder = image::codecs::gif::GifDecoder::new(input.into_inner())?;
        let gif_frames = gif_decoder.into_frames();

        #[cfg(feature = "progress")]
        {
            eprint!("{ESCAPE}[A{ESCAPE}[2K"); // move up 1 line for progress output and clear it
        }

        #[allow(clippy::unused_enumerate_index)]
        for (_idx, frame) in gif_frames.enumerate() {
            #[cfg(feature = "progress")]
            {
                eprintln!("Decoded {_idx:} frame{}", if _idx <= 1 { "" } else { "s" });
                eprint!("{ESCAPE}[A"); // move up 1 line for progress output
            }

            frames.push(DynamicImage::ImageRgb8(
                DynamicImage::ImageRgba8(frame?.into_buffer()).to_rgb8(),
            ));
        }
        frames.shrink_to_fit();
    } else {
        frames.push(input.decode()?);
        #[cfg(feature = "progress")]
        {
            eprintln!("Decoded 1 frame");
            eprint!("{ESCAPE}[A"); // move up 1 line for progress output
        }
    }

    let total_frame_count = frames.len();

    #[cfg(feature = "progress")]
    {
        eprint!("{ESCAPE}[2K"); // clear the line
    }
    let mut new_width = frames[0].width() * 2;
    let mut new_height = frames[0].height();

    if new_width > IMG_SIZE_MAX * 2 || new_height > IMG_SIZE_MAX * 2 {
        let max_dim = u64::from(IMG_SIZE_MAX);
        let w = u64::from(new_width);
        let h = u64::from(new_height);

        // Preserving aspect ratio fitting within IMG_SIZE_MAX x IMG_SIZE_MAX
        if h <= w {
            // Bound by width
            new_width = IMG_SIZE_MAX;
            new_height = (h * max_dim / w) as u32;
        } else {
            // Bound by height
            new_height = IMG_SIZE_MAX;
            new_width = (w * max_dim / h) as u32;
        }
    }
    // double the width of each frame to account for character aspect ratio
    let horizontal_filters = lanczos::precompute_weights(frames[0].width(), new_width);
    let vertical_filters = lanczos::precompute_weights(frames[0].height(), new_height);

    std::thread::scope(|s| {
        // resizing loop
        for frame in &mut frames {
            let horizontal_filters = &horizontal_filters;
            let vertical_filters = &vertical_filters;
            s.spawn(move || {
                //*frame = frame.resize_exact(new_width, new_height, image::imageops::FilterType::Lanczos3);
                *frame = lanczos::lanczos3_resize(
                    frame,
                    new_width,
                    new_height,
                    horizontal_filters,
                    vertical_filters,
                );
            });
        }
    });

    let height = frames[0].height();

    let mut output_frames: Vec<String> = Vec::with_capacity(total_frame_count);
    eprint!("{ESCAPE}[2K"); // clear the line

    #[allow(clippy::unused_enumerate_index)]
    output_frames.resize(total_frame_count, String::new());

    #[cfg(feature = "progress")]
    {
        eprintln!("Rasciizing frames...");
        eprint!("{ESCAPE}[A");
    }

    let frames_ref = &frames;
    std::thread::scope(|s| {
        for (frame_number, (this_frame, output_frame_slot)) in
            frames.iter().zip(output_frames.iter_mut()).enumerate()
        {
            let intensity_lookup = &intensity_lookup;
            s.spawn(move || {
                #[cfg(feature = "progress")]
                {
                    eprintln!(
                        "Rasciizing frame {:}/{total_frame_count:}",
                        frame_number + 1
                    );
                    eprint!("{ESCAPE}[A"); // move up 1 line for progress output
                }
                let img = this_frame.to_rgb8();
                let prev_img = if frame_number > 0 {
                    Some(frames_ref[frame_number - 1].to_rgb8())
                } else {
                    None
                };

                // row by row, pixel by pixel, get the luminance, find the nearest character, and print it to output using true colors obtained from maximize function
                let mut output_buffer = String::with_capacity(
                    ((this_frame.width() + 10) * this_frame.height()) as usize,
                );

                #[cfg(not(feature = "color"))]
                {
                    output_buffer.push_str("{ESCAPE}[37m");
                }

                let get_pixel_info =
                    |x: u32, y: u32, p: &Rgb<u8>, f_num: usize| -> (char, Rgb<u8>) {
                        let sc_r = p[0];
                        let sc_g = p[1];
                        let sc_b = p[2];
                        let lum = luminance(sc_r, sc_g, sc_b);
                        let mut lum_scaled = unsafe { fdiv_fast(lum, 255.0) };

                        if spicy {
                            // slightly randomize
                            let seed_data: u64 = u64::from(x)
                                ^ u64::from(y).rotate_left(16)
                                ^ (f_num as u64).rotate_left(32)
                                ^ u64::from(sc_r)
                                ^ u64::from(sc_g).rotate_left(8)
                                ^ u64::from(sc_b).rotate_left(24)
                                ^ u64::from(lum.to_bits()).rotate_left(48); // a little bit of nonlinearity from the sqrt function in luminance()

                            let random_value: f32 = {
                                const DOMAIN_CONST: u32 = 0x045d_9f3b;

                                #[inline(always)]
                                fn murmur3_64to32_seeded<const SEED: u32>(data: u64) -> u32 {
                                    let mut h: u32 = (data as u32) ^ (data >> 32) as u32;

                                    h ^= SEED;

                                    h ^= h >> 16;
                                    h *= 0x7feb_352d;
                                    h ^= h >> 15;
                                    h *= 0x846c_a68b;
                                    h ^= h >> 16;

                                    h
                                }

                                let hash = {
                                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                                    if is_x86_feature_detected!("sse4.2") {
                                        unsafe {
                                            std::arch::x86_64::_mm_crc32_u64(
                                                u64::from(DOMAIN_CONST),
                                                seed_data,
                                            ) as u32
                                        }
                                    } else {
                                        murmur3_64to32_seeded::<DOMAIN_CONST>(seed_data)
                                    }

                                    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                                    murmur3_64to32_seeded::<DOMAIN_CONST>(seed_data)
                                };

                                let bits = (hash & 0x007f_ffff) | 0x3f80_0000; // 0x3f80_0000 is 1.0 in f32
                                let res = f32::from_bits(bits);
                                unsafe { fsub_fast(res, 1.5) }
                            };

                            lum_scaled = unsafe {
                                fadd_fast(
                                    lum_scaled,
                                    fmul_fast(FloatPrecision::from(random_value), 0.0001),
                                )
                            };
                        }
                        let ch = find_nearest_optimized(intensity_lookup, lum_scaled)
                            .unwrap()
                            .1;

                        let color = maximize(*p);
                        (*ch, color)
                    };

                #[cfg(feature = "truecolor")]
                let mut last_color: Option<Rgb<u8>> = None;

                #[cfg(all(not(feature = "truecolor"), feature = "color"))]
                let mut last_ansi_color: Option<u8> = None;

                for y in 0..img.height() {
                    let mut skip_count = 0;
                    for x in 0..img.width() {
                        let pixel = img.get_pixel(x, y);
                        let (ch, color) = get_pixel_info(x, y, pixel, frame_number);

                        let mut is_redundant = false;
                        if let Some(ref p_img) = prev_img {
                            let prev_pixel = p_img.get_pixel(x, y);
                            let (prev_ch, prev_color) =
                                get_pixel_info(x, y, prev_pixel, frame_number - 1);

                            if ch == prev_ch {
                                #[cfg(feature = "truecolor")]
                                {
                                    if color == prev_color {
                                        is_redundant = true;
                                    }
                                }
                                #[cfg(all(not(feature = "truecolor"), feature = "color"))]
                                {
                                    let get_16_col = |c: Rgb<u8>| {
                                        let r = if c[0] > 127 { 1 } else { 0 };
                                        let g = if c[1] > 127 { 2 } else { 0 };
                                        let b = if c[2] > 127 { 4 } else { 0 };
                                        let i = if c[0] > 200 || c[1] > 200 || c[2] > 200 {
                                            60
                                        } else {
                                            0
                                        };
                                        30 + r + g + b + i
                                    };
                                    if get_16_col(color) == get_16_col(prev_color) {
                                        is_redundant = true;
                                    }
                                }
                                #[cfg(not(feature = "color"))]
                                {
                                    is_redundant = true;
                                }
                            }
                        }

                        if is_redundant {
                            skip_count += 1;
                        } else {
                            if skip_count > 0 {
                                write!(output_buffer, "{ESCAPE}[{skip_count}C").unwrap();
                                skip_count = 0;
                            }

                            #[cfg(feature = "truecolor")]
                            {
                                if last_color != Some(color) {
                                    write!(
                                        output_buffer,
                                        "{ESCAPE}[38;2;{};{};{}m",
                                        color[0], color[1], color[2]
                                    )
                                    .unwrap();
                                    last_color = Some(color);
                                }
                                write!(output_buffer, "{ch}").unwrap();
                            }

                            #[cfg(not(feature = "color"))]
                            {
                                write!(output_buffer, "{ch}").unwrap();
                            }

                            #[cfg(all(not(feature = "truecolor"), feature = "color"))]
                            {
                                // map to closest 16 color palette color
                                let r = if color[0] > 127 { 1 } else { 0 };
                                let g = if color[1] > 127 { 2 } else { 0 };
                                let b = if color[2] > 127 { 4 } else { 0 };
                                let intensity =
                                    if color[0] > 200 || color[1] > 200 || color[2] > 200 {
                                        60
                                    } else {
                                        0
                                    };
                                let ansi_code = 30 + r + g + b + intensity;

                                if last_ansi_color != Some(ansi_code) {
                                    write!(output_buffer, "{ESCAPE}[{}m", ansi_code).unwrap();
                                    last_ansi_color = Some(ansi_code);
                                }
                                write!(output_buffer, "{}", ch).unwrap();
                            }
                        }
                    }
                    #[cfg(all(not(feature = "truecolor"), feature = "color"))]
                    {
                        // reset color
                        write!(output_buffer, "{ESCAPE}[0m").unwrap();
                    }
                    output_buffer.push('\n');
                }

                // Delete next line
                //output_buffer.push_str("{ESCAPE}[B{ESCAPE}[M{ESCAPE}[A");

                if total_frame_count > 1 {
                    output_buffer.push_str(format!("{ESCAPE}[{height}A").as_str()); // move up N lines
                }

                *output_frame_slot = output_buffer;
                output_frame_slot.shrink_to_fit();
            });
        }
    });
    output_frames.shrink_to_fit();

    #[cfg(feature = "progress")]
    {
        let end_time = std::time::Instant::now();

        eprintln!(
            "Rendering took {:} seconds",
            end_time.duration_since(_start_time).as_secs_f32()
        );

        #[cfg(debug_assertions)]
        {
            // print total size of output_frames
            let total_size = output_frames
                .iter()
                .map(std::string::String::len)
                .sum::<usize>();
            eprintln!("Total size of output_frames: {total_size} bytes");
        }
    }

    let stdout = io::stdout();
    let mut stdout_handle = io::BufWriter::new(stdout);

    let mut looped = false;
    stdout_handle.write_all(output_frames[0].as_bytes())?;

    if output_frames.len() == 1 {
        stdout_handle.flush()?;
    } else {
        loop {
            for (idx, output_frame) in output_frames.iter().enumerate().skip(usize::from(!looped)) {
                looped = true;
                stdout_handle.flush()?;
                stdout_handle.write_all(output_frame.as_bytes())?;
                prefetch_read_data::<_, 2>(
                    output_frames[if idx + 1 < output_frames.len() {
                        idx + 1
                    } else {
                        0
                    }]
                    .as_ptr(),
                );
                sleep(Duration::from_millis(15));
            }
        }
    }
    Ok(())
}
