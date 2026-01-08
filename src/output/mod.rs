use crate::*;
use ab_glyph::FontRef;
use image::{ImageReader, Rgb};
use std::arch::x86_64::*;
use std::intrinsics::prefetch_read_data;
use std::io::{self, Write};
use std::thread::sleep;
use std::time::Duration;

mod lanczos;

const IMG_SIZE_MAX: u32 = 64;

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
    print!("\x1b[38;2;243;123;33m");

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

    print!("\x1b[0m"); // Reset terminal formatting
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
    let start_time = std::time::Instant::now();
    let mut frames = Vec::new();
    if let Some(image::ImageFormat::Gif) = input.format() {
        // crazy shuffling between DynamicImage is fine because it's done once
        let gif_decoder = image::codecs::gif::GifDecoder::new(input.into_inner())?;
        let gif_frames = gif_decoder.into_frames();

        #[cfg(feature = "progress")]
        {
            eprint!("\x1b[A\x1b[2K"); // move up 1 line for progress output and clear it
        }

        #[allow(clippy::unused_enumerate_index)]
        for (_idx, frame) in gif_frames.enumerate() {
            #[cfg(feature = "progress")]
            {
                eprintln!("Decoded {_idx:} frame{}", if _idx <= 1 { "" } else { "s" });
                eprint!("\x1b[A"); // move up 1 line for progress output
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
            eprint!("\x1b[A"); // move up 1 line for progress output
        }
    }

    let total_frame_count = frames.len();

    #[cfg(feature = "progress")]
    {
        eprint!("\x1b[2K"); // clear the line
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
    eprint!("\x1b[2K"); // clear the line

    #[allow(clippy::unused_enumerate_index)]
    for (frame_number, this_frame) in frames.iter().enumerate() {
        #[cfg(feature = "progress")]
        {
            eprintln!(
                "Rasciizing frame {:}/{total_frame_count:}",
                frame_number + 1
            );
            eprint!("\x1b[A"); // move up 1 line for progress output
        }

        let img = this_frame.to_rgb8();

        // row by row, pixel by pixel, get the luminance, find the nearest character, and print it to output using true colors obtained from maximize function
        let mut output_buffer =
            String::with_capacity(((this_frame.width() + 10) * this_frame.height()) as usize);

        #[cfg(not(feature = "color"))]
        {
            output_buffer.push_str("\x1B[37m");
        }

        for y in 0..img.height() {
            for x in 0..img.width() {
                let pixel = img.get_pixel(x, y);
                let lum = luminance(pixel[0], pixel[1], pixel[2]);
                let mut lum_scaled = unsafe { fdiv_fast(lum, 255.0) };
                if spicy {
                    // slightly randomize
                    let seed_data: u64 = (u64::from(pixel[0])
                        | u64::from(pixel[1]) << 8
                        | u64::from(pixel[2]) << 16
                        | u64::from(y) << 24
                        | u64::from(x) << 32
                        | (frame_number as u64) << 40)
                        ^ (u64::from(lum.to_bits()).rotate_left(40));

                    let random_value: f32 = {
                        #[cfg(target_feature = "sse4.2")]
                        unsafe {
                            let hash = std::arch::x86_64::_mm_crc32_u64(0x045d_9f3b, seed_data);
                            let bits = ((hash as u32) & 0x007f_ffff) | 0x3f80_0000; // 0x3f80_0000 is 1.0 in f32
                            let res = f32::from_bits(bits);
                            fsub_fast(res, 1.5)
                        }
                        #[cfg(not(target_feature = "sse4.2"))]
                        0.0
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
                #[cfg(feature = "color")]
                {
                    let color = maximize(*pixel);
                    write!(
                        output_buffer,
                        "\x1b[38;2;{};{};{}m{ch}",
                        color[0], color[1], color[2]
                    )?;
                }
                #[cfg(not(feature = "color"))]
                {
                    write!(output_buffer, "{}", ch)?;
                }
            }
            output_buffer.push('\n');
        }

        // Delete next line
        //output_buffer.push_str("\x1b[B\x1b[M\x1b[A");

        if frames.len() > 1 {
            output_buffer.push_str(format!("\x1b[{height}A").as_str()); // move up N lines
        }

        output_frames.push(output_buffer);
    }
    output_frames.shrink_to_fit();
    let end_time = std::time::Instant::now();
    eprintln!(
        "Rendering took {:} seconds",
        end_time.duration_since(start_time).as_secs_f32()
    );

    let stdout = io::stdout();
    let mut stdout_handle = io::BufWriter::new(stdout);

    stdout_handle.write_all(output_frames[0].as_bytes())?;

    if output_frames.len() == 1 {
        stdout_handle.flush()?;
    } else {
        loop {
            for (idx, output_frame) in output_frames.iter().enumerate() {
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
