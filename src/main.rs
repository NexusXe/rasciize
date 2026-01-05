#![feature(btree_cursors)]
#![feature(map_try_insert)]
#![feature(extend_one)]
#![feature(core_intrinsics)]
#![allow(internal_features)]
#![warn(clippy::pedantic)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::wildcard_imports)]

use ab_glyph::{Font, FontRef};
use image::{AnimationDecoder, DynamicImage};
use ordered_float::OrderedFloat;
use std::{
    collections::BTreeMap,
    intrinsics::{fadd_fast, fmul_fast, fdiv_fast},
    io::Write,
};

const FONT_TTF: &[u8] = include_bytes!("./fonts/CascadiaMono-Regular.ttf");

mod font;
use font::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const LUT_FONT_SIZE: u16 = 256; // Size in pixels
    const RASTER_SIZE: u16 = 32; // Size in pixels
    let font = FontRef::try_from_slice(FONT_TTF)?;

    let mut intensity_lookup = BTreeMap::new();
    // manually insert space character with intensity 0.0
    intensity_lookup.insert(OrderedFloat(0.0), ' ');
    // manually insert full block character with intensity 1.0
    intensity_lookup.insert(OrderedFloat(1.0), '█');

    for (id, character) in font.codepoint_ids() {
        // let omitted_ranges = [
        //     '\u{1FB00}'..='\u{1FBFF}', // Symbols for Legacy Computing
        //     '\u{1CC00}'..='\u{1CEBF}', // Symbols for Legacy Computing Supplement
        //     '\u{1FA00}'..='\u{1FA6F}', // Chess Symbols
        //     '\u{2587}'..='\u{2590}',   // Block Elements (specifically horizontally-partial blocks)
        // ];

        // let omitted_characters = ['\u{2502}', '\u{2503}', '\u{2506}', '\u{2507}', '\u{250A}', '\u{250B}', '▆','█', '◾', '\u{AD}']; // Box Drawing Characters (specifically vertical lines)
        // //debug_assert!(omitted_characters.is_sorted());
        // let omitted_str: &'static str = "██▅▅██▖▖██▄▄██╹╹██◼◼██⬛⬛██▬▬∙¦┉--—▕▕◖◖ׂׂ₅₅♥♥╼׀׀●●ׁׁ══··╾╾╾╾ׁׁ␣␣⬢⬢вв║║ْْ▭▭▰▰̄̄\u{AD}\u{AD}――▮▮ــ۔۔۔۔۔۔۔۔۔۔۔۔۔۔̲̲──ַַַַַַַַַַַַַַ––▪━━╴╴..⁸⁸▣▣▣▣⁸⁸▯▯₃₃шш˙˙▩▩█▃‌▃‌‌";
        // if omitted_ranges.iter().any(|range| range.contains(&character)) || omitted_characters.contains(&character) || omitted_str.contains(character) {
        //     continue;
        // }
        if let Some(mut intensity) = get_intensity(&font, f32::from(LUT_FONT_SIZE), id) {
            while intensity_lookup
                .try_insert(OrderedFloat(intensity), character)
                .is_err()
            {
                intensity = intensity.next_up();
            }
        }
    }

    // if the `--text` flag is provided, read from stdin and render the text
    let args: Vec<String> = std::env::args().collect();
    if args.len() <= 1 {
        println!("Usage: {} --text", args[0]);
        println!("Then enter text to render in the terminal.");
        std::process::exit(1)
    }

    match args[1].as_str() {
        "--text" => {
            let mut input = String::new();
            print!("Enter text to render >> ");
            std::io::stdout().flush()?;
            std::io::stdin().read_line(&mut input)?;
            let input = input.trim_end(); // remove trailing newline
            println!();

            let mut output: Vec<Vec<Vec<char>>> = Vec::new();
            for c in input.chars() {
                let Some(coverage_array) = get_coverage_array(&font, RASTER_SIZE, font.glyph_id(c))
                else {
                    continue;
                };

                let mut char_array: Vec<Vec<char>> = Vec::new();
                for row in coverage_array {
                    let mut char_row: Vec<char> = Vec::new();
                    for coverage in row {
                        let ch = find_nearest_optimized(
                            &intensity_lookup,
                            FloatPrecision::from(coverage),
                        )
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

            let r = 243;
            let g = 123;
            let b = 33;
            let reset = "\x1b[0m"; // Code to reset terminal formatting

            // The format string constructs the ANSI escape code sequence
            print!("\x1b[38;2;{r};{g};{b}m");

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

            print!("{reset}"); // Reset terminal formatting
            std::io::stdout().flush()?;

            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            Ok(())
        }

        "--image" => match args.get(2) {
            None => {
                eprintln!("Please provide an output filename for the image.");
                std::process::exit(1)
            }

            Some(filename) => {
                use image::{ImageReader, Rgb};
                use std::fmt::Write;

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
                    let max_component =
                        FloatPrecision::from(*[color[0], color[1], color[2]].iter().max().unwrap());
                    if max_component == 0.0 {
                        return Rgb([0, 0, 0]);
                    }
                    let scale = 255.0 / max_component;
                    let maximum_r = unsafe { fmul_fast(FloatPrecision::from(color[0]), scale) };
                    let maximum_g = unsafe { fmul_fast(FloatPrecision::from(color[1]), scale) };
                    let maximum_b = unsafe { fmul_fast(FloatPrecision::from(color[2]), scale) };
                    let half_r = unsafe { fdiv_fast(fadd_fast(maximum_r, FloatPrecision::from(color[0])), 2.0) };
                    let half_g = unsafe { fdiv_fast(fadd_fast(maximum_g, FloatPrecision::from(color[1])), 2.0) };
                    let half_b = unsafe { fdiv_fast(fadd_fast(maximum_b, FloatPrecision::from(color[2])), 2.0) };

                    Rgb([half_r as u8, half_g as u8, half_b as u8])

                }
                let input = ImageReader::open(filename)?;

                let mut frames = Vec::new();
                match input.format() {
                    Some(image::ImageFormat::Gif) => {
                        let gif_decoder = image::codecs::gif::GifDecoder::new(input.into_inner())?;
                        let gif_frames = gif_decoder.into_frames();
                        for frame in gif_frames {
                            frames.push(DynamicImage::ImageRgb8(
                                DynamicImage::ImageRgba8(frame?.into_buffer()).to_rgb8(),
                            ));
                        }
                    }
                    _ => {
                        frames.push(input.decode()?);
                    }
                }

                while frames.len() > 1 {
                    for frame_number in 0..frames.len() {
                        let mut this_frame = frames[frame_number].clone();
                        const IMG_SIZE_MAX: u32 = 64;
                        if this_frame.width() > IMG_SIZE_MAX * 2
                            || this_frame.height() > IMG_SIZE_MAX * 2
                        {
                            this_frame = this_frame.resize(
                                IMG_SIZE_MAX,
                                IMG_SIZE_MAX,
                                image::imageops::FilterType::Lanczos3,
                            );
                        }
                        let wide_img = this_frame.resize_exact(
                            this_frame.width() * 2,
                            this_frame.height(),
                            image::imageops::FilterType::Nearest,
                        );
                        let img = wide_img.to_rgb8();

                        // row by row, pixel by pixel, get the luminance, find the nearest character, and print it to output using true colors obtained from maximize function
                        let mut output_buffer = String::with_capacity(
                            ((wide_img.width() + 10) * wide_img.height()) as usize,
                        );
                        for y in 0..img.height() {
                            for x in 0..img.width() {
                                let pixel = img.get_pixel(x, y);
                                let mut lum = luminance(pixel[0], pixel[1], pixel[2]) / 255.0;
                                if args.contains(&"--spicy".to_owned()) {
                                    // slightly randomize
                                    lum += (rand::random::<f64>() - 0.5) * 0.001;
                                }
                                let ch = find_nearest_optimized(&intensity_lookup, lum).unwrap().1;
                                let color = maximize(*pixel);
                                //let color = *pixel;
                                write!(
                                    output_buffer,
                                    "\x1b[38;2;{};{};{}m{ch}",
                                    color[0], color[1], color[2]
                                )?;
                            }
                            output_buffer.push('\n');
                        }
                        output_buffer.push_str("\x1b[0m"); // Reset terminal formatting

                        print!("{output_buffer}");
                        // print!("\x1b[{N}A"); // move up N lines
                        std::io::stdout().flush()?;
                        if frames.len() > 1 {
                            std::thread::sleep(std::time::Duration::from_millis(10));
                            print!("\x1b[{}A", wide_img.height()); // move up N lines
                        }
                    }
                }
                Ok(())
            }
        },

        _ => {
            eprintln!("Unknown argument: {}", args[1]);
            std::process::exit(1)
        }
    }
}
