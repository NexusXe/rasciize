#![feature(btree_cursors)]
#![feature(map_try_insert)]
#![feature(extend_one)]
#![feature(core_intrinsics)]
#![feature(stdarch_x86_avx512_bf16)] // for bf16 in resize.rs
#![feature(iter_next_chunk)]
#![feature(int_roundings)]
#![feature(portable_simd)]
#![allow(internal_features)] // for core_intrinsics
#![warn(clippy::pedantic)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::used_underscore_items)]
#![allow(clippy::inline_always)]
#![allow(clippy::used_underscore_binding)]

use ab_glyph::Font;
use image::{AnimationDecoder, DynamicImage};
use std::intrinsics::{fadd_fast, fdiv_fast, fmul_fast, fsub_fast};

const FONT_TTF: &[u8] = include_bytes!("./fonts/CascadiaMono-Regular.ttf");
const LUT_FONT_SIZE: u16 = 256; // Size in pixels
const RASTER_SIZE: u16 = 32; // Size in pixels

mod font;
use font::*;
mod output;
use output::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (font, intensity_lookup) = prepare_font::<LUT_FONT_SIZE>(FONT_TTF)?;

    // if the `--text` flag is provided, read from stdin and render the text
    let args: Vec<String> = std::env::args().collect();
    if args.len() <= 1 {
        println!("Usage: {} --text", args[0]);
        println!("Then enter text to render in the terminal.");
        std::process::exit(1)
    }

    match args[1].as_str() {
        "--text" => text(&font, &intensity_lookup),

        "--image" => match args.get(2) {
            None => {
                eprintln!("Please provide an output filename for the image.");
                std::process::exit(1)
            }

            Some(filename) => {
                let spicy = args.contains(&"--spicy".to_string());
                image(&intensity_lookup, filename, spicy)
            }
        },

        _ => {
            eprintln!("Unknown argument: {}", args[1]);
            std::process::exit(1)
        }
    }
}
