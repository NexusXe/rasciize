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
use ordered_float::OrderedFloat;
use std::{collections::BTreeMap, io::Write};

const FONT_TTF: &[u8] = include_bytes!("./fonts/CascadiaMono-Regular.ttf");

mod font;
use font::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const LUT_FONT_SIZE: u16 = 1024; // Size in pixels
    const RASTER_SIZE: u16 = 32; // Size in pixels
    let font = FontRef::try_from_slice(FONT_TTF)?;

    let mut intensity_lookup = BTreeMap::new();
    // manually insert space character with intensity 0.0
    intensity_lookup.insert(OrderedFloat(0.0), ' ');
    // manually insert full block character with intensity 1.0
    intensity_lookup.insert(OrderedFloat(1.0), '█');

    for (id, character) in font.codepoint_ids() {
        if let Some(mut intensity) = get_intensity(&font, f32::from(LUT_FONT_SIZE), id) {
            while intensity_lookup
                .try_insert(OrderedFloat(intensity), character)
                .is_err()
            {
                intensity = intensity.next_up();
            }
        }
    }

    let mut input = String::new();
    print!("Enter text to render >> ");
    std::io::stdout().flush()?;
    std::io::stdin().read_line(&mut input)?;
    let input = input.trim_end(); // remove trailing newline
    println!();

    let mut output: Vec<Vec<Vec<char>>> = Vec::new();
    for c in input.chars() {
        let Some(coverage_array) = get_coverage_array(&font, RASTER_SIZE, font.glyph_id(c)) else {
            continue;
        };

        let mut char_array: Vec<Vec<char>> = Vec::new();
        for row in coverage_array {
            let mut char_row: Vec<char> = Vec::new();
            for coverage in row {
                let ch = find_nearest_optimized(&intensity_lookup, FloatPrecision::from(coverage))
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
