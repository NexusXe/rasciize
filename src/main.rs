#![feature(btree_cursors)]
#![feature(map_try_insert)]
#![feature(extend_one)]
#![feature(core_intrinsics)]
#![allow(internal_features)]

use ab_glyph::{Font, FontRef, GlyphId, OutlinedGlyph, PxScale};
use ordered_float::OrderedFloat;
use std::collections::BTreeMap;

const FONT_TTF: &[u8] = include_bytes!("./fonts/NotoSansMono-Regular.ttf");
type FloatPrecision = f64;

fn get_coverage_array(font: &FontRef, font_size: f32, glyph_id: GlyphId) -> Vec<Vec<f32>> {
    match font.outline_glyph(glyph_id.with_scale(PxScale::from(font_size))) {
        Some(outlined_glyph) => {
            let bounds = outlined_glyph.px_bounds();
            let width = bounds.width() as usize;
            let height = bounds.height() as usize;
            let mut coverage_array: Vec<Vec<f32>> = vec![vec![0.0; width]; height];
            outlined_glyph.draw(|x, y, coverage| {
                coverage_array[y as usize][x as usize] = coverage;
            });
            coverage_array
        }
        None => {
            vec![vec![0.0; 1]; 1]
        }
    }
}

fn get_intensity(font: &FontRef, font_size: f32, glyph_id: GlyphId) -> Option<FloatPrecision> {
    use std::intrinsics::{fdiv_fast, fmul_fast};
    let outlined_glyph: OutlinedGlyph =
        font.outline_glyph(glyph_id.with_scale(PxScale::from(font_size)))?;
    let bounds = outlined_glyph.px_bounds();
    let mut sum: u32 = 0;
    outlined_glyph.draw(|_, _, coverage| {
        let scaled_coverage = unsafe { fmul_fast(coverage, 255.0) } as u8;
        sum += scaled_coverage as u32;
    });
    Some(unsafe {
        fdiv_fast(
            sum as FloatPrecision,
            (bounds.width() * bounds.height() * 255.0) as FloatPrecision,
        )
    })
}

pub fn find_nearest_optimized<V>(
    map: &BTreeMap<OrderedFloat<FloatPrecision>, V>,
    target: FloatPrecision,
) -> Option<(&OrderedFloat<FloatPrecision>, &V)> {
    let key = OrderedFloat(target);

    // Find cursor at the first key >= target
    let cursor: std::collections::btree_map::Cursor<'_, OrderedFloat<FloatPrecision>, V> =
        map.lower_bound(std::ops::Bound::Included(&key));

    let ceil = cursor.peek_next(); // The element exactly at or after target
    let floor = cursor.peek_prev(); // The element before that

    match (floor, ceil) {
        (Some(f), Some(c)) => {
            if (key.0 - f.0.0).abs() < (c.0.0 - key.0).abs() {
                Some(f)
            } else {
                Some(c)
            }
        }
        (Some(f), None) => Some(f),
        (None, Some(c)) => Some(c),
        (None, None) => None,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let font = FontRef::try_from_slice(FONT_TTF)?;
    //dbg!(font.glyph_count());
    const FONT_SIZE: usize = 32; // Size in pixels

    let mut intensity_lookup: BTreeMap<OrderedFloat<FloatPrecision>, char> = BTreeMap::new();
    // manually insert space character with intensity 0.0
    intensity_lookup.insert(OrderedFloat(0.0), ' ');
    // manually insert full block character with intensity 1.0
    //intensity_lookup.insert(OrderedFloat(1.0), '█');

    //let mut collisions: u16 = 0;
    for (id, character) in font.codepoint_ids() {
        //println!("{id:?} -> {character}");
        if let Some(mut intensity) = get_intensity(&font, FONT_SIZE as f32, id) {
            while intensity_lookup
                .try_insert(OrderedFloat(intensity), character)
                .is_err()
            {
                //break;
                // In case of collision, slightly adjust intensity
                intensity = intensity.next_up();
                //collisions += 1;
            }
        }
    }

    //println!("Total collisions resolved: {}", collisions);
    //dbg!(&intensity_lookup);
    const INPUT: &str = "h-d";
    let mut output: Vec<Vec<char>> = Vec::new();
    for c in INPUT.chars() {
        let coverage_array = get_coverage_array(&font, FONT_SIZE as f32, font.glyph_id(c));
        let mut char_rows: Vec<Vec<char>> = vec![Vec::new(); coverage_array.len()];
        for (y, row) in coverage_array.iter().enumerate() {
            for &coverage in row {
                if let Some((_, &ch)) =
                    find_nearest_optimized(&intensity_lookup, coverage as FloatPrecision)
                {
                    char_rows[y].push(ch);
                } else {
                    char_rows[y].push(' ');
                }
            }
        }
        // Append char_rows to output
        for (y, row) in char_rows.into_iter().enumerate() {
            if output.len() <= y {
                output.push(row);
            } else {
                output[y].extend(row);
            }
        }
    }

    for row in output {
        for ch in row {
            print!("{}", ch.to_string().repeat(4));
        }
        println!();
    }
    // let example_coverage = get_coverage_array(&font, FONT_SIZE as f32, font.glyph_id('g'));
    // for row in example_coverage {
    //     for coverage in row {
    //         print!("{}", find_nearest_optimized(&intensity_lookup, coverage as FloatPrecision).unwrap().1.to_string().repeat(4))
    //     }
    //     println!();
    // }
    Ok(())
}
