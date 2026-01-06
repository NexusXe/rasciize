use ab_glyph::{Font, FontRef, GlyphId, OutlinedGlyph, PxScale};
use ordered_float::OrderedFloat;
use std::{
    collections::BTreeMap,
    intrinsics::{fdiv_fast, fmul_fast, fsub_fast},
};

pub type FloatPrecision = f32;
pub type IntensityMap<V> = BTreeMap<OrderedFloat<FloatPrecision>, V>;

#[inline]
pub fn get_coverage_array(
    font: &FontRef,
    font_size: u16,
    glyph_id: GlyphId,
) -> Option<Vec<Vec<f32>>> {
    // initially get a coverage array at 4x scale and only keep every fourth row to get a 4x wide character for printing in a terminal
    let font_size = f32::from(font_size * 4);
    match font.outline_glyph(glyph_id.with_scale(PxScale::from(font_size))) {
        Some(outlined_glyph) => {
            let bounds = outlined_glyph.px_bounds();
            let mut coverage_array: Vec<Vec<f32>> =
                vec![vec![0.0; bounds.width() as usize]; bounds.height() as usize];
            outlined_glyph.draw(|x, y, coverage| {
                coverage_array[y as usize][x as usize] = coverage;
            });
            // Now keep only every fourth row to get 4x wide characters
            let coverage_array: Vec<Vec<f32>> = coverage_array
                .into_iter()
                .enumerate()
                .filter_map(|(i, row)| if i % 4 == 0 { Some(row) } else { None })
                .collect();
            Some(coverage_array)
        }
        None => None,
    }
}

#[inline]
pub fn get_intensity(font: &FontRef, font_size: f32, glyph_id: GlyphId) -> Option<FloatPrecision> {
    // horrific hack to try and filter out weird glyphs that aren't really characters and might not show up as monospaced
    let standard_glyph_width_wide = font
        .outline_glyph(font.glyph_id('M').with_scale(PxScale::from(font_size)))?
        .px_bounds()
        .width();
    let standard_glyph_width_narrow = font
        .outline_glyph(font.glyph_id('i').with_scale(PxScale::from(font_size)))?
        .px_bounds()
        .width();
    let outlined_glyph: OutlinedGlyph =
        font.outline_glyph(glyph_id.with_scale(PxScale::from(font_size)))?;
    let bounds = outlined_glyph.px_bounds();
    if unsafe {
        fsub_fast(bounds.width(), standard_glyph_width_wide).abs() > font_size.sqrt()
            && (fsub_fast(bounds.width(), standard_glyph_width_narrow).abs() > font_size.sqrt())
    } {
        return None;
    }
    let mut sum: u32 = 0;
    outlined_glyph.draw(|_, _, coverage| {
        let scaled_coverage = unsafe { fmul_fast(coverage, 255.0) } as u8;
        sum += u32::from(scaled_coverage);
    });

    Some(
        #[allow(clippy::cast_lossless, clippy::cast_precision_loss)]
        unsafe {
            fdiv_fast(
                sum as FloatPrecision,
                FloatPrecision::from(fmul_fast(fmul_fast(bounds.width(), bounds.height()), 255.0)),
            )
        },
    )
}

#[inline]
pub fn find_nearest_optimized<V>(
    map: &IntensityMap<V>,
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
            if unsafe { fsub_fast(key.0, f.0.0).abs() < fsub_fast(c.0.0, key.0).abs() } {
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

#[inline(always)]
pub fn prepare_font<const LUT_FONT_SIZE: u16>(
    font_bytes: &'static [u8],
) -> Result<(FontRef<'static>, IntensityMap<char>), Box<dyn std::error::Error>> {
    let font = FontRef::try_from_slice(font_bytes)?;

    let mut intensity_lookup: IntensityMap<char> = IntensityMap::new();
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

    Ok((font, intensity_lookup))
}
