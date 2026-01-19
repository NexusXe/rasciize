use ab_glyph::{Font, FontRef, GlyphId, OutlinedGlyph, PxScale};
use ordered_float::OrderedFloat;
use std::{
    collections::BTreeMap,
    intrinsics::{fdiv_fast, fmul_fast, fsub_fast},
};

pub type FloatPrecision = f32;
pub type IntensityMap<V> = BTreeMap<OrderedFloat<FloatPrecision>, V>;

// source: https://9p.io/sources/plan9/sys/include/ctype.h
#[allow(unused)]
#[inline(always)]
const fn glyph_is_safe(input: u16) -> bool {
    // #define	isalpha(c)	(_ctype[(unsigned char)(c)]&(_U|_L))
    // #define	isupper(c)	(_ctype[(unsigned char)(c)]&_U)
    // #define	islower(c)	(_ctype[(unsigned char)(c)]&_L)
    // #define	isdigit(c)	(_ctype[(unsigned char)(c)]&_N)
    // #define	isxdigit(c)	(_ctype[(unsigned char)(c)]&_X)
    // #define	isspace(c)	(_ctype[(unsigned char)(c)]&_S)
    // #define	ispunct(c)	(_ctype[(unsigned char)(c)]&_P)
    // #define	isalnum(c)	(_ctype[(unsigned char)(c)]&(_U|_L|_N))
    // #define	isprint(c)	(_ctype[(unsigned char)(c)]&(_P|_U|_L|_N|_B))
    // #define	isgraph(c)	(_ctype[(unsigned char)(c)]&(_P|_U|_L|_N))
    // #define	iscntrl(c)	(_ctype[(unsigned char)(c)]&_C)
    // #define	isascii(c)	((unsigned char)(c)<=0177)
    // #define	_toupper(c)	((c)-'a'+'A')
    // #define	_tolower(c)	((c)-'A'+'a')
    // #define	toascii(c)	((c)&0177)
    const U: u16 = 0x01;
    const L: u16 = 0x02;
    const N: u16 = 0x04;
    const S: u16 = 0x010;
    const P: u16 = 0x020;
    const Z: u16 = 0x040;
    const C: u16 = 0x080;
    const B: u16 = 0x0100;
    const X: u16 = 0x0200;

    const fn is_alpha(input: u16) -> bool {
        (input & (U | L)) != 0
    }

    const fn is_upper(input: u16) -> bool {
        (input & U) != 0
    }

    const fn is_lower(input: u16) -> bool {
        (input & L) != 0
    }

    const fn is_digit(input: u16) -> bool {
        (input & N) != 0
    }

    const fn is_xdigit(input: u16) -> bool {
        (input & X) != 0
    }

    const fn is_space(input: u16) -> bool {
        (input & S) != 0
    }

    const fn is_punct(input: u16) -> bool {
        (input & P) != 0
    }

    const fn is_alnum(input: u16) -> bool {
        (input & (U | L | N)) != 0
    }

    const fn is_print(input: u16) -> bool {
        (input & (P | U | L | N | B)) != 0
    }

    const fn is_graph(input: u16) -> bool {
        (input & (P | U | L | N)) != 0
    }

    const fn is_cntrl(input: u16) -> bool {
        (input & C) != 0
    }

    const fn is_ascii(input: u16) -> bool {
        input <= 0x7F
    }

    const fn to_ascii(input: u16) -> u16 {
        input & 0x7F
    }

    (is_print(input)) && !is_cntrl(input)
}

#[inline]
pub fn get_coverage_array(
    font: &FontRef,
    font_size: u16,
    glyph_id: GlyphId,
) -> Option<Vec<Vec<f32>>> {
    // initially get a coverage array at 4x scale and only keep every fourth row to get a 4x wide character for printing in a terminal
    let font_size = f32::from(font_size * 4);
    font.outline_glyph(glyph_id.with_scale(PxScale::from(font_size)))
        .map(|outlined_glyph| {
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
            coverage_array
        })
}

#[inline]
pub fn get_intensity(
    font: &FontRef,
    font_size: f32,
    glyph_id: GlyphId,
    glyph_char: char,
) -> Option<FloatPrecision> {
    let outlined_glyph: OutlinedGlyph =
        font.outline_glyph(glyph_id.with_scale(PxScale::from(font_size)))?;
    let bounds = outlined_glyph.px_bounds();

    #[cfg(not(feature = "ascii-only"))]
    {
        // horrific hack to try and filter out weird glyphs that aren't really characters and might not show up as monospaced
        let standard_glyph_width_wide = font
            .outline_glyph(font.glyph_id('M').with_scale(PxScale::from(font_size)))?
            .px_bounds()
            .width();
        let standard_glyph_width_narrow = font
            .outline_glyph(font.glyph_id('i').with_scale(PxScale::from(font_size)))?
            .px_bounds()
            .width();

        if unsafe {
            fsub_fast(bounds.width(), standard_glyph_width_wide).abs() > font_size.sqrt()
                && (fsub_fast(bounds.width(), standard_glyph_width_narrow).abs() > font_size.sqrt())
        } && !glyph_is_safe(glyph_char as u16)
        {
            return None;
        }
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
    // manually insert space and full block characters since they get skipped over by width checks and might not get the correct intensity otherwise
    intensity_lookup.insert(OrderedFloat(0.0), ' ');
    intensity_lookup.insert(OrderedFloat(1.0), '█');

    for (id, character) in font.codepoint_ids() {
        #[cfg(feature = "ascii-only")]
        if !character.is_ascii() {
            continue;
        }

        if let Some(mut intensity) = get_intensity(&font, f32::from(LUT_FONT_SIZE), id, character) {
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
