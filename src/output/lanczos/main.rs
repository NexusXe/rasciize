use std::env;
use std::path::Path;
use std::process;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 4 {
        eprintln!("Usage: {} <input_image> <width> <height>", args[0]);
        process::exit(1);
    }

    let input_path = &args[1];
    let width_str = &args[2];
    let height_str = &args[3];

    let width: u32 = match width_str.parse() {
        Ok(w) => w,
        Err(_) => {
            eprintln!("Error: Width must be a positive integer.");
            process::exit(1);
        }
    };

    let height: u32 = match height_str.parse() {
        Ok(h) => h,
        Err(_) => {
            eprintln!("Error: Height must be a positive integer.");
            process::exit(1);
        }
    };

    println!("Loading image from '{}'...", input_path);
    let img = match image::open(input_path) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Error loading image: {}", e);
            process::exit(1);
        }
    };

    println!("Resizing to {}x{}...", width, height);
    let start = Instant::now();
    let resized = lanczos_simd::lanczos::lanczos3_resize(&img, width, height);
    let duration = start.elapsed();
    println!("Resized in {:.2}!", duration.as_secs_f32());

    let path = Path::new(input_path);
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("png");

    let output_filename = format!("{}_Resized.{}", stem, ext);
    let output_path = path.with_file_name(output_filename);

    println!("Saving to '{}'...", output_path.display());
    if let Err(e) = resized.save(&output_path) {
        eprintln!("Error saving image: {}", e);
        process::exit(1);
    }

    println!("Done!");
}
