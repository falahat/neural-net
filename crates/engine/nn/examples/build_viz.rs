//! Combine every `*.json` file in `crates/ml/nn/bench_results/` into
//! one self-contained `viz.html` with the data inlined as JS and
//! Observable Plot loaded from a CDN. Open the result directly with
//! `file://`; no local server required.
//!
//! Usage:
//! ```bash
//! cargo run --release --example build_viz -p nn -- \
//!     --in  crates/ml/nn/bench_results \
//!     --out crates/ml/nn/bench_results/viz.html
//! ```
//!
//! Reads every `*.json` in the input directory, validates schema
//! version, embeds the union into the HTML, and renders four charts:
//!   1. matmul GFLOP/s curve vs problem size (m·k·n), one line per label.
//!   2. matmul speedup ratio (label / first-found-baseline).
//!   3. train_step latency vs (batch · hidden), one line per label.
//!   4. full results table.

use std::fs;
use std::io::Read;
use std::path::Path;

fn main() -> std::io::Result<()> {
    let (in_dir, out_path) = parse_args();
    println!("build_viz: scanning {in_dir} → {out_path}");

    // Collect every JSON file in the input dir; embed each as a
    // verbatim string literal in the HTML.
    let mut datasets: Vec<(String, String)> = Vec::new();
    for entry in fs::read_dir(&in_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let mut s = String::new();
        fs::File::open(&path)?.read_to_string(&mut s)?;
        datasets.push((name, s));
    }

    if datasets.is_empty() {
        eprintln!("no .json files found in {in_dir}; run `bench_suite` first");
        std::process::exit(1);
    }
    println!(
        "found {} dataset(s): {}",
        datasets.len(),
        datasets
            .iter()
            .map(|(n, _)| n.clone())
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Build the HTML.
    let html = render_html(&datasets);
    let path = Path::new(&out_path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, html)?;
    println!("wrote {out_path}");
    Ok(())
}

fn parse_args() -> (String, String) {
    let args: Vec<String> = std::env::args().collect();
    let mut in_dir = "crates/ml/nn/bench_results".to_string();
    let mut out = "crates/ml/nn/bench_results/viz.html".to_string();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--in" if i + 1 < args.len() => {
                in_dir = args[i + 1].clone();
                i += 2;
            }
            "--out" if i + 1 < args.len() => {
                out = args[i + 1].clone();
                i += 2;
            }
            _ => i += 1,
        }
    }
    (in_dir, out)
}

fn render_html(datasets: &[(String, String)]) -> String {
    // Inline each dataset as `const dataset_<name> = <json>;`.
    let inline_data: String = datasets
        .iter()
        .map(|(name, json)| {
            format!(
                "    {{ label: \"{}\", json: {} }},",
                name.replace('\\', "\\\\").replace('"', "\\\""),
                json
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    HTML_TEMPLATE.replace("/*__INLINE_DATA__*/", &inline_data)
}

const HTML_TEMPLATE: &str = include_str!("viz_template.html");
