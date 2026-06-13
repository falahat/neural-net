//! Hand-rolled binary checkpoint format. See module docstring for layout.

use crate::autograd::ParamId;
use crate::module::{Module, ParamVisitor, ParamVisitorMut};
use crate::tensor::Tensor;

#[derive(Debug)]
pub enum CheckpointError {
    Truncated,
    BadHeader(String),
    UnknownParam(String),
    ShapeMismatch {
        path: String,
        want: Vec<usize>,
        got: Vec<usize>,
    },
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckpointError::Truncated => write!(f, "checkpoint truncated"),
            CheckpointError::BadHeader(s) => write!(f, "bad header: {s}"),
            CheckpointError::UnknownParam(p) => write!(f, "unknown param: {p}"),
            CheckpointError::ShapeMismatch { path, want, got } => {
                write!(f, "param '{path}': want {want:?}, got {got:?}")
            }
        }
    }
}

impl std::error::Error for CheckpointError {}

/// Serialise every param in `m` to a `Vec<u8>` in the native format.
/// Param order follows `visit_params` traversal — declaration order.
pub fn save_bytes<M: Module + ?Sized>(m: &M) -> Vec<u8> {
    // Gather params and their flat data while walking.
    struct Collector<'a> {
        out: &'a mut Vec<(String, Vec<usize>, Vec<f32>)>,
    }
    impl<'a> ParamVisitor for Collector<'a> {
        fn visit(&mut self, path: &str, p: &Tensor, _id: ParamId) {
            self.out
                .push((path.to_string(), p.shape().to_vec(), p.data().to_vec()));
        }
    }
    let mut entries: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    m.visit_params(&mut Collector { out: &mut entries });

    // Build manifest JSON by hand. Strings here come from
    // Sequential's index prefixes + "weight"/"bias" — all ASCII, no
    // JSON-escaping needed.
    let mut header = String::from(r#"{"version":1,"params":["#);
    let mut data: Vec<u8> = Vec::new();
    let mut element_offset = 0usize;
    for (i, (path, shape, vals)) in entries.iter().enumerate() {
        if i > 0 {
            header.push(',');
        }
        header.push_str(r#"{"path":""#);
        header.push_str(path);
        header.push_str(r#"","shape":["#);
        for (j, d) in shape.iter().enumerate() {
            if j > 0 {
                header.push(',');
            }
            header.push_str(&d.to_string());
        }
        // offset and len both in *element* count (f32 = 4 bytes).
        header.push_str("],\"offset\":");
        header.push_str(&element_offset.to_string());
        header.push_str(",\"len\":");
        header.push_str(&vals.len().to_string());
        header.push('}');
        for v in vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        element_offset += vals.len();
    }
    header.push_str("]}");

    let mut out = Vec::with_capacity(4 + header.len() + data.len());
    out.extend_from_slice(&(header.len() as u32).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    out
}

/// Load weights from `bytes` into `m`. Param paths must match those
/// `save_bytes` wrote (i.e. the model topology must agree).
pub fn load_bytes<M: Module + ?Sized>(m: &mut M, bytes: &[u8]) -> Result<(), CheckpointError> {
    if bytes.len() < 4 {
        return Err(CheckpointError::Truncated);
    }
    let header_len = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    if bytes.len() < 4 + header_len {
        return Err(CheckpointError::Truncated);
    }
    let header = std::str::from_utf8(&bytes[4..4 + header_len])
        .map_err(|e| CheckpointError::BadHeader(e.to_string()))?;
    let data = &bytes[4 + header_len..];

    let entries = parse_header(header)?;

    // Use a Loader visitor that, for each visited param, finds the
    // matching entry by path and writes its values in place.
    struct Loader<'a> {
        entries: &'a [Entry],
        data: &'a [u8],
        err: Option<CheckpointError>,
    }
    impl<'a> ParamVisitorMut for Loader<'a> {
        fn visit(&mut self, path: &str, p: &mut Tensor, _id: ParamId) {
            if self.err.is_some() {
                return;
            }
            let entry = match self.entries.iter().find(|e| e.path == path) {
                Some(e) => e,
                None => {
                    self.err = Some(CheckpointError::UnknownParam(path.into()));
                    return;
                }
            };
            if entry.shape != p.shape() {
                self.err = Some(CheckpointError::ShapeMismatch {
                    path: path.into(),
                    want: p.shape().to_vec(),
                    got: entry.shape.clone(),
                });
                return;
            }
            let start = entry.offset * 4;
            let end = start + entry.len * 4;
            if end > self.data.len() {
                self.err = Some(CheckpointError::Truncated);
                return;
            }
            let data = p.data_mut();
            for (i, chunk) in self.data[start..end].chunks_exact(4).enumerate() {
                data[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
    }
    let mut loader = Loader {
        entries: &entries,
        data,
        err: None,
    };
    m.visit_params_mut(&mut loader);
    if let Some(e) = loader.err {
        Err(e)
    } else {
        Ok(())
    }
}

#[derive(Debug)]
struct Entry {
    path: String,
    shape: Vec<usize>,
    offset: usize,
    len: usize,
}

// ─── Tiny ad-hoc JSON parser, just enough for our manifest ─────────
//
// Robustness expectation: header is produced by us in the same crate,
// so we don't need a general-purpose parser. We tolerate whitespace
// and the exact key set we emit; anything else is a `BadHeader`.

fn parse_header(s: &str) -> Result<Vec<Entry>, CheckpointError> {
    let bytes = s.as_bytes();
    let mut p = 0usize;
    skip_ws(bytes, &mut p);
    expect(bytes, &mut p, b"{")?;
    // version field — we read it, but don't act on it (version 1 only).
    skip_ws(bytes, &mut p);
    expect(bytes, &mut p, b"\"version\"")?;
    skip_ws(bytes, &mut p);
    expect(bytes, &mut p, b":")?;
    skip_ws(bytes, &mut p);
    let _v = read_int(bytes, &mut p)?;
    skip_ws(bytes, &mut p);
    expect(bytes, &mut p, b",")?;
    skip_ws(bytes, &mut p);
    expect(bytes, &mut p, b"\"params\"")?;
    skip_ws(bytes, &mut p);
    expect(bytes, &mut p, b":")?;
    skip_ws(bytes, &mut p);
    expect(bytes, &mut p, b"[")?;

    let mut entries = Vec::new();
    skip_ws(bytes, &mut p);
    if peek(bytes, p) == Some(b']') {
        return Ok(entries);
    }
    loop {
        entries.push(parse_entry(bytes, &mut p)?);
        skip_ws(bytes, &mut p);
        match peek(bytes, p) {
            Some(b',') => {
                p += 1;
                skip_ws(bytes, &mut p);
            }
            Some(b']') => break,
            _ => return Err(CheckpointError::BadHeader("expected ',' or ']'".into())),
        }
    }
    Ok(entries)
}

fn parse_entry(bytes: &[u8], p: &mut usize) -> Result<Entry, CheckpointError> {
    expect(bytes, p, b"{")?;
    let mut path = String::new();
    let mut shape: Vec<usize> = Vec::new();
    let mut offset: usize = 0;
    let mut len: usize = 0;
    loop {
        skip_ws(bytes, p);
        let key = read_string(bytes, p)?;
        skip_ws(bytes, p);
        expect(bytes, p, b":")?;
        skip_ws(bytes, p);
        match key.as_str() {
            "path" => path = read_string(bytes, p)?,
            "shape" => shape = read_int_array(bytes, p)?,
            "offset" => offset = read_int(bytes, p)?,
            "len" => len = read_int(bytes, p)?,
            other => return Err(CheckpointError::BadHeader(format!("unknown key '{other}'"))),
        }
        skip_ws(bytes, p);
        match peek(bytes, *p) {
            Some(b',') => *p += 1,
            Some(b'}') => {
                *p += 1;
                break;
            }
            _ => return Err(CheckpointError::BadHeader("expected ',' or '}'".into())),
        }
    }
    Ok(Entry {
        path,
        shape,
        offset,
        len,
    })
}

fn skip_ws(bytes: &[u8], p: &mut usize) {
    while *p < bytes.len() && matches!(bytes[*p], b' ' | b'\t' | b'\n' | b'\r') {
        *p += 1;
    }
}

fn peek(bytes: &[u8], p: usize) -> Option<u8> {
    if p < bytes.len() {
        Some(bytes[p])
    } else {
        None
    }
}

fn expect(bytes: &[u8], p: &mut usize, tok: &[u8]) -> Result<(), CheckpointError> {
    if *p + tok.len() > bytes.len() || &bytes[*p..*p + tok.len()] != tok {
        return Err(CheckpointError::BadHeader(format!(
            "expected {:?} at offset {}",
            std::str::from_utf8(tok).unwrap_or("?"),
            p
        )));
    }
    *p += tok.len();
    Ok(())
}

fn read_string(bytes: &[u8], p: &mut usize) -> Result<String, CheckpointError> {
    expect(bytes, p, b"\"")?;
    let start = *p;
    while *p < bytes.len() && bytes[*p] != b'"' {
        *p += 1;
    }
    if *p >= bytes.len() {
        return Err(CheckpointError::BadHeader("unterminated string".into()));
    }
    let s = std::str::from_utf8(&bytes[start..*p])
        .map_err(|e| CheckpointError::BadHeader(e.to_string()))?
        .to_string();
    *p += 1;
    Ok(s)
}

fn read_int(bytes: &[u8], p: &mut usize) -> Result<usize, CheckpointError> {
    let start = *p;
    while *p < bytes.len() && bytes[*p].is_ascii_digit() {
        *p += 1;
    }
    if start == *p {
        return Err(CheckpointError::BadHeader("expected integer".into()));
    }
    std::str::from_utf8(&bytes[start..*p])
        .unwrap()
        .parse()
        .map_err(|_| CheckpointError::BadHeader("integer parse".into()))
}

fn read_int_array(bytes: &[u8], p: &mut usize) -> Result<Vec<usize>, CheckpointError> {
    expect(bytes, p, b"[")?;
    let mut out = Vec::new();
    skip_ws(bytes, p);
    if peek(bytes, *p) == Some(b']') {
        *p += 1;
        return Ok(out);
    }
    loop {
        skip_ws(bytes, p);
        out.push(read_int(bytes, p)?);
        skip_ws(bytes, p);
        match peek(bytes, *p) {
            Some(b',') => *p += 1,
            Some(b']') => {
                *p += 1;
                break;
            }
            _ => {
                return Err(CheckpointError::BadHeader(
                    "expected ',' or ']' in array".into(),
                ))
            }
        }
    }
    Ok(out)
}
