//! The in-order chunked fan-out both batch builders share (scoped threads; no rayon here).

/// Map `f` over `items` on at most `n_threads` threads, results IN INDEX ORDER (`<= 1`: serial).
///
/// # Errors
/// The FIRST error in index order, or `panicked` when a worker panicked (never crosses the FFI).
pub(crate) fn map_in_order<T, U, F>(
    items: &[T],
    n_threads: usize,
    panicked: &str,
    f: F,
) -> Result<Vec<U>, String>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> Result<U, String> + Sync,
{
    if items.is_empty() {
        return Ok(Vec::new());
    }
    let threads = n_threads.max(1).min(items.len());
    if threads == 1 {
        return items.iter().map(f).collect();
    }
    let chunk = items.len().div_ceil(threads);
    let f = &f;
    let mut per_chunk: Vec<Result<Vec<U>, String>> = Vec::new();
    std::thread::scope(|scope| {
        let handles: Vec<_> = items
            .chunks(chunk)
            .map(|slice| scope.spawn(move || slice.iter().map(f).collect()))
            .collect();
        for h in handles {
            per_chunk.push(h.join().unwrap_or_else(|_| Err(panicked.to_string())));
        }
    });
    let mut out = Vec::with_capacity(items.len());
    for chunk_result in per_chunk {
        out.extend(chunk_result?);
    }
    Ok(out)
}
