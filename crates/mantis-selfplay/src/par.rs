//! The in-order chunked fan-out both batch builders share (scoped threads; no rayon here).

/// Map `f` over `items` on at most `n_threads` threads, results IN INDEX ORDER (`<= 1`: serial).
///
/// # Errors
/// The FIRST error in index order, or `panicked` with the worker's panic message when a worker
/// panicked (never crosses the FFI).
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
            per_chunk.push(h.join().unwrap_or_else(|payload| {
                Err(format!("{panicked}: {}", panic_message(payload.as_ref())))
            }));
        }
    });
    let mut out = Vec::with_capacity(items.len());
    for chunk_result in per_chunk {
        out.extend(chunk_result?);
    }
    Ok(out)
}

/// A panic's message, whether `panic!` was handed a literal or a formatted string.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("a panic with a non-string payload")
}

#[cfg(test)]
mod tests {
    use super::map_in_order;

    #[test]
    fn a_panicking_worker_names_its_payload() {
        let items: Vec<u32> = (0..8).collect();
        for (n, msg) in [
            (5, "verify_contract: edge 7 out of range"),
            (6, "a formatted 6"),
        ] {
            let err = map_in_order(&items, 4, "a worker panicked", |&i| {
                if i == n && n == 5 {
                    panic!("verify_contract: edge 7 out of range");
                }
                if i == n {
                    panic!("a formatted {}", i);
                }
                Ok(i)
            })
            .expect_err("the panic becomes an error");
            assert!(err.starts_with("a worker panicked"), "{err}");
            assert!(err.contains(msg), "the payload is lost: {err}");
        }
    }
}
