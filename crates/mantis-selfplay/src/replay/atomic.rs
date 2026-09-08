//! Atomic file publication for the replay rings (R345(b)(3)).
//!
//! Both ring formats used to open their FINAL path with `File::create`, which truncates an
//! existing file to zero before the first new byte is written. For the one artifact a resume
//! cannot be reconstructed without, that is the worst available order: a process killed
//! mid-save left no ring at all, rather than the previous one. The ring is the most expensive
//! file the run writes and the only one whose loss costs hours of self-play.
//!
//! The fix is the ordinary one and the ordering is the whole of it: write a temp sibling,
//! `sync_all` it so the bytes are durable, rename over the target (atomic with respect to any
//! reader), then fsync the DIRECTORY so the rename itself survives a power loss. Skipping the
//! last step leaves a window where the name resolves but the directory entry does not, which
//! is the failure a bundle manifest cannot detect because the manifest is in the same
//! directory.

use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicU64, Ordering};

/// Distinguishes concurrent writers of the same target. A shared fixed `<name>.tmp` makes two
/// writers race each other's temp file and one of them renames a file the other is still
/// writing; pid alone is not enough because one process can publish the same path twice.
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path_for(path: &str) -> String {
    let n = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{path}.{}.{n}.tmp", std::process::id())
}

/// Publish `path` atomically: `write` fills a temp sibling, which is synced and renamed.
///
/// `write` receives a `BufWriter` over the temp file and writes the whole payload. On any
/// error — from `write` or from the IO around it — the temp file is removed and `path` keeps
/// whatever it held.
///
/// # Errors
/// The temp file could not be created, written, synced or renamed, with the failing path and
/// the OS error named. A failure to remove the temp file after another failure is deliberately
/// NOT reported over the original: the first error is the one that explains the state.
pub fn atomic_save<F>(path: &str, write: F) -> Result<(), String>
where
    F: FnOnce(&mut BufWriter<std::fs::File>) -> Result<(), String>,
{
    let tmp = temp_path_for(path);
    let result = (|| -> Result<(), String> {
        let file = std::fs::File::create(&tmp).map_err(|e| format!("cannot create {tmp}: {e}"))?;
        let mut w = BufWriter::new(file);
        write(&mut w)?;
        w.flush().map_err(|e| format!("cannot flush {tmp}: {e}"))?;
        let file = w
            .into_inner()
            .map_err(|e| format!("cannot finish writing {tmp}: {e}"))?;
        file.sync_all()
            .map_err(|e| format!("cannot fsync {tmp}: {e}"))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("cannot rename {tmp} -> {path}: {e}"))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
        return result;
    }
    sync_parent_dir(path)
}

/// fsync the directory containing `path`, so the rename that published it is itself durable.
///
/// # Errors
/// The parent directory could not be opened or synced. A path with no parent component is
/// treated as the current directory rather than as an error.
fn sync_parent_dir(path: &str) -> Result<(), String> {
    let parent = std::path::Path::new(path)
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map_or_else(
            || std::path::PathBuf::from("."),
            std::path::Path::to_path_buf,
        );
    let dir = std::fs::File::open(&parent)
        .map_err(|e| format!("cannot open {} to fsync: {e}", parent.display()))?;
    dir.sync_all()
        .map_err(|e| format!("cannot fsync {}: {e}", parent.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn read(path: &std::path::Path) -> Vec<u8> {
        let mut buf = Vec::new();
        std::fs::File::open(path)
            .expect("the published file must exist")
            .read_to_end(&mut buf)
            .expect("the published file must be readable");
        buf
    }

    #[test]
    fn a_successful_save_publishes_the_bytes_and_leaves_no_temp() {
        let dir = tempdir();
        let path = dir.join("ring.bin");
        let p = path.to_string_lossy().to_string();
        atomic_save(&p, |w| w.write_all(b"payload").map_err(|e| e.to_string())).unwrap();
        assert_eq!(read(&path), b"payload");
        assert_eq!(entries(&dir), vec!["ring.bin".to_string()]);
    }

    #[test]
    fn a_failed_save_leaves_the_previous_file_intact() {
        // The planted break for R345(b)(3): the property `File::create` gives up. Under the
        // old writer this assertion is unsatisfiable — the target is truncated before the
        // writer is ever called, so there is nothing left to be intact.
        let dir = tempdir();
        let path = dir.join("ring.bin");
        let p = path.to_string_lossy().to_string();
        atomic_save(&p, |w| w.write_all(b"first").map_err(|e| e.to_string())).unwrap();

        let err = atomic_save(&p, |w| {
            w.write_all(b"second, but only ha")
                .map_err(|e| e.to_string())?;
            Err("planted mid-write failure".to_string())
        })
        .unwrap_err();

        assert_eq!(err, "planted mid-write failure");
        assert_eq!(
            read(&path),
            b"first",
            "the interrupted write reached the target"
        );
        assert_eq!(
            entries(&dir),
            vec!["ring.bin".to_string()],
            "a temp file survived the failure"
        );
    }

    #[test]
    fn concurrent_writers_do_not_share_a_temp_name() {
        let a = temp_path_for("/tmp/x");
        let b = temp_path_for("/tmp/x");
        assert_ne!(
            a, b,
            "two temp names collided, so two writers would race one file"
        );
    }

    fn tempdir() -> std::path::PathBuf {
        let n = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("mantis-atomic-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("test temp dir must be creatable");
        dir
    }

    fn entries(dir: &std::path::Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .expect("test temp dir must be readable")
            .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().to_string()))
            .collect();
        names.sort();
        names
    }
}
