//! The one poison-recovering lock, for state no guarded section can leave torn.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

/// Take `mutex`, recovering a poisoned guard; `counter`, when given, counts each recovery.
#[inline]
pub fn lock_or_recover<'a, T>(
    mutex: &'a Mutex<T>,
    counter: Option<&AtomicUsize>,
) -> MutexGuard<'a, T> {
    mutex.lock().unwrap_or_else(|poisoned| {
        if let Some(counter) = counter {
            counter.fetch_add(1, Ordering::SeqCst);
        }
        poisoned.into_inner()
    })
}

/// Poison `m` the way a panicking thread does: it unwinds while holding the guard.
#[cfg(test)]
pub(crate) fn poison<T: Send>(m: &Mutex<T>) {
    std::thread::scope(|s| {
        let joined = s
            .spawn(|| {
                let _held = m.lock();
                panic!("planted: a thread panics holding the lock");
            })
            .join();
        assert!(joined.is_err(), "the planted panic did not fire");
    });
    assert!(m.is_poisoned(), "the planted panic did not poison the lock");
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use super::{lock_or_recover, poison};

    /// A clean take counts nothing; a poisoned one recovers the value and counts once per take.
    #[test]
    fn a_poisoned_take_recovers_the_value_and_counts() {
        let mutex = Mutex::new(7_u32);
        let counter = AtomicUsize::new(0);
        assert_eq!(*lock_or_recover(&mutex, Some(&counter)), 7);
        assert_eq!(counter.load(Ordering::SeqCst), 0, "a clean take counted");
        poison(&mutex);
        assert_eq!(*lock_or_recover(&mutex, Some(&counter)), 7);
        assert_eq!(*lock_or_recover(&mutex, None), 7);
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "each counted recovery moves the counter"
        );
    }
}
