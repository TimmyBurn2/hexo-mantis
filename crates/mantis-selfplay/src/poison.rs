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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use super::lock_or_recover;

    /// A clean take counts nothing; a poisoned one recovers the value and counts once per take.
    #[test]
    fn a_poisoned_take_recovers_the_value_and_counts() {
        let mutex = Mutex::new(7_u32);
        let counter = AtomicUsize::new(0);
        assert_eq!(*lock_or_recover(&mutex, Some(&counter)), 7);
        assert_eq!(counter.load(Ordering::SeqCst), 0, "a clean take counted");
        std::thread::scope(|s| {
            let joined = s
                .spawn(|| {
                    let _held = mutex.lock();
                    panic!("planted: poisons the lock");
                })
                .join();
            assert!(joined.is_err());
        });
        assert!(mutex.is_poisoned());
        assert_eq!(*lock_or_recover(&mutex, Some(&counter)), 7);
        assert_eq!(*lock_or_recover(&mutex, None), 7);
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "each counted recovery moves the counter"
        );
    }
}
