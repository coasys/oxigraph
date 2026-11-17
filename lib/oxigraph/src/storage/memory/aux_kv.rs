//! Auxiliary key-value space on the in-memory storage (feature `aux-kv`).
//!
//! The space is an ordered map from keys to version chains. A version chain is a list of
//! `(version, value)` pairs in increasing version order; a `None` value is a deletion.
//! A reader at version `v` sees the last pair with a version `<= v`.
//! The versions are the same as the quad versions of [`MemoryStorage`].
//!
//! A transaction keeps its writes in a private ordered map, so it sees its own writes and
//! a rollback only drops that map. The commit adds one pair per written key to the shared
//! map, under its write lock, and publishes the new version before it releases that lock.
//! Then the commit drops the pairs that no open scan and no future reader can see.
#![allow(
    clippy::multiple_inherent_impl,
    reason = "the auxiliary key-value code stays in its own file to keep the diff of the stock files small"
)]

use super::{MemoryStorage, MemoryStorageTransaction};
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, VecDeque};
use std::ops::Bound;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, RwLock, RwLockWriteGuard};

/// Maximum number of entries that a scan reads under one read lock.
const SCAN_BATCH_SIZE: usize = 256;

type VersionChain = Vec<(usize, Option<Vec<u8>>)>;

/// Pending writes of a transaction. A `None` value is a deletion.
pub type AuxWrites = BTreeMap<Vec<u8>, Option<Vec<u8>>>;

#[derive(Default)]
pub struct AuxSpace {
    entries: RwLock<BTreeMap<Vec<u8>, VersionChain>>,
    /// Number of open scans for each version.
    open_scans: Mutex<BTreeMap<usize, usize>>,
}

fn value_at(chain: &VersionChain, version: usize) -> Option<&Vec<u8>> {
    chain
        .iter()
        .rev()
        .find(|(v, _)| *v <= version)
        .and_then(|(_, value)| value.as_ref())
}

impl AuxSpace {
    /// Adds the transaction writes at `new_version`.
    ///
    /// The caller must publish `new_version` before it drops the returned guard,
    /// so that readers never see the new pairs with an old version.
    pub fn apply(
        &self,
        writes: AuxWrites,
        new_version: usize,
    ) -> Option<RwLockWriteGuard<'_, BTreeMap<Vec<u8>, VersionChain>>> {
        if writes.is_empty() {
            return None;
        }
        let mut entries = self.entries.write().unwrap();
        // Scans register under the read lock, so this set does not change while we hold the
        // write lock. Scans that start later see `new_version`.
        let oldest_visible = self
            .open_scans
            .lock()
            .unwrap()
            .keys()
            .next()
            .map_or(new_version, |v| new_version.min(*v));
        for (key, value) in writes {
            match entries.entry(key) {
                Entry::Occupied(mut entry) => {
                    let chain = entry.get_mut();
                    chain.push((new_version, value));
                    // The last pair with a version <= oldest_visible is the oldest pair
                    // that a reader can still see.
                    if let Some(first_kept) = chain.iter().rposition(|(v, _)| *v <= oldest_visible)
                    {
                        chain.drain(..first_kept);
                    }
                    if matches!(chain.as_slice(), [(v, None)] if *v <= oldest_visible) {
                        entry.remove();
                    }
                }
                Entry::Vacant(entry) => {
                    // No reader sees the deletion of a key that has no value.
                    if value.is_some() {
                        entry.insert(vec![(new_version, value)]);
                    }
                }
            }
        }
        Some(entries)
    }
}

impl MemoryStorage {
    pub fn aux_get(&self, key: &[u8]) -> Option<Vec<u8>> {
        let entries = self.aux.entries.read().unwrap();
        // Commits publish their version under the write lock: this pair is consistent.
        let version = self.version_counter.load(Ordering::Acquire);
        entries
            .get(key)
            .and_then(|chain| value_at(chain, version))
            .cloned()
    }

    /// Does nothing: the in-memory space has no files to compact.
    #[expect(clippy::unused_self)]
    pub fn compact_aux_range(&self, _lower: &[u8], _upper: &[u8]) {}

    pub fn aux_scan(&self, lower: &[u8], upper: &[u8], reverse: bool) -> MemoryAuxIter {
        let upper = (!upper.is_empty()).then(|| upper.to_vec());
        let version = {
            let _entries = self.aux.entries.read().unwrap();
            let version = self.version_counter.load(Ordering::Acquire);
            *self
                .aux
                .open_scans
                .lock()
                .unwrap()
                .entry(version)
                .or_default() += 1;
            version
        };
        MemoryAuxIter {
            space: Arc::clone(&self.aux),
            version,
            exhausted: upper.as_deref().is_some_and(|upper| lower >= upper),
            lower: lower.to_vec(),
            upper,
            reverse,
            cursor: None,
            buffer: VecDeque::new(),
        }
    }
}

impl MemoryStorageTransaction<'_> {
    pub fn aux_put(&mut self, key: &[u8], value: &[u8]) {
        self.aux_writes.insert(key.to_vec(), Some(value.to_vec()));
    }

    pub fn aux_delete(&mut self, key: &[u8]) {
        self.aux_writes.insert(key.to_vec(), None);
    }

    /// Same result as RocksDB's `UInt64AddOperator`: a value that does not have 8 bytes counts as 0.
    pub fn aux_merge_add(&mut self, key: &[u8], delta: i64) {
        let current = self
            .aux_get(key)
            .and_then(|value| <[u8; 8]>::try_from(value.as_slice()).ok())
            .map_or(0, i64::from_le_bytes);
        self.aux_writes.insert(
            key.to_vec(),
            Some(current.wrapping_add(delta).to_le_bytes().to_vec()),
        );
    }

    pub fn aux_get(&self, key: &[u8]) -> Option<Vec<u8>> {
        if let Some(pending) = self.aux_writes.get(key) {
            return pending.clone();
        }
        // Only one transaction runs at a time: `snapshot_id` is the latest committed version.
        self.storage
            .aux
            .entries
            .read()
            .unwrap()
            .get(key)
            .and_then(|chain| value_at(chain, self.snapshot_id))
            .cloned()
    }
}

/// Iterator on a bounded range of the in-memory auxiliary space, at a fixed version.
///
/// It reads the map in batches and does not hold a lock between two calls to `next`.
pub struct MemoryAuxIter {
    space: Arc<AuxSpace>,
    version: usize,
    lower: Vec<u8>,
    upper: Option<Vec<u8>>,
    reverse: bool,
    /// Last key that the iterator read from the map.
    cursor: Option<Vec<u8>>,
    buffer: VecDeque<(Vec<u8>, Vec<u8>)>,
    exhausted: bool,
}

impl MemoryAuxIter {
    fn fill_buffer(&mut self) {
        let entries = self.space.entries.read().unwrap();
        let upper_bound = self
            .upper
            .as_deref()
            .map_or(Bound::Unbounded, Bound::Excluded);
        let range = if self.reverse {
            entries.range::<[u8], _>((
                Bound::Included(self.lower.as_slice()),
                self.cursor.as_deref().map_or(upper_bound, Bound::Excluded),
            ))
        } else {
            entries.range::<[u8], _>((
                self.cursor
                    .as_deref()
                    .map_or(Bound::Included(self.lower.as_slice()), Bound::Excluded),
                upper_bound,
            ))
        };
        let (cursor, exhausted) = if self.reverse {
            read_batch(range.rev(), self.version, &mut self.buffer)
        } else {
            read_batch(range, self.version, &mut self.buffer)
        };
        if cursor.is_some() {
            self.cursor = cursor;
        }
        self.exhausted = exhausted;
    }
}

/// Reads at most [`SCAN_BATCH_SIZE`] keys of `range` into `buffer`.
///
/// Returns the last key read and `true` if the range has no more keys.
fn read_batch<'a>(
    range: impl Iterator<Item = (&'a Vec<u8>, &'a VersionChain)>,
    version: usize,
    buffer: &mut VecDeque<(Vec<u8>, Vec<u8>)>,
) -> (Option<Vec<u8>>, bool) {
    let mut last_key: Option<&Vec<u8>> = None;
    for (read, (key, chain)) in range.enumerate() {
        if read == SCAN_BATCH_SIZE {
            return (last_key.cloned(), false);
        }
        if let Some(value) = value_at(chain, version) {
            buffer.push_back((key.clone(), value.clone()));
        }
        last_key = Some(key);
    }
    (last_key.cloned(), true)
}

impl Iterator for MemoryAuxIter {
    type Item = (Vec<u8>, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(entry) = self.buffer.pop_front() {
                return Some(entry);
            }
            if self.exhausted {
                return None;
            }
            self.fill_buffer();
        }
    }
}

impl Drop for MemoryAuxIter {
    fn drop(&mut self) {
        let mut open_scans = self.space.open_scans.lock().unwrap();
        if let Some(count) = open_scans.get_mut(&self.version) {
            *count -= 1;
            if *count == 0 {
                open_scans.remove(&self.version);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain_lengths(storage: &MemoryStorage) -> Vec<(Vec<u8>, usize)> {
        storage
            .aux
            .entries
            .read()
            .unwrap()
            .iter()
            .map(|(k, chain)| (k.clone(), chain.len()))
            .collect()
    }

    fn commit(storage: &MemoryStorage, write: impl FnOnce(&mut MemoryStorageTransaction<'_>)) {
        let mut transaction = storage.start_transaction();
        write(&mut transaction);
        transaction.commit();
    }

    #[test]
    fn commits_drop_versions_that_no_reader_can_see() {
        let storage = MemoryStorage::new();
        for i in 0..100_i64 {
            commit(&storage, |t| {
                t.aux_merge_add(b"counter", 1);
                t.aux_put(b"value", &i.to_le_bytes());
            });
        }
        assert_eq!(
            chain_lengths(&storage),
            [(b"counter".to_vec(), 1), (b"value".to_vec(), 1)]
        );

        // An open scan keeps the versions that it can see.
        let scan = storage.aux_scan(b"", b"", false);
        for _ in 0..10 {
            commit(&storage, |t| t.aux_merge_add(b"counter", 1));
        }
        assert_eq!(
            chain_lengths(&storage),
            [(b"counter".to_vec(), 11), (b"value".to_vec(), 1)]
        );
        assert_eq!(scan.count(), 2);
        assert!(storage.aux.open_scans.lock().unwrap().is_empty());

        // The next commit on the key drops the old versions, and deletions leave no entry.
        commit(&storage, |t| {
            t.aux_merge_add(b"counter", 1);
            t.aux_delete(b"value");
            t.aux_delete(b"missing");
        });
        assert_eq!(chain_lengths(&storage), [(b"counter".to_vec(), 1)]);
        assert_eq!(
            storage.aux_get(b"counter"),
            Some(111_i64.to_le_bytes().to_vec())
        );
    }

    #[test]
    fn rollback_drops_pending_writes() {
        let storage = MemoryStorage::new();
        {
            let mut transaction = storage.start_transaction();
            transaction.aux_put(b"k", b"v");
            assert_eq!(transaction.aux_get(b"k"), Some(b"v".to_vec()));
        }
        assert_eq!(storage.aux_get(b"k"), None);
        assert!(chain_lengths(&storage).is_empty());
    }
}
