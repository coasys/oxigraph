//! Auxiliary key-value space (feature `aux-kv`).
//!
//! The `aux-kv` cargo feature adds an optional ordered auxiliary key-value space to the [`Store`].
//! Its writes commit atomically with the quads.
//! Applications can use it for data that RDF quads do not model well, for example:
//! * lists in an application order, read page by page with a cursor key,
//! * counters that many transactions change concurrently,
//! * word indexes for text search,
//! * metadata that is not RDF.
//!
//! The space is separate from the quads: SPARQL does not see it, and [`Store::clear`],
//! dumps and loads do not change it.
//!
//! Keys and values are byte strings:
//! * Keys are ordered bytewise. Scans can start from any key, in both directions.
//! * Values are opaque bytes, or counters: 8-byte little-endian two's-complement `i64`
//!   values written by [`Transaction::aux_merge_add`] and read by [`Store::aux_count`].
//!
//! All writes go through a [`Transaction`]. They become visible on [`Transaction::commit`],
//! together with the quad inserts and removes of the same transaction, or not at all.
//!
//! Both storages have the space:
//! * On-disk stores ([`Store::open`]) keep it in a RocksDB column family named `aux`.
//!   The writes go into the same RocksDB write batch as the quads.
//! * In-memory stores ([`Store::new`]) keep it in an ordered map with versions,
//!   with the same visibility rules as the quads.
//!
//! <div class="warning">After a database is opened once with this feature, it has the
//! `aux` column family. A build without the feature can no longer open it,
//! because RocksDB refuses to open a database without all of its column families.
//! This change is one-way.</div>
//!
//! [`Store::open`] adds the `aux` column family to an existing database.
//! [`Store::open_read_only`] cannot add it: open the database once with [`Store::open`] first.
#![allow(
    clippy::multiple_inherent_impl,
    reason = "the auxiliary key-value code stays in its own file to keep the diff of the stock files small"
)]

use super::{Store, Transaction};
use crate::storage::StorageError;
use crate::storage::aux_kv::AuxIter as StorageAuxIter;

impl Store {
    /// Scans the auxiliary space on `[lower, upper)` on a snapshot of the store.
    ///
    /// An empty `upper` means "no upper bound".
    /// * `reverse == false`: starts at the first key `>= lower` and goes forward.
    /// * `reverse == true`: starts at the last key `< upper` and goes backward
    ///   down to `lower` (inclusive).
    ///
    /// The scan does not see writes committed after this call.
    ///
    /// ```
    /// use oxigraph::store::Store;
    ///
    /// let store = Store::new()?;
    /// let mut transaction = store.start_transaction()?;
    /// for (key, value) in [(b"list/1", b"a"), (b"list/2", b"b"), (b"list/3", b"c")] {
    ///     transaction.aux_put(key, value);
    /// }
    /// transaction.commit()?;
    ///
    /// // The page after the cursor key "list/1"
    /// let page = store
    ///     .aux_scan(b"list/1\0", b"list0", false)
    ///     .collect::<Result<Vec<_>, _>>()?;
    /// assert_eq!(
    ///     page,
    ///     [
    ///         (b"list/2".to_vec(), b"b".to_vec()),
    ///         (b"list/3".to_vec(), b"c".to_vec())
    ///     ]
    /// );
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn aux_scan(&self, lower: &[u8], upper: &[u8], reverse: bool) -> AuxIter {
        AuxIter {
            inner: self.storage.aux_scan(lower, upper, reverse),
        }
    }

    /// Reads one value of the auxiliary space in the latest committed state.
    pub fn aux_get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, StorageError> {
        self.storage.aux_get(key)
    }

    /// Reads a counter written by [`Transaction::aux_merge_add`].
    ///
    /// Returns 0 if the key has no value. Returns a corruption error if the value
    /// does not have 8 bytes (for example a value written with [`Transaction::aux_put`]).
    ///
    /// ```
    /// use oxigraph::store::Store;
    ///
    /// let store = Store::new()?;
    /// let mut transaction = store.start_transaction()?;
    /// transaction.aux_merge_add(b"counter", 3);
    /// transaction.aux_merge_add(b"counter", -1);
    /// transaction.commit()?;
    /// assert_eq!(store.aux_count(b"counter")?, 2);
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn aux_count(&self, key: &[u8]) -> Result<i64, StorageError> {
        self.storage.aux_count(key)
    }

    /// Compacts `[lower, upper)` of the auxiliary space.
    ///
    /// An empty `upper` means "no upper bound".
    /// On-disk stores run a manual RocksDB compaction on the `aux` column family only.
    /// The call blocks until the compaction ends. Use it after many deletes in one range
    /// to remove their tombstones. In-memory stores do nothing.
    pub fn compact_aux_range(&self, lower: &[u8], upper: &[u8]) -> Result<(), StorageError> {
        self.storage.compact_aux_range(lower, upper)
    }
}

impl Transaction<'_> {
    /// Writes `value` at `key` in the auxiliary space. It replaces the previous value.
    pub fn aux_put(&mut self, key: &[u8], value: &[u8]) {
        self.inner.aux_put(key, value);
    }

    /// Removes `key` from the auxiliary space with a RocksDB `Delete`.
    ///
    /// Use it for counters and for keys that can have more than one put or merge since
    /// their last removal. Do not use it on keys that you also remove with
    /// [`aux_single_delete`](Self::aux_single_delete).
    pub fn aux_delete(&mut self, key: &[u8]) {
        self.inner.aux_delete(key);
    }

    /// Removes `key` from the auxiliary space with a RocksDB `SingleDelete`.
    ///
    /// A `SingleDelete` and its put remove each other in compactions, so it leaves fewer
    /// tombstones than [`aux_delete`](Self::aux_delete) on keys that change often.
    ///
    /// <div class="warning">The key must have exactly one put and no merge since its last
    /// single delete, and you must never remove it with <code>aux_delete</code>.
    /// Otherwise RocksDB behaviour is undefined: older values can reappear and
    /// compactions can fail.
    /// Use <code>aux_delete</code> for counters and for keys that you put more than once.</div>
    ///
    /// For example, on one key:
    /// * put → `aux_single_delete` → put → `aux_single_delete` is correct.
    /// * put → put → `aux_single_delete` is not correct: use `aux_delete`.
    /// * put → [`aux_delete`](Self::aux_delete) → put → `aux_single_delete` is not correct:
    ///   RocksDB does not allow a `SingleDelete` and a `Delete` on the same key. Use
    ///   `aux_delete` for all the removals of this key.
    /// * [`aux_merge_add`](Self::aux_merge_add) → `aux_single_delete` is not correct: use `aux_delete`.
    ///
    /// See the [RocksDB documentation](https://github.com/facebook/rocksdb/wiki/Single-Delete).
    ///
    /// In-memory stores do the same as [`aux_delete`](Self::aux_delete).
    pub fn aux_single_delete(&mut self, key: &[u8]) {
        self.inner.aux_single_delete(key);
    }

    /// Adds `delta` to the counter at `key` with a RocksDB merge.
    ///
    /// The counter is a signed 64-bit integer and the sum wraps on overflow.
    /// A missing value counts as 0, and so does a value that does not have 8 bytes.
    pub fn aux_merge_add(&mut self, key: &[u8], delta: i64) {
        self.inner.aux_merge_add(key, delta);
    }

    /// Reads one value of the auxiliary space, with the uncommitted writes of this transaction.
    pub fn aux_get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, StorageError> {
        self.inner.aux_get(key)
    }

    /// Reads a counter, with the uncommitted writes of this transaction.
    pub fn aux_count(&self, key: &[u8]) -> Result<i64, StorageError> {
        self.inner.aux_count(key)
    }
}

/// Iterator returned by [`Store::aux_scan`]. Yields `(key, value)` pairs.
#[must_use]
pub struct AuxIter {
    inner: StorageAuxIter,
}

impl Iterator for AuxIter {
    type Item = Result<(Vec<u8>, Vec<u8>), StorageError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}
