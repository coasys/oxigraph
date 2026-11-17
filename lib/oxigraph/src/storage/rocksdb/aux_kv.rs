//! Auxiliary key-value space on the RocksDB storage (feature `aux-kv`).
//!
//! The space is its own column family, `aux`. It uses the default bytewise
//! comparator and no prefix extractor, so seeks and range scans are total-order.
//! Its merge operator is RocksDB's `UInt64AddOperator`: operands are 8-byte little-endian
//! integers and the sum wraps, so two's-complement `i64` deltas give signed counters.
#![allow(
    clippy::multiple_inherent_impl,
    reason = "the auxiliary key-value code stays in its own file to keep the diff of the stock files small"
)]

use super::{ColumnFamilyDefinition, RocksDbStorage, RocksDbStorageReadableTransaction};
use crate::storage::aux_kv::decode_counter;
use crate::storage::error::StorageError;
use crate::storage::rocksdb_wrapper::aux_kv::RangeIter;

pub const AUX_CF: &str = "aux";

/// Definition of the `aux` column family.
pub fn column_family() -> ColumnFamilyDefinition {
    ColumnFamilyDefinition {
        name: AUX_CF,
        use_iter: true,
        min_prefix_size: 0, // no prefix extractor: total-order seeks
        unordered_writes: false,
        uint64_add_merge: true,
    }
}

impl RocksDbStorage {
    pub fn aux_scan(&self, lower: &[u8], upper: &[u8], reverse: bool) -> RocksDbAuxIter {
        RocksDbAuxIter {
            iter: self
                .db
                .snapshot()
                .scan_range(&self.aux_cf, lower, upper, reverse),
            done: false,
        }
    }

    pub fn aux_get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, StorageError> {
        Ok(self.db.snapshot().get(&self.aux_cf, key)?.map(Vec::from))
    }

    pub fn aux_count(&self, key: &[u8]) -> Result<i64, StorageError> {
        decode_counter(self.db.snapshot().get(&self.aux_cf, key)?.as_deref())
    }

    pub fn compact_aux_range(&self, lower: &[u8], upper: &[u8]) -> Result<(), StorageError> {
        self.db.compact_range(&self.aux_cf, lower, upper)
    }
}

impl RocksDbStorageReadableTransaction<'_> {
    pub fn aux_put(&mut self, key: &[u8], value: &[u8]) {
        self.transaction.insert(&self.storage.aux_cf, key, value);
    }

    pub fn aux_delete(&mut self, key: &[u8]) {
        self.transaction.remove(&self.storage.aux_cf, key);
    }

    pub fn aux_single_delete(&mut self, key: &[u8]) {
        self.transaction.single_delete(&self.storage.aux_cf, key);
    }

    pub fn aux_merge_add(&mut self, key: &[u8], delta: i64) {
        self.transaction
            .merge(&self.storage.aux_cf, key, &delta.to_le_bytes());
    }

    pub fn aux_get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, StorageError> {
        Ok(self
            .transaction
            .reader()
            .get(&self.storage.aux_cf, key)?
            .map(Vec::from))
    }

    pub fn aux_count(&self, key: &[u8]) -> Result<i64, StorageError> {
        decode_counter(
            self.transaction
                .reader()
                .get(&self.storage.aux_cf, key)?
                .as_deref(),
        )
    }
}

/// Iterator on a bounded range of the `aux` column family, on a snapshot.
pub struct RocksDbAuxIter {
    iter: RangeIter<'static>,
    done: bool,
}

impl Iterator for RocksDbAuxIter {
    type Item = Result<(Vec<u8>, Vec<u8>), StorageError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        if let Some((key, value)) = self.iter.entry() {
            let entry = (key.to_vec(), value.to_vec());
            self.iter.advance();
            return Some(Ok(entry));
        }
        self.done = true;
        self.iter.status().err().map(Err)
    }
}
