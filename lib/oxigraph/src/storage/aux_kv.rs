//! Dispatch of the auxiliary key-value space (feature `aux-kv`) to the storage backends.
#![allow(
    clippy::multiple_inherent_impl,
    reason = "the auxiliary key-value code stays in its own file to keep the diff of the stock files small"
)]

use super::{
    CorruptionError, Storage, StorageError, StorageKind, StorageReadableTransaction,
    StorageReadableTransactionKind,
};
use crate::storage::memory::aux_kv::MemoryAuxIter;
#[cfg(all(not(target_family = "wasm"), feature = "rocksdb"))]
use crate::storage::rocksdb::aux_kv::RocksDbAuxIter;

/// Decodes a counter value: 8 bytes, little-endian, two's complement. A missing value is 0.
pub fn decode_counter(value: Option<&[u8]>) -> Result<i64, StorageError> {
    let Some(value) = value else {
        return Ok(0);
    };
    let bytes: [u8; 8] = value.try_into().map_err(|_| {
        CorruptionError::msg(format!(
            "An auxiliary counter value must have 8 bytes, found {}",
            value.len()
        ))
    })?;
    Ok(i64::from_le_bytes(bytes))
}

impl Storage {
    pub fn aux_scan(&self, lower: &[u8], upper: &[u8], reverse: bool) -> AuxIter {
        AuxIter {
            kind: match &self.kind {
                #[cfg(all(not(target_family = "wasm"), feature = "rocksdb"))]
                StorageKind::RocksDb(storage) => {
                    AuxIterKind::RocksDb(storage.aux_scan(lower, upper, reverse))
                }
                StorageKind::Memory(storage) => {
                    AuxIterKind::Memory(storage.aux_scan(lower, upper, reverse))
                }
            },
        }
    }

    #[cfg_attr(
        not(all(not(target_family = "wasm"), feature = "rocksdb")),
        expect(clippy::unnecessary_wraps)
    )]
    pub fn aux_get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, StorageError> {
        match &self.kind {
            #[cfg(all(not(target_family = "wasm"), feature = "rocksdb"))]
            StorageKind::RocksDb(storage) => storage.aux_get(key),
            StorageKind::Memory(storage) => Ok(storage.aux_get(key)),
        }
    }

    pub fn aux_count(&self, key: &[u8]) -> Result<i64, StorageError> {
        match &self.kind {
            #[cfg(all(not(target_family = "wasm"), feature = "rocksdb"))]
            StorageKind::RocksDb(storage) => storage.aux_count(key),
            StorageKind::Memory(storage) => decode_counter(storage.aux_get(key).as_deref()),
        }
    }

    #[cfg_attr(
        not(all(not(target_family = "wasm"), feature = "rocksdb")),
        expect(clippy::unnecessary_wraps)
    )]
    pub fn compact_aux_range(&self, lower: &[u8], upper: &[u8]) -> Result<(), StorageError> {
        match &self.kind {
            #[cfg(all(not(target_family = "wasm"), feature = "rocksdb"))]
            StorageKind::RocksDb(storage) => storage.compact_aux_range(lower, upper),
            StorageKind::Memory(storage) => {
                storage.compact_aux_range(lower, upper);
                Ok(())
            }
        }
    }
}

impl StorageReadableTransaction<'_> {
    pub fn aux_put(&mut self, key: &[u8], value: &[u8]) {
        match &mut self.kind {
            #[cfg(all(not(target_family = "wasm"), feature = "rocksdb"))]
            StorageReadableTransactionKind::RocksDb(transaction) => transaction.aux_put(key, value),
            StorageReadableTransactionKind::Memory(transaction) => transaction.aux_put(key, value),
        }
    }

    pub fn aux_delete(&mut self, key: &[u8]) {
        match &mut self.kind {
            #[cfg(all(not(target_family = "wasm"), feature = "rocksdb"))]
            StorageReadableTransactionKind::RocksDb(transaction) => transaction.aux_delete(key),
            StorageReadableTransactionKind::Memory(transaction) => transaction.aux_delete(key),
        }
    }

    pub fn aux_single_delete(&mut self, key: &[u8]) {
        match &mut self.kind {
            #[cfg(all(not(target_family = "wasm"), feature = "rocksdb"))]
            StorageReadableTransactionKind::RocksDb(transaction) => {
                transaction.aux_single_delete(key)
            }
            // The in-memory space has no tombstone stacking: a single delete is a delete.
            StorageReadableTransactionKind::Memory(transaction) => transaction.aux_delete(key),
        }
    }

    pub fn aux_merge_add(&mut self, key: &[u8], delta: i64) {
        match &mut self.kind {
            #[cfg(all(not(target_family = "wasm"), feature = "rocksdb"))]
            StorageReadableTransactionKind::RocksDb(transaction) => {
                transaction.aux_merge_add(key, delta)
            }
            StorageReadableTransactionKind::Memory(transaction) => {
                transaction.aux_merge_add(key, delta)
            }
        }
    }

    #[cfg_attr(
        not(all(not(target_family = "wasm"), feature = "rocksdb")),
        expect(clippy::unnecessary_wraps)
    )]
    pub fn aux_get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, StorageError> {
        match &self.kind {
            #[cfg(all(not(target_family = "wasm"), feature = "rocksdb"))]
            StorageReadableTransactionKind::RocksDb(transaction) => transaction.aux_get(key),
            StorageReadableTransactionKind::Memory(transaction) => Ok(transaction.aux_get(key)),
        }
    }

    pub fn aux_count(&self, key: &[u8]) -> Result<i64, StorageError> {
        match &self.kind {
            #[cfg(all(not(target_family = "wasm"), feature = "rocksdb"))]
            StorageReadableTransactionKind::RocksDb(transaction) => transaction.aux_count(key),
            StorageReadableTransactionKind::Memory(transaction) => {
                decode_counter(transaction.aux_get(key).as_deref())
            }
        }
    }
}

/// Iterator on a bounded range of the auxiliary key-value space.
#[must_use]
pub struct AuxIter {
    kind: AuxIterKind,
}

enum AuxIterKind {
    #[cfg(all(not(target_family = "wasm"), feature = "rocksdb"))]
    RocksDb(RocksDbAuxIter),
    Memory(MemoryAuxIter),
}

impl Iterator for AuxIter {
    type Item = Result<(Vec<u8>, Vec<u8>), StorageError>;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.kind {
            #[cfg(all(not(target_family = "wasm"), feature = "rocksdb"))]
            AuxIterKind::RocksDb(iter) => iter.next(),
            AuxIterKind::Memory(iter) => iter.next().map(Ok),
        }
    }
}
