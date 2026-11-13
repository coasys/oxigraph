//! RocksDB primitives used by the auxiliary key-value space (feature `aux-kv`).
//!
//! This module is a child of `rocksdb_wrapper` so it can use the private handles
//! of the wrapper types without changing their layout.
#![allow(
    clippy::multiple_inherent_impl,
    reason = "the auxiliary key-value code stays in its own file to keep the diff of the stock files small"
)]

use super::{
    ColumnFamily, Db, DbKind, ErrorStatus, InnerReader, ReadableTransaction, Reader, StorageError,
};
use oxrocksdb_sys::*;
use std::{ptr, slice};

impl Db {
    /// Runs a manual compaction on `[lower, upper)` of one column family.
    ///
    /// An empty `upper` means "no upper bound".
    pub fn compact_range(
        &self,
        column_family: &ColumnFamily,
        lower: &[u8],
        upper: &[u8],
    ) -> Result<(), StorageError> {
        let DbKind::ReadWrite(db) = &self.inner else {
            return Err(StorageError::Other(
                "Compact are only possible on read-write instances".into(),
            ));
        };
        unsafe {
            rocksdb_compact_range_cf_opt(
                db.db,
                column_family.0,
                db.compaction_options,
                lower.as_ptr().cast(),
                lower.len(),
                if upper.is_empty() {
                    ptr::null()
                } else {
                    upper.as_ptr().cast()
                },
                upper.len(),
            )
        }
        Ok(())
    }
}

impl<'a> Reader<'a> {
    /// Opens a bounded iterator on `[lower, upper)`.
    ///
    /// An empty `upper` means "no upper bound".
    /// Forward: seeks to `lower`. Reverse: seeks to the last key strictly before `upper`.
    pub fn scan_range(
        &self,
        column_family: &ColumnFamily,
        lower: &[u8],
        upper: &[u8],
        reverse: bool,
    ) -> RangeIter<'a> {
        let lower = lower.to_vec();
        let upper = (!upper.is_empty()).then(|| upper.to_vec());
        unsafe {
            let options = oxrocksdb_readoptions_create_copy(self.options);
            assert!(
                !options.is_null(),
                "rocksdb_readoptions_create returned null"
            );
            // The read options keep raw pointers to the bounds: `RangeIter` owns the vectors.
            if !lower.is_empty() {
                rocksdb_readoptions_set_iterate_lower_bound(
                    options,
                    lower.as_ptr().cast(),
                    lower.len(),
                );
            }
            if let Some(upper) = &upper {
                rocksdb_readoptions_set_iterate_upper_bound(
                    options,
                    upper.as_ptr().cast(),
                    upper.len(),
                );
            }
            let iter = match &self.inner {
                InnerReader::ReadOnly(inner) => {
                    rocksdb_create_iterator_cf(inner.db, options, column_family.0)
                }
                InnerReader::ReadWrite(inner) => {
                    rocksdb_create_iterator_cf(inner.db.db, options, column_family.0)
                }
                InnerReader::Transaction(inner) => {
                    oxrocksdb_writebatch_wi_create_iterator_with_base_readopts_cf(
                        inner.batch,
                        rocksdb_create_iterator_cf(inner.db.db, options, column_family.0),
                        options,
                        column_family.0,
                    )
                }
            };
            assert!(!iter.is_null(), "rocksdb_create_iterator returned null");
            if reverse {
                if let Some(upper) = &upper {
                    rocksdb_iter_seek_for_prev(iter, upper.as_ptr().cast(), upper.len());
                    // `upper` is exclusive
                    if rocksdb_iter_valid(iter) != 0 && iter_key(iter) == upper.as_slice() {
                        rocksdb_iter_prev(iter);
                    }
                } else {
                    rocksdb_iter_seek_to_last(iter);
                }
            } else {
                rocksdb_iter_seek(iter, lower.as_ptr().cast(), lower.len());
            }
            let mut result = RangeIter {
                inner: iter,
                options,
                reverse,
                is_currently_valid: false,
                lower,
                upper,
                _reader: self.clone(),
            };
            result.refresh_valid();
            result
        }
    }
}

unsafe fn iter_key<'b>(iter: *mut rocksdb_iterator_t) -> &'b [u8] {
    unsafe {
        let key = oxrocksdb_iter_key_slice(iter);
        slice::from_raw_parts(key.data.cast(), key.size)
    }
}

/// A bounded iterator in one direction.
pub struct RangeIter<'a> {
    inner: *mut rocksdb_iterator_t,
    options: *mut rocksdb_readoptions_t,
    reverse: bool,
    is_currently_valid: bool,
    lower: Vec<u8>,
    upper: Option<Vec<u8>>,
    _reader: Reader<'a>,
}

impl Drop for RangeIter<'_> {
    fn drop(&mut self) {
        unsafe {
            rocksdb_iter_destroy(self.inner);
            rocksdb_readoptions_destroy(self.options);
        }
    }
}

unsafe impl Send for RangeIter<'_> {}
unsafe impl Sync for RangeIter<'_> {}

impl RangeIter<'_> {
    fn refresh_valid(&mut self) {
        // Base-delta iterators (transaction readers) do not always honour the
        // lower bound when they move backward, so we check it here too.
        self.is_currently_valid = unsafe { rocksdb_iter_valid(self.inner) != 0 }
            && self.key().is_some_and(|k| {
                k >= self.lower.as_slice() && self.upper.as_deref().is_none_or(|upper| k < upper)
            });
    }

    pub fn status(&self) -> Result<(), StorageError> {
        unsafe {
            ffi_result!(rocksdb_iter_get_error(self.inner))?;
        }
        Ok(())
    }

    /// Moves one step in the direction of the scan.
    pub fn advance(&mut self) {
        unsafe {
            if self.reverse {
                rocksdb_iter_prev(self.inner);
            } else {
                rocksdb_iter_next(self.inner);
            }
        }
        self.refresh_valid();
    }

    fn key(&self) -> Option<&[u8]> {
        unsafe { (rocksdb_iter_valid(self.inner) != 0).then(|| iter_key(self.inner)) }
    }

    /// Current entry, if the iterator is valid.
    pub fn entry(&self) -> Option<(&[u8], &[u8])> {
        if !self.is_currently_valid {
            return None;
        }
        unsafe {
            let key = iter_key(self.inner);
            let mut value_len = 0;
            let value = rocksdb_iter_value(self.inner, &raw mut value_len);
            Some((key, slice::from_raw_parts(value.cast(), value_len)))
        }
    }
}

impl ReadableTransaction<'_> {
    /// Adds a merge operand for `key` to the write batch.
    pub fn merge(&mut self, column_family: &ColumnFamily, key: &[u8], operand: &[u8]) {
        unsafe {
            rocksdb_writebatch_wi_merge_cf(
                self.batch,
                column_family.0,
                key.as_ptr().cast(),
                key.len(),
                operand.as_ptr().cast(),
                operand.len(),
            );
        }
    }

    /// Adds a `SingleDelete` of `key` to the write batch.
    pub fn single_delete(&mut self, column_family: &ColumnFamily, key: &[u8]) {
        unsafe {
            rocksdb_writebatch_wi_singledelete_cf(
                self.batch,
                column_family.0,
                key.as_ptr().cast(),
                key.len(),
            );
        }
    }
}
