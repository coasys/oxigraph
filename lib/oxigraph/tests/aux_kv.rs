//! Tests of the auxiliary key-value space (feature `aux-kv`).
//!
//! Most tests run on both storages: an in-memory store and an on-disk store.
#![cfg(all(test, feature = "aux-kv"))]
#![allow(
    clippy::panic_in_result_fn,
    clippy::missing_asserts_for_indexing,
    clippy::tests_outside_test_module
)]

use oxigraph::model::*;
use oxigraph::store::{StorageError, Store};
use std::error::Error;
#[cfg(all(not(target_family = "wasm"), feature = "rocksdb"))]
use tempfile::TempDir;

type TestResult = Result<(), Box<dyn Error>>;

/// Runs `test` on a new in-memory store, then on a new on-disk store.
fn on_each_store(test: impl Fn(&Store) -> TestResult) -> TestResult {
    test(&Store::new()?).map_err(|e| format!("in-memory store: {e}"))?;
    #[cfg(all(not(target_family = "wasm"), feature = "rocksdb"))]
    {
        let dir = TempDir::new()?;
        test(&Store::open(dir.path())?).map_err(|e| format!("on-disk store: {e}"))?;
    }
    Ok(())
}

fn quad(id: &str) -> Quad {
    let s = NamedNode::new(format!("http://example.com/{id}")).unwrap();
    Quad::new(
        s.clone(),
        NamedNode::new("http://example.com/p").unwrap(),
        Literal::from(id),
        GraphName::NamedNode(s),
    )
}

fn key(i: u32) -> Vec<u8> {
    let mut k = b"list/".to_vec();
    k.extend_from_slice(&i.to_be_bytes());
    k
}

fn key_id(k: &[u8]) -> u32 {
    u32::from_be_bytes(k[5..9].try_into().unwrap())
}

fn scan_ids(store: &Store, lower: &[u8], upper: &[u8], reverse: bool) -> Vec<u32> {
    store
        .aux_scan(lower, upper, reverse)
        .map(|e| key_id(&e.unwrap().0))
        .collect()
}

#[test]
fn aux_writes_commit_atomically_with_quads() -> TestResult {
    on_each_store(|store| {
        // Committed transaction: quad and aux entry are both visible.
        let q1 = quad("a");
        let mut t = store.start_transaction()?;
        t.insert(&q1);
        t.aux_put(b"k1", b"v1");
        // Own writes are visible before commit, and invisible outside before commit.
        assert_eq!(t.aux_get(b"k1")?, Some(b"v1".to_vec()));
        assert_eq!(store.aux_get(b"k1")?, None);
        assert!(store.aux_scan(b"", b"", false).next().is_none());
        assert!(!store.contains(&q1)?);
        t.commit()?;
        assert!(store.contains(&q1)?);
        assert_eq!(store.aux_get(b"k1")?, Some(b"v1".to_vec()));

        // Transaction dropped without commit: neither the quad nor the aux entries are written.
        let q2 = quad("b");
        {
            let mut t = store.start_transaction()?;
            t.insert(&q2);
            t.aux_put(b"k2", b"v2");
            t.aux_put(b"k1", b"changed");
            t.aux_merge_add(b"c2", 5);
            assert_eq!(t.aux_count(b"c2")?, 5);
            assert_eq!(t.aux_get(b"k1")?, Some(b"changed".to_vec()));
            drop(t);
        }
        assert!(!store.contains(&q2)?);
        assert_eq!(store.aux_get(b"k2")?, None);
        assert_eq!(store.aux_get(b"k1")?, Some(b"v1".to_vec()));
        assert_eq!(store.aux_count(b"c2")?, 0);

        // Quad removal and aux single delete in the same transaction.
        let mut t = store.start_transaction()?;
        t.remove(&q1);
        t.aux_single_delete(b"k1");
        assert_eq!(t.aux_get(b"k1")?, None);
        t.commit()?;
        assert!(!store.contains(&q1)?);
        assert_eq!(store.aux_get(b"k1")?, None);
        Ok(())
    })
}

#[test]
fn failed_transaction_rolls_back_aux_writes() -> TestResult {
    on_each_store(|store| {
        let mut t = store.start_transaction()?;
        t.aux_put(b"k", b"v");
        t.aux_merge_add(b"c", 1);
        t.commit()?;

        // The application logic fails in the middle of the transaction: it returns before commit.
        let result: Result<(), Box<dyn Error>> = (|| {
            let mut t = store.start_transaction()?;
            t.insert(&quad("x"));
            t.aux_delete(b"k");
            t.aux_merge_add(b"c", 10);
            t.aux_put(b"new", b"value");
            Err("application error".into())
        })();
        assert!(result.is_err());
        assert!(!store.contains(&quad("x"))?);
        assert_eq!(store.aux_get(b"k")?, Some(b"v".to_vec()));
        assert_eq!(store.aux_count(b"c")?, 1);
        assert_eq!(store.aux_get(b"new")?, None);
        assert_eq!(store.aux_scan(b"", b"", false).count(), 2);
        Ok(())
    })
}

#[test]
fn forward_and_reverse_seek_with_bounds() -> TestResult {
    on_each_store(|store| {
        // 10k even ids: 0, 2, ..., 19998
        let mut t = store.start_transaction()?;
        for i in 0..10_000 {
            t.aux_put(&key(i * 2), &(i * 2).to_le_bytes());
        }
        // Keys outside the "list/" prefix, to check that bounds stop the scan.
        t.aux_put(b"aaaa", b"before");
        t.aux_put(b"zzzz", b"after");
        t.commit()?;

        // Forward, bounds on existing keys: lower inclusive, upper exclusive.
        let ids = scan_ids(store, &key(1000), &key(2000), false);
        assert_eq!(ids.len(), 500);
        assert_eq!(ids.first(), Some(&1000));
        assert_eq!(ids.last(), Some(&1998));
        assert!(ids.windows(2).all(|w| w[0] < w[1]));

        // Reverse, same bounds: starts at the last key < upper.
        let ids = scan_ids(store, &key(1000), &key(2000), true);
        assert_eq!(ids.len(), 500);
        assert_eq!(ids.first(), Some(&1998));
        assert_eq!(ids.last(), Some(&1000));
        assert!(ids.windows(2).all(|w| w[0] > w[1]));

        // Bounds between keys (odd ids do not exist).
        assert_eq!(
            scan_ids(store, &key(1001), &key(1007), false),
            vec![1002, 1004, 1006]
        );
        assert_eq!(
            scan_ids(store, &key(1001), &key(1007), true),
            vec![1006, 1004, 1002]
        );

        // Whole prefix in both directions: the bounds exclude "aaaa" and "zzzz".
        let ids = scan_ids(store, b"list/", b"list0", false);
        assert_eq!(ids.len(), 10_000);
        assert_eq!((ids[0], ids[9_999]), (0, 19_998));
        let ids = scan_ids(store, b"list/", b"list0", true);
        assert_eq!(ids.len(), 10_000);
        assert_eq!((ids[0], ids[9_999]), (19_998, 0));

        // Top-k: take the first 3 in reverse.
        let top: Vec<u32> = store
            .aux_scan(b"list/", b"list0", true)
            .take(3)
            .map(|e| key_id(&e.unwrap().0))
            .collect();
        assert_eq!(top, vec![19_998, 19_996, 19_994]);

        // Values come with keys.
        let (k, v) = store.aux_scan(&key(42), &key(43), false).next().unwrap()?;
        assert_eq!((key_id(&k), v), (42, 42_u32.to_le_bytes().to_vec()));

        // Empty upper = no upper bound.
        let all: Vec<Vec<u8>> = store
            .aux_scan(b"", b"", true)
            .map(|e| e.unwrap().0)
            .collect();
        assert_eq!(all.len(), 10_002);
        assert_eq!(all[0], b"zzzz");
        assert_eq!(all[10_001], b"aaaa");
        let all: Vec<Vec<u8>> = store
            .aux_scan(b"list/", b"", false)
            .map(|e| e.unwrap().0)
            .collect();
        assert_eq!(all.len(), 10_001);
        assert_eq!(all[10_000], b"zzzz");

        // Empty and inverted ranges.
        assert!(scan_ids(store, &key(1001), &key(1002), false).is_empty());
        assert!(scan_ids(store, &key(1001), &key(1002), true).is_empty());
        assert!(scan_ids(store, &key(2000), &key(2000), false).is_empty());
        assert!(scan_ids(store, &key(2000), &key(2000), true).is_empty());
        assert!(scan_ids(store, &key(2000), &key(1000), false).is_empty());
        assert!(scan_ids(store, &key(2000), &key(1000), true).is_empty());
        Ok(())
    })
}

#[test]
fn cursor_pagination() -> TestResult {
    on_each_store(|store| {
        let mut t = store.start_transaction()?;
        for i in 0..1000 {
            t.aux_put(&key(i), b"");
        }
        t.commit()?;

        // Forward pages: the next page starts just after the cursor key.
        let mut pages = Vec::new();
        let mut lower = key(0);
        loop {
            let page = scan_ids(store, &lower, b"list0", false)
                .into_iter()
                .take(300)
                .collect::<Vec<_>>();
            let Some(last) = page.last() else {
                break;
            };
            lower = key(*last);
            lower.push(0);
            pages.push(page);
        }
        assert_eq!(pages.len(), 4);
        assert_eq!(pages.concat(), (0..1000).collect::<Vec<_>>());

        // Reverse pages: the next page ends just before the cursor key.
        let mut pages = Vec::new();
        let mut upper = b"list0".to_vec();
        loop {
            let page = scan_ids(store, b"list/", &upper, true)
                .into_iter()
                .take(300)
                .collect::<Vec<_>>();
            let Some(last) = page.last() else {
                break;
            };
            upper = key(*last);
            pages.push(page);
        }
        assert_eq!(pages.len(), 4);
        assert_eq!(pages.concat(), (0..1000).rev().collect::<Vec<_>>());
        Ok(())
    })
}

#[test]
fn scan_is_on_a_snapshot() -> TestResult {
    on_each_store(|store| {
        let mut t = store.start_transaction()?;
        for i in 0..1000 {
            t.aux_put(&key(i), b"old");
        }
        t.commit()?;
        let mut forward = store.aux_scan(&key(0), &key(2000), false);
        let reverse = store.aux_scan(&key(0), &key(2000), true);
        // Read a part of the forward scan before the next commit.
        assert_eq!(forward.next().unwrap()?, (key(0), b"old".to_vec()));

        let mut t = store.start_transaction()?;
        for i in 0..1000 {
            if i % 2 == 0 {
                t.aux_delete(&key(i));
            } else {
                t.aux_put(&key(i), b"new");
            }
        }
        t.aux_put(&key(1500), b"new");
        t.commit()?;

        // The open scans still see the old state.
        let rest = forward.collect::<Result<Vec<_>, _>>()?;
        assert_eq!(rest.len(), 999);
        assert!(rest.iter().all(|(_, v)| v == b"old"));
        let all = reverse.collect::<Result<Vec<_>, _>>()?;
        assert_eq!(all.len(), 1000);
        assert!(all.iter().all(|(_, v)| v == b"old"));

        // New scans see the new state.
        let all = store
            .aux_scan(&key(0), &key(2000), false)
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(all.len(), 501);
        assert!(all.iter().all(|(_, v)| v == b"new"));
        assert_eq!(store.aux_get(&key(0))?, None);
        assert_eq!(store.aux_get(&key(1))?, Some(b"new".to_vec()));
        Ok(())
    })
}

#[test]
fn many_commits_during_an_open_scan() -> TestResult {
    on_each_store(|store| {
        let mut t = store.start_transaction()?;
        t.aux_put(b"k", b"0");
        t.commit()?;
        let scan = store.aux_scan(b"", b"", false);
        for i in 1..=100 {
            let mut t = store.start_transaction()?;
            t.aux_put(b"k", i.to_string().as_bytes());
            t.aux_merge_add(b"n", 1);
            t.commit()?;
        }
        assert_eq!(
            scan.collect::<Result<Vec<_>, _>>()?,
            [(b"k".to_vec(), b"0".to_vec())]
        );
        assert_eq!(store.aux_get(b"k")?, Some(b"100".to_vec()));
        assert_eq!(store.aux_count(b"n")?, 100);
        // After the scan is dropped, more commits keep the latest values.
        let mut t = store.start_transaction()?;
        t.aux_merge_add(b"n", 1);
        t.aux_delete(b"k");
        t.commit()?;
        assert_eq!(store.aux_get(b"k")?, None);
        assert_eq!(store.aux_count(b"n")?, 101);
        assert_eq!(store.aux_scan(b"", b"", false).count(), 1);
        Ok(())
    })
}

#[test]
fn single_delete_then_scan() -> TestResult {
    on_each_store(|store| {
        let mut t = store.start_transaction()?;
        for i in 0..100 {
            t.aux_put(&key(i), b"x");
        }
        t.commit()?;
        let mut t = store.start_transaction()?;
        for i in (0..100).filter(|i| i % 3 == 0) {
            t.aux_single_delete(&key(i));
        }
        t.commit()?;
        let expected: Vec<u32> = (0..100).filter(|i| i % 3 != 0).collect();
        assert_eq!(scan_ids(store, &key(0), &key(100), false), expected);
        let mut reversed = expected.clone();
        reversed.reverse();
        assert_eq!(scan_ids(store, &key(0), &key(100), true), reversed);

        // Still correct after a compaction (single deletes meet their puts).
        store.compact_aux_range(&key(0), &key(100))?;
        assert_eq!(scan_ids(store, &key(0), &key(100), false), expected);
        assert_eq!(store.aux_get(&key(3))?, None);
        assert_eq!(store.aux_get(&key(4))?, Some(b"x".to_vec()));

        // Put again after single delete, then single delete again.
        let mut t = store.start_transaction()?;
        t.aux_put(&key(3), b"y");
        t.commit()?;
        assert_eq!(store.aux_get(&key(3))?, Some(b"y".to_vec()));
        let mut t = store.start_transaction()?;
        t.aux_single_delete(&key(3));
        t.commit()?;
        assert_eq!(store.aux_get(&key(3))?, None);
        store.compact_aux_range(b"", b"")?;
        assert_eq!(store.aux_get(&key(3))?, None);
        Ok(())
    })
}

#[test]
fn delete() -> TestResult {
    on_each_store(|store| {
        // A key with more than one put: aux_delete removes it.
        let mut t = store.start_transaction()?;
        t.aux_put(b"k", b"1");
        t.commit()?;
        let mut t = store.start_transaction()?;
        t.aux_put(b"k", b"2");
        t.commit()?;
        let mut t = store.start_transaction()?;
        t.aux_delete(b"k");
        assert_eq!(t.aux_get(b"k")?, None);
        t.commit()?;
        assert_eq!(store.aux_get(b"k")?, None);
        store.compact_aux_range(b"", b"")?;
        assert_eq!(store.aux_get(b"k")?, None);

        // Put, delete and put again in the same transaction.
        let mut t = store.start_transaction()?;
        t.aux_put(b"k", b"3");
        t.aux_delete(b"k");
        assert_eq!(t.aux_get(b"k")?, None);
        t.aux_put(b"k", b"4");
        assert_eq!(t.aux_get(b"k")?, Some(b"4".to_vec()));
        t.commit()?;
        assert_eq!(store.aux_get(b"k")?, Some(b"4".to_vec()));

        // Delete of a missing key does nothing.
        let mut t = store.start_transaction()?;
        t.aux_delete(b"missing");
        t.commit()?;
        assert_eq!(store.aux_get(b"missing")?, None);

        // A counter: aux_delete resets it, and later merges start again from 0.
        let mut t = store.start_transaction()?;
        t.aux_merge_add(b"counter", 5);
        t.commit()?;
        let mut t = store.start_transaction()?;
        t.aux_merge_add(b"counter", 2);
        t.commit()?;
        assert_eq!(store.aux_count(b"counter")?, 7);
        let mut t = store.start_transaction()?;
        t.aux_delete(b"counter");
        assert_eq!(t.aux_count(b"counter")?, 0);
        t.commit()?;
        assert_eq!(store.aux_count(b"counter")?, 0);
        assert_eq!(store.aux_get(b"counter")?, None);
        store.compact_aux_range(b"", b"")?;
        assert_eq!(store.aux_count(b"counter")?, 0);
        let mut t = store.start_transaction()?;
        t.aux_merge_add(b"counter", 1);
        t.commit()?;
        assert_eq!(store.aux_count(b"counter")?, 1);

        // Delete and merge in the same transaction.
        let mut t = store.start_transaction()?;
        t.aux_delete(b"counter");
        t.aux_merge_add(b"counter", -4);
        assert_eq!(t.aux_count(b"counter")?, -4);
        t.commit()?;
        assert_eq!(store.aux_count(b"counter")?, -4);

        assert_eq!(
            store
                .aux_scan(b"", b"", false)
                .map(|e| e.map(|(k, _)| k))
                .collect::<Result<Vec<_>, _>>()?,
            [b"counter".to_vec(), b"k".to_vec()]
        );
        Ok(())
    })
}

#[test]
fn merge_counters() -> TestResult {
    on_each_store(|store| {
        assert_eq!(store.aux_count(b"count/a")?, 0);
        for delta in [1, 1, 1, -1] {
            let mut t = store.start_transaction()?;
            t.aux_merge_add(b"count/a", delta);
            t.commit()?;
        }
        assert_eq!(store.aux_count(b"count/a")?, 2);

        // Negative counters: two's-complement wrapping of the u64 add operator.
        let mut t = store.start_transaction()?;
        t.aux_merge_add(b"count/b", -1);
        t.aux_merge_add(b"count/b", -1);
        t.aux_merge_add(b"count/b", -1);
        // Own writes include pending merges, also on top of the committed state.
        assert_eq!(t.aux_count(b"count/b")?, -3);
        t.aux_merge_add(b"count/a", 10);
        assert_eq!(t.aux_count(b"count/a")?, 12);
        t.commit()?;
        assert_eq!(store.aux_count(b"count/b")?, -3);
        assert_eq!(store.aux_count(b"count/a")?, 12);

        // A merge on top of a put value.
        let mut t = store.start_transaction()?;
        t.aux_put(b"count/c", &7_i64.to_le_bytes());
        t.aux_merge_add(b"count/c", -10);
        assert_eq!(t.aux_count(b"count/c")?, -3);
        t.commit()?;
        assert_eq!(store.aux_count(b"count/c")?, -3);

        // Back to exactly zero.
        let mut t = store.start_transaction()?;
        t.aux_merge_add(b"count/b", 3);
        t.commit()?;
        assert_eq!(store.aux_count(b"count/b")?, 0);
        assert_eq!(
            store.aux_get(b"count/b")?,
            Some(0_i64.to_le_bytes().to_vec())
        );

        // Extreme values wrap like i64.
        let mut t = store.start_transaction()?;
        t.aux_merge_add(b"count/max", i64::MAX);
        t.aux_merge_add(b"count/max", 1);
        t.commit()?;
        assert_eq!(store.aux_count(b"count/max")?, i64::MIN);

        // A non-counter value is reported as corruption by aux_count.
        let mut t = store.start_transaction()?;
        t.aux_put(b"not-a-counter", b"abc");
        assert!(matches!(
            t.aux_count(b"not-a-counter"),
            Err(StorageError::Corruption(_))
        ));
        t.commit()?;
        assert!(matches!(
            store.aux_count(b"not-a-counter"),
            Err(StorageError::Corruption(_))
        ));
        // A merge on a non-counter value starts from 0.
        let mut t = store.start_transaction()?;
        t.aux_merge_add(b"not-a-counter", 2);
        t.commit()?;
        assert_eq!(store.aux_count(b"not-a-counter")?, 2);

        // Counters are visible to scans as 8-byte values.
        let (_, v) = store
            .aux_scan(b"count/a", b"count/b", false)
            .next()
            .unwrap()?;
        assert_eq!(v, 12_i64.to_le_bytes());
        Ok(())
    })
}

#[test]
fn compact_aux_range_keeps_data() -> TestResult {
    on_each_store(|store| {
        for batch in 0..10 {
            let mut t = store.start_transaction()?;
            for i in (batch * 1000)..((batch + 1) * 1000) {
                t.aux_put(&key(i), &[1; 32]);
                t.aux_merge_add(b"total", 1);
            }
            t.commit()?;
        }
        let mut t = store.start_transaction()?;
        for i in 2000..8000 {
            t.aux_single_delete(&key(i));
            t.aux_merge_add(b"total", -1);
        }
        t.commit()?;
        store.compact_aux_range(&key(2000), &key(8000))?;
        store.compact_aux_range(b"", b"")?; // whole space
        assert_eq!(store.aux_count(b"total")?, 4000);
        let ids = scan_ids(store, &key(0), &key(10_000), false);
        assert_eq!(ids.len(), 4000);
        assert_eq!(ids[1999], 1999);
        assert_eq!(ids[2000], 8000);
        let ids = scan_ids(store, &key(0), &key(10_000), true);
        assert_eq!((ids[0], ids[1999], ids[2000]), (9999, 8000, 1999));
        Ok(())
    })
}

#[test]
fn aux_is_separate_from_quads() -> TestResult {
    on_each_store(|store| {
        let mut t = store.start_transaction()?;
        t.insert(&quad("a"));
        t.aux_put(b"k", b"v");
        t.commit()?;
        assert_eq!(store.len()?, 1);
        store.clear()?;
        assert_eq!(store.len()?, 0);
        assert_eq!(store.aux_get(b"k")?, Some(b"v".to_vec()));
        Ok(())
    })
}

#[cfg(all(not(target_family = "wasm"), feature = "rocksdb"))]
#[test]
fn on_disk_data_survives_reopen_and_compaction() -> TestResult {
    let dir = TempDir::new()?;
    {
        let store = Store::open(dir.path())?;
        let mut t = store.start_transaction()?;
        t.aux_merge_add(b"count/a", 12);
        t.aux_merge_add(b"count/b", -3);
        t.aux_put(b"list/1", b"one");
        t.aux_put(b"list/2", b"two");
        t.commit()?;
        let mut t = store.start_transaction()?;
        t.aux_single_delete(b"list/2");
        t.commit()?;
    }
    {
        // After reopen (operands replayed from the WAL)
        let store = Store::open(dir.path())?;
        assert_eq!(store.aux_count(b"count/a")?, 12);
        assert_eq!(store.aux_count(b"count/b")?, -3);
        let mut t = store.start_transaction()?;
        t.aux_merge_add(b"count/a", -1);
        t.commit()?;
        // After a full compaction (operands folded into one value in SST files)
        store.optimize()?;
        assert_eq!(store.aux_count(b"count/a")?, 11);
        assert_eq!(store.aux_count(b"count/b")?, -3);
        assert_eq!(store.aux_get(b"list/1")?, Some(b"one".to_vec()));
        assert_eq!(store.aux_get(b"list/2")?, None);
    }
    {
        let store = Store::open_read_only(dir.path())?;
        assert_eq!(store.aux_count(b"count/a")?, 11);
        assert_eq!(
            store
                .aux_scan(b"list/", b"list0", false)
                .collect::<Result<Vec<_>, _>>()?,
            [(b"list/1".to_vec(), b"one".to_vec())]
        );
        store.aux_get(b"list/1")?;
    }
    Ok(())
}
