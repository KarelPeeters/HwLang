use std::num::NonZeroUsize;

/// Compute the [Damerau–Levenshtein distance](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)
/// between two strings.
pub fn edit_distance(a: &str, b: &str) -> usize {
    // possibly swap stings to minimize buffer size
    let (a, b, b_len) = {
        let a_len = a.chars().count();
        let b_len = b.chars().count();
        if a_len < b_len { (b, a, a_len) } else { (a, b, b_len) }
    };

    // dynamic programming
    let mut row_prev_prev = vec![0; b_len + 1];
    let mut row_prev = (0..=b_len).collect::<Vec<_>>();
    let mut row_curr = vec![0; b_len + 1];

    let mut prev_ca = None;
    for (ia, ca) in a.chars().enumerate() {
        row_curr[0] = ia + 1;

        let mut prev_cb = None;
        for (ib, cb) in b.chars().enumerate() {
            let cost_sub = row_prev[ib] + usize::from(ca != cb);
            let cost_delete = row_prev[ib + 1] + 1;
            let cost_insert = row_curr[ib] + 1;
            let cost_swap = if prev_ca == Some(cb) && prev_cb == Some(ca) {
                row_prev_prev[ib - 1] + 1
            } else {
                usize::MAX
            };

            let cost = cost_sub.min(cost_delete).min(cost_insert).min(cost_swap);
            row_curr[ib + 1] = cost;

            prev_cb = Some(cb);
        }

        std::mem::swap(&mut row_prev_prev, &mut row_prev);
        std::mem::swap(&mut row_prev, &mut row_curr);
        prev_ca = Some(ca);
    }
    row_prev[b_len]
}

/// Build a diagnostic hint message to suggest `options` as alternatives to `target`.
/// The resulting message contains up to `max_count` suggestions,
/// sorted by similarity and only kept if they're similar enough.
pub fn hint_options_by_edit_distance<'s>(
    target: &str,
    options: impl IntoIterator<Item = &'s str>,
    max_count: NonZeroUsize,
) -> Option<String> {
    // only keep options that are close enough to plausibly be a typo of the target,
    //   allowing an edit distance of `(len + 2) / 3`
    let max_distance = target.chars().count().div_ceil(3);

    // sort by edit distance, with lexicographic tiebreak
    let mut options = options
        .into_iter()
        .map(|option| (edit_distance(target, option), option))
        .filter(|&(distance, _)| distance <= max_distance)
        .collect::<Vec<_>>();
    options.sort_unstable();
    options.dedup();

    // only keep top options
    let total_options_count = options.len();
    options.truncate(max_count.get());
    let options_dropped = total_options_count - options.len();

    // build message
    let options = options
        .iter()
        .map(|(_, option)| format!("`{option}`"))
        .collect::<Vec<_>>();
    let msg = match options.as_slice() {
        [] => return None,
        _ if options_dropped > 0 => format!("did you mean one of {}, ...?", options.join(", ")),
        [single] => format!("did you mean {single}?"),
        [init @ .., last] => format!("did you mean {} or {last}?", init.join(", ")),
    };
    Some(msg)
}

#[cfg(test)]
mod tests {
    use crate::util::edit_distance::{edit_distance, hint_options_by_edit_distance};
    use std::num::NonZeroUsize;

    #[test]
    fn test_edit_distance() {
        // basic cases
        assert_eq!(edit_distance("", ""), 0);
        assert_eq!(edit_distance("abc", ""), 3);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("abc", "abc"), 0);
        assert_eq!(edit_distance("kitten", "sitting"), 3);
        assert_eq!(edit_distance("flaw", "lawn"), 2);
        assert_eq!(edit_distance("héllo", "hello"), 1);

        // swap cases
        assert_eq!(edit_distance("ab", "ba"), 1);
        assert_eq!(edit_distance("lbi", "lib"), 1);
        assert_eq!(edit_distance("counetr", "counter"), 1);
        assert_eq!(edit_distance("abcd", "badc"), 2);
        assert_eq!(edit_distance("héllo", "ehllo"), 2);
        assert_eq!(edit_distance("éh", "hé"), 1);

        // sadly no multi-swap support yet
        assert_eq!(edit_distance("ca", "abc"), 3);
        assert_eq!(edit_distance("abc", "ca"), 3);
    }

    #[test]
    fn test_hint_options() {
        let c = |c| NonZeroUsize::new(c).unwrap();
        let hint = |target, options: &[&'static str], max_count| {
            hint_options_by_edit_distance(target, options.iter().copied(), c(max_count))
        };

        // nothing similar
        assert_eq!(hint("test", &[], 2), None);
        assert_eq!(hint("test", &["banana"], 2), None);

        // formatting depending on the number of options
        assert_eq!(hint("tes", &["test"], 2).as_deref(), Some("did you mean `test`?"));
        assert_eq!(
            hint("countr", &["counter", "count"], 2).as_deref(),
            Some("did you mean `count` or `counter`?")
        );
        assert_eq!(
            hint("cnt", &["cat", "cut", "cot"], 3).as_deref(),
            Some("did you mean `cat`, `cot` or `cut`?")
        );
        assert_eq!(
            hint("countr", &["counter", "count", "countrr"], 2).as_deref(),
            Some("did you mean one of `count`, `counter`, ...?")
        );

        // max distance is `(len + 2) / 3`
        assert_eq!(hint("ab", &["ax", "xy"], 2).as_deref(), Some("did you mean `ax`?"));
        assert_eq!(
            hint("abcd", &["abxy", "xbcy", "xycd"], 5).as_deref(),
            Some("did you mean `abxy`, `xbcy` or `xycd`?")
        );
        assert_eq!(hint("abcd", &["axyz"], 5), None);
    }
}
