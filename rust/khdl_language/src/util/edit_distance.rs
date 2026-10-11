/// Compute the [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) between two strings.
pub fn edit_distance(a: &str, b: &str) -> usize {
    // possibly swap stings to minimize buffer size
    let (a, b, b_len) = {
        let a_len = a.chars().count();
        let b_len = b.chars().count();
        if a_len < b_len { (b, a, a_len) } else { (a, b, b_len) }
    };

    // dynamic programming
    let mut row = (0..=b_len).collect::<Vec<_>>();
    for (ia, ca) in a.chars().enumerate() {
        let mut diag = row[0];
        row[0] = ia + 1;
        for (ib, cb) in b.chars().enumerate() {
            let cost_substitute = diag + usize::from(ca != cb);
            let cost_delete = row[ib + 1] + 1;
            let cost_insert = row[ib] + 1;
            let cost = cost_substitute.min(cost_delete).min(cost_insert);

            diag = row[ib + 1];
            row[ib + 1] = cost;
        }
    }
    row[b_len]
}

/// Select the (at most) `max_count` options that are most similar to `target`, most similar first.
/// Ties are broken alphabetically, so the result is deterministic.
pub fn most_similar<'s>(target: &str, options: impl IntoIterator<Item = &'s str>, max_count: usize) -> Vec<&'s str> {
    let mut options = options
        .into_iter()
        .map(|option| (edit_distance(target, option), option))
        .collect::<Vec<_>>();
    options.sort_unstable();
    options.truncate(max_count);
    options.into_iter().map(|(_, option)| option).collect()
}

#[cfg(test)]
mod tests {
    use crate::util::edit_distance::{edit_distance, most_similar};

    #[test]
    fn test_edit_distance() {
        assert_eq!(edit_distance("", ""), 0);
        assert_eq!(edit_distance("abc", ""), 3);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("abc", "abc"), 0);
        assert_eq!(edit_distance("kitten", "sitting"), 3);
        assert_eq!(edit_distance("flaw", "lawn"), 2);
        assert_eq!(edit_distance("ab", "ba"), 2);
        assert_eq!(edit_distance("héllo", "hello"), 1);
    }

    #[test]
    fn test_most_similar() {
        let options = ["apple", "apply", "banana", "maple", "ape"];
        assert_eq!(most_similar("appel", options, 3), vec!["ape", "apple", "apply"]);
        assert_eq!(most_similar("x", options, 0), Vec::<&str>::new());
        assert_eq!(most_similar("x", ["b", "a"], 5), vec!["a", "b"]);
    }
}
