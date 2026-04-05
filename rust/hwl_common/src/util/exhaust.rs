/// Utility to exhaustively explore all combinations of choices, mostly useful for testing.
///
/// The sequence of choices (both the kind and number) is allowed to depend on the result of earlier choices,
/// but must otherwise be deterministic between multiple calls to `f`.
pub fn exhaust(mut f: impl FnMut(&mut Exhaust)) {
    let mut ex = Exhaust {
        iteration: 0,
        position: 0,
        stack: vec![],
    };
    loop {
        f(&mut ex);
        ex = match ex.increment() {
            Ok(e) => e,
            Err(()) => break,
        };
    }
}

#[derive(Debug)]
pub struct Exhaust {
    iteration: u64,
    position: usize,
    stack: Vec<Entry>,
}

#[derive(Debug)]
struct Entry {
    next_value: u64,
    max_value: u64,
}

impl Exhaust {
    pub fn choose(&mut self, n: u64) -> u64 {
        assert!(n > 0, "invalid choice, cannot choose from empty set");
        let max_value = n - 1;

        assert!(self.position <= self.stack.len());
        if self.position == self.stack.len() {
            self.stack.push(Entry {
                next_value: 0,
                max_value,
            });

            self.position += 1;
            0
        } else {
            let entry = &mut self.stack[self.position];
            assert_eq!(entry.max_value, max_value, "invalid choice, set size changed");

            self.position += 1;
            entry.next_value
        }
    }

    pub fn choose_range(&mut self, start: u64, end: u64) -> u64 {
        assert!(start < end, "invalid choice, range is empty");
        self.choose(end - start) + start
    }

    pub fn choose_bool(&mut self) -> bool {
        self.choose(2) == 1
    }

    pub fn iteration(&self) -> u64 {
        self.iteration
    }

    fn increment(mut self) -> Result<Self, ()> {
        assert_eq!(self.position, self.stack.len());
        self.position = 0;
        self.iteration += 1;

        for (i, slot) in self.stack.iter_mut().enumerate().rev() {
            if slot.next_value != slot.max_value {
                slot.next_value += 1;
                self.stack.truncate(i + 1);
                return Ok(self);
            }
        }

        Err(())
    }
}

#[cfg(test)]
mod tests {
    use crate::util::exhaust::exhaust;

    #[test]
    fn basic() {
        let mut actual = vec![];
        exhaust(|ex| {
            actual.push((ex.choose(3), ex.choose(1), ex.choose_bool()));
        });

        let mut expected = vec![];
        for a in 0..3 {
            for b in 0..1 {
                for c in [false, true] {
                    expected.push((a, b, c));
                }
            }
        }

        assert_eq!(actual, expected);
    }

    #[test]
    fn dependant() {
        let mut actual = vec![];
        exhaust(|ex| {
            let mut line = vec![];
            let n = ex.choose(3);

            line.push(n);
            for _ in 0..n {
                line.push(ex.choose(2));
            }
            actual.push(line);
        });

        let expected = vec![
            vec![0],
            vec![1, 0],
            vec![1, 1],
            vec![2, 0, 0],
            vec![2, 0, 1],
            vec![2, 1, 0],
            vec![2, 1, 1],
        ];
        assert_eq!(actual, expected);
    }
}
