use itertools::zip_eq;
use std::hash::{Hash, Hasher};
use std::iter::zip;
use std::sync::Arc;

/// An n-dimensional array, with support for indexing and slicing.
#[derive(Debug)]
pub struct NdArray<T> {
    shared: Arc<NdArrayShared<T>>,

    // shape: the number of axes to skip in shared.shape, and the len to replace the first axis with
    axes_skipped: usize,
    first_len: Option<usize>,

    // values: the start and length of the slice of values actually in this array
    flat_start: usize,
    flat_len: usize,
}

#[derive(Debug)]
struct NdArrayShared<T> {
    shape: Vec<usize>,
    values: Vec<T>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum NdArrayShape<'a> {
    Scalar,
    Array { first: usize, rest: &'a [usize] },
}

#[derive(Debug, Copy, Clone)]
pub struct ShapeMismatch;

impl<T> NdArray<T> {
    pub fn new(shape: Vec<usize>, values: Vec<T>) -> Self {
        let len = shape.iter().copied().product();
        assert_eq!(len, values.len());
        NdArray {
            axes_skipped: 0,
            first_len: shape.first().copied(),
            flat_start: 0,
            flat_len: len,
            shared: Arc::new(NdArrayShared { shape, values }),
        }
    }

    pub fn single(value: T) -> Self {
        Self::new(vec![], vec![value])
    }

    pub fn shape(&self) -> NdArrayShape<'_> {
        match self.first_len {
            None => NdArrayShape::Scalar,
            Some(first) => NdArrayShape::Array {
                first,
                rest: &self.shared.shape[self.axes_skipped + 1..],
            },
        }
    }

    /// The length of the first axis, or `None` if the array is zero-dimensional.
    pub fn first_len(&self) -> Option<usize> {
        self.first_len
    }

    /// The number of elements in each sub-array along the first axis.
    fn inner_len(&self) -> usize {
        self.shared.shape[self.axes_skipped + 1..].iter().copied().product()
    }

    /// If this is a scalar value, the value. `None` otherwise.
    pub fn as_single(&self) -> Option<&T> {
        match self.first_len {
            None => Some(&self.flat_values()[0]),
            Some(_) => None,
        }
    }

    pub fn flat_values(&self) -> &[T] {
        &self.shared.values[self.flat_start..self.flat_start + self.flat_len]
    }

    pub fn step_index(&self, index: usize) -> Result<Self, ShapeMismatch> {
        if !self.first_len.is_some_and(|first_len| index < first_len) {
            return Err(ShapeMismatch);
        }

        let axes_skipped = self.axes_skipped + 1;
        let inner_len = self.inner_len();
        Ok(NdArray {
            shared: Arc::clone(&self.shared),
            axes_skipped,
            first_len: self.shared.shape.get(axes_skipped).copied(),
            flat_start: self.flat_start + index * inner_len,
            flat_len: inner_len,
        })
    }

    pub fn step_slice(&self, start: usize, length: usize) -> Result<Self, ShapeMismatch> {
        if !self.first_len.is_some_and(|first_len| start + length <= first_len) {
            return Err(ShapeMismatch);
        }

        let inner_len = self.inner_len();
        Ok(NdArray {
            shared: Arc::clone(&self.shared),
            axes_skipped: self.axes_skipped,
            first_len: Some(length),
            flat_start: self.flat_start + start * inner_len,
            flat_len: length * inner_len,
        })
    }

    pub fn zip<'a, U>(&'a self, other: &'a NdArray<U>) -> Result<impl Iterator<Item = (&'a T, &'a U)>, ShapeMismatch> {
        if self.shape() != other.shape() {
            return Err(ShapeMismatch);
        }
        Ok(zip_eq(self.flat_values(), other.flat_values()))
    }
}

pub fn try_for_each_nd_index<E>(shape: &[usize], mut f: impl FnMut(&[usize]) -> Result<(), E>) -> Result<(), E> {
    if shape.contains(&0) {
        return Ok(());
    }

    let mut index = vec![0; shape.len()];
    loop {
        f(&index)?;

        let mut carry = true;
        for (&s, i) in zip(shape, &mut index).rev() {
            *i += 1;
            if *i < s {
                carry = false;
                break;
            }
            *i = 0;
        }
        if carry {
            return Ok(());
        }
    }
}

impl NdArrayShape<'_> {
    pub fn rank(&self) -> usize {
        match self {
            NdArrayShape::Scalar => 0,
            NdArrayShape::Array { first: _, rest } => 1 + rest.len(),
        }
    }

    pub fn to_vec(&self) -> Vec<usize> {
        match self {
            NdArrayShape::Scalar => vec![],
            NdArrayShape::Array { first, rest } => {
                let mut result = Vec::with_capacity(1 + rest.len());
                result.push(*first);
                result.extend_from_slice(rest);
                result
            }
        }
    }
}

impl<T> Clone for NdArray<T> {
    fn clone(&self) -> Self {
        NdArray {
            shared: Arc::clone(&self.shared),
            axes_skipped: self.axes_skipped,
            first_len: self.first_len,
            flat_start: self.flat_start,
            flat_len: self.flat_len,
        }
    }
}

impl<T: Eq> Eq for NdArray<T> {}

impl<T: PartialEq> PartialEq for NdArray<T> {
    fn eq(&self, other: &Self) -> bool {
        self.shape() == other.shape() && self.flat_values() == other.flat_values()
    }
}

impl<T: Hash> Hash for NdArray<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.shape().hash(state);
        self.flat_values().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use crate::util::nd_array::{NdArray, try_for_each_nd_index};
    use crate::util::{Never, ResultNeverExt};

    #[test]
    fn index_slice() {
        let array = {
            let shape = vec![3, 2];
            let mut values = vec![];
            try_for_each_nd_index::<Never>(&shape, |i| {
                values.push(i.to_vec());
                Ok(())
            })
            .remove_never();
            NdArray::new(shape, values)
        };

        assert_eq!(array.flat_values()[1], vec![0, 1]);
        assert_eq!(array.flat_values()[2], vec![1, 0]);

        let row = array.step_index(1).unwrap();
        assert_eq!(row.shape().to_vec(), vec![2]);
        assert_eq!(row.flat_values(), &[vec![1, 0], vec![1, 1]]);
        assert_eq!(row.step_index(1).unwrap().as_single(), Some(&vec![1, 1]));

        let slice = array.step_slice(1, 2).unwrap();
        assert_eq!(slice.shape().to_vec(), vec![2, 2]);
        assert_eq!(slice.step_index(1).unwrap().flat_values(), &[vec![2, 0], vec![2, 1]]);
        assert_eq!(slice.shape(), NdArray::new(vec![2, 2], vec![(); 4]).shape());
        assert_ne!(slice.shape(), row.shape());

        let empty = array.step_slice(3, 0).unwrap();
        assert_eq!(empty.shape().to_vec(), vec![0, 2]);
        assert!(empty.flat_values().is_empty());

        let single = NdArray::single(5);
        assert_eq!(single.shape().to_vec(), Vec::<usize>::new());
        assert_eq!(single.as_single(), Some(&5));
    }

    #[test]
    fn zero_inner_axis() {
        let array = NdArray::new(vec![3, 0], vec![(); 0]);
        let slice = array.step_slice(1, 2).unwrap();
        assert_eq!(slice.shape().to_vec(), vec![2, 0]);
        assert_eq!(slice.step_index(1).unwrap().shape().to_vec(), vec![0]);

        let array = NdArray::new(vec![0, 2], vec![(); 0]);
        assert_eq!(array.inner_len(), 2);
        assert_eq!(array.step_slice(0, 0).unwrap().shape().to_vec(), vec![0, 2]);
    }

    #[test]
    fn from_fn_zero_length() {
        let array = NdArray::<()>::new(vec![2, 0, 3], vec![]);
        assert_eq!(array.shape().to_vec(), vec![2, 0, 3]);
    }

    #[test]
    fn for_each_nd_index() {
        let cases: &[(&[usize], &[&[usize]])] = &[
            (&[], &[&[]]),
            (&[0], &[]),
            (&[1, 2], &[&[0, 0], &[0, 1]]),
            (&[2, 1], &[&[0, 0], &[1, 0]]),
            (&[2, 0, 3], &[]),
            (&[1, 1, 1], &[&[0, 0, 0]]),
            (&[2, 2], &[&[0, 0], &[0, 1], &[1, 0], &[1, 1]]),
        ];

        for &(shape, expected) in cases {
            let mut actual = vec![];
            try_for_each_nd_index::<Never>(shape, |i| {
                actual.push(i.to_vec());
                Ok(())
            })
            .remove_never();
            assert_eq!(actual, expected, "shape {shape:?}");
        }

        let result = try_for_each_nd_index(&[2, 2], |i| if i == [1, 0] { Err(i.to_vec()) } else { Ok(()) });
        assert_eq!(result, Err(vec![1, 0]));
    }
}
