use crate::mid::ir::{IrStructType, IrType};
use crate::util::big_int::BigUint;
use crate::util::range::ClosedRange;
use crate::util::sparse_change_array::SparseChangeArray;
use crate::util::{Never, ResultNeverExt};
use hwl_util::swrite;
use itertools::{Itertools, enumerate};
use std::fmt::Debug;
use std::ops::ControlFlow;
use unwrap_match::unwrap_match;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum IrMask<T> {
    Scalar(T),
    Array(Box<SparseChangeArray<IrMask<T>>>),
    TupleOrStruct(Vec<IrMask<T>>),
}

impl<T> IrMask<T> {
    pub fn new(ty: &IrType, init: T) -> IrMask<T>
    where
        T: Clone,
    {
        match ty {
            IrType::Bool | IrType::Int(_) | IrType::Enum(_) => IrMask::Scalar(init),
            IrType::Array(ty_inner, len) => {
                let array = if len.is_zero() {
                    SparseChangeArray::new_empty()
                } else {
                    let mask_inner = IrMask::new(ty_inner, init);
                    SparseChangeArray::new(len.clone(), mask_inner)
                };
                IrMask::Array(Box::new(array))
            }
            IrType::Tuple(ty_fields) => {
                let mask_fields = ty_fields
                    .iter()
                    .map(|ty_field| IrMask::new(ty_field, init.clone()))
                    .collect_vec();
                IrMask::TupleOrStruct(mask_fields)
            }
            IrType::Struct(ty_info) => {
                let IrStructType {
                    ty: _,
                    debug_info_name: _,
                    fields,
                } = ty_info;
                let mask_fields = fields
                    .values()
                    .map(|ty_field| IrMask::new(ty_field, init.clone()))
                    .collect_vec();
                IrMask::TupleOrStruct(mask_fields)
            }
        }
    }

    pub fn canonicalize(&mut self)
    where
        T: Eq,
    {
        match self {
            IrMask::Scalar(_) => {}
            IrMask::Array(slf) => {
                slf.for_each_block_mut(|_, inner| {
                    inner.canonicalize();
                    ControlFlow::Continue(())
                })
                .remove_never();
                slf.canonicalize()
            }
            IrMask::TupleOrStruct(slf) => slf.iter_mut().for_each(Self::canonicalize),
        }
    }

    pub fn for_each_path<B>(
        &self,
        ty: &IrType,
        mut cond: impl FnMut(&T) -> bool,
        mut report: impl FnMut(&str) -> ControlFlow<B>,
    ) -> ControlFlow<B>
    where
        T: Debug,
    {
        fn g<T: Debug, B>(
            slf: &IrMask<T>,
            ty: &IrType,
            prefix: &mut String,
            mut cond: &mut impl FnMut(&T) -> bool,
            report: &mut impl FnMut(&str) -> ControlFlow<B>,
        ) -> ControlFlow<B> {
            // stop at scalars
            if let IrMask::Scalar(slf) = slf {
                if cond(slf) {
                    report(prefix)?;
                }
                return ControlFlow::Continue(());
            }

            // stop when uniform, to avoid creating unnecessarily long prefixes
            if slf.all(&mut cond) {
                report(prefix)?;
                return ControlFlow::Continue(());
            }

            // branch on compounds
            let prefix_len = prefix.len();
            match ty {
                IrType::Bool | IrType::Int(_) | IrType::Enum(_) => {
                    unreachable!("scalars already handled")
                }
                IrType::Array(ty_inner, _) => {
                    let slf = unwrap_match!(slf, IrMask::Array(slf) => slf);
                    slf.for_each_block(|range, mask| {
                        let range_full = ClosedRange {
                            start: &BigUint::ZERO,
                            end: slf.len(),
                        };
                        if range == range_full {
                            swrite!(prefix, "[..]")
                        } else if let Some(single) = range.as_single() {
                            swrite!(prefix, "[{single}]")
                        } else {
                            swrite!(prefix, "[{range}]")
                        };
                        g(mask, ty_inner, prefix, cond, report)?;
                        prefix.truncate(prefix_len);
                        ControlFlow::Continue(())
                    })?;
                }
                IrType::Tuple(ty_fields) => {
                    let slf = unwrap_match!(slf, IrMask::TupleOrStruct(slf) => slf);
                    assert_eq!(slf.len(), ty_fields.len());

                    for (field_index, field_ty) in enumerate(ty_fields) {
                        swrite!(prefix, ".{field_index}");
                        g(&slf[field_index], field_ty, prefix, cond, report)?;
                        prefix.truncate(prefix_len);
                    }
                }
                IrType::Struct(ty_info) => {
                    let slf = unwrap_match!(slf, IrMask::TupleOrStruct(slf) => slf);
                    assert_eq!(slf.len(), ty_info.fields.len());

                    for (field_index, (field_name, field_ty)) in enumerate(&ty_info.fields) {
                        swrite!(prefix, ".{field_name}");
                        g(&slf[field_index], field_ty, prefix, cond, report)?;
                        prefix.truncate(prefix_len);
                    }
                }
            }

            ControlFlow::Continue(())
        }

        let mut prefix = String::new();
        g(self, ty, &mut prefix, &mut cond, &mut report)
    }

    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.for_each_leaf_mut(|leaf| {
            *leaf = value.clone();
            ControlFlow::Continue(())
        })
        .remove_never();
    }

    pub fn for_each_leaf<B>(&self, mut f: impl FnMut(&T) -> ControlFlow<B>) -> ControlFlow<B> {
        fn for_each_leaf_impl<T, B>(slf: &IrMask<T>, f: &mut impl FnMut(&T) -> ControlFlow<B>) -> ControlFlow<B> {
            match slf {
                IrMask::Scalar(slf) => f(slf),
                IrMask::Array(slf) => slf.for_each_block(|_, m| for_each_leaf_impl(m, f)),
                IrMask::TupleOrStruct(values) => values.iter().try_for_each(|m| for_each_leaf_impl(m, f)),
            }
        }
        for_each_leaf_impl(self, &mut f)
    }

    pub fn for_each_leaf_mut<B>(&mut self, mut f: impl FnMut(&mut T) -> ControlFlow<B>) -> ControlFlow<B> {
        fn for_each_leaf_mut_impl<T, B>(
            slf: &mut IrMask<T>,
            f: &mut impl FnMut(&mut T) -> ControlFlow<B>,
        ) -> ControlFlow<B> {
            match slf {
                IrMask::Scalar(slf) => f(slf),
                IrMask::Array(slf) => slf.for_each_block_mut(|_, m| for_each_leaf_mut_impl(m, f)),
                IrMask::TupleOrStruct(values) => values.iter_mut().try_for_each(|m| for_each_leaf_mut_impl(m, f)),
            }
        }
        for_each_leaf_mut_impl(self, &mut f)
    }

    pub fn any(&self, mut f: impl FnMut(&T) -> bool) -> bool {
        self.for_each_leaf(|v| {
            if f(v) {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })
        .is_break()
    }

    pub fn all(&self, mut f: impl FnMut(&T) -> bool) -> bool {
        !self.any(|v| !f(v))
    }

    pub fn zip2_for_each<U: Debug>(a: &IrMask<T>, b: &IrMask<U>, f: &mut impl FnMut(&T, &U)) {
        match a {
            IrMask::Scalar(a) => {
                let b = unwrap_match!(b, IrMask::Scalar(b) => b);
                f(a, b)
            }
            IrMask::Array(a) => {
                let b = unwrap_match!(b, IrMask::Array(b) => b);
                SparseChangeArray::zip2_for_each_block::<_, Never>(a, b, |_, a, b| {
                    Self::zip2_for_each(a, b, f);
                    ControlFlow::Continue(())
                })
                .remove_never();
            }
            IrMask::TupleOrStruct(a) => {
                let b = unwrap_match!(b, IrMask::TupleOrStruct(b) => b);
                for i in 0..a.len() {
                    Self::zip2_for_each(&a[i], &b[i], f);
                }
            }
        }
    }

    pub fn zip2_for_each_mut<U: Debug>(a: &mut IrMask<T>, b: &IrMask<U>, f: &mut impl FnMut(&mut T, &U))
    where
        T: Clone,
    {
        IrMask::zip3_for_each_mut::<U, Never>(a, Some(b), None, &mut |a, b, c| {
            let b = b.unwrap();
            match c {
                None => {}
                Some(never) => never.unreachable(),
            }
            f(a, b)
        })
    }

    pub fn zip3_for_each_mut<U: Debug, V: Debug>(
        a: &mut IrMask<T>,
        b: Option<&IrMask<U>>,
        c: Option<&IrMask<V>>,
        f: &mut impl FnMut(&mut T, Option<&U>, Option<&V>),
    ) where
        T: Clone,
    {
        match a {
            IrMask::Scalar(a) => {
                let b = b.map(|b| unwrap_match!(b, IrMask::Scalar(b) => b));
                let c = c.map(|c| unwrap_match!(c, IrMask::Scalar(c) => c));
                f(a, b, c)
            }
            IrMask::Array(a) => {
                let b = b.map(|b| unwrap_match!(b, IrMask::Array(b) => &**b));
                let c = c.map(|c| unwrap_match!(c, IrMask::Array(c) => &**c));
                SparseChangeArray::zip3_for_each_block_mut::<_, _, Never>(a, b, c, |_, a, b, c| {
                    Self::zip3_for_each_mut(a, b, c, f);
                    ControlFlow::Continue(())
                })
                .remove_never();
            }
            IrMask::TupleOrStruct(a) => {
                let b = b.map(|b| unwrap_match!(b, IrMask::TupleOrStruct(b) => b));
                let c = c.map(|c| unwrap_match!(c, IrMask::TupleOrStruct(c) => c));

                for i in 0..a.len() {
                    let b = b.map(|b| &b[i]);
                    let c = c.map(|c| &c[i]);
                    Self::zip3_for_each_mut(&mut a[i], b, c, f);
                }
            }
        }
    }
}
