use inkwell::targets::TargetData;
use inkwell::types::AnyType;
use std::alloc::Layout;
use std::ffi::c_void;
use std::fmt::{Display, Formatter};
use std::ptr::NonNull;
use std::{alloc, fmt};

pub struct Buffer {
    layout: Layout,
    ptr: NonNull<u8>,
}

#[derive(Debug)]
pub enum BufferError {
    InvalidLayout,
    AllocationFailed,
}

impl Buffer {
    pub fn new_zeroed(target: &TargetData, ty: &dyn AnyType) -> Result<Buffer, BufferError> {
        let size = target
            .get_store_size(ty)
            .try_into()
            .map_err(|_| BufferError::InvalidLayout)?;
        let align = target
            .get_abi_alignment(ty)
            .try_into()
            .map_err(|_| BufferError::InvalidLayout)?;
        let layout = Layout::from_size_align(size, align).map_err(|_| BufferError::InvalidLayout)?;

        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).ok_or(BufferError::AllocationFailed)?;

        Ok(Buffer { layout, ptr })
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr.as_ptr() as *mut c_void
    }

    pub unsafe fn read<T>(&self, offset_bytes: usize) -> T {
        unsafe { std::ptr::read::<T>(self.as_ptr().add(offset_bytes) as *const T) }
    }

    pub unsafe fn write<T>(&self, offset_bytes: usize, value: T) {
        unsafe { std::ptr::write::<T>(self.as_ptr().add(offset_bytes) as *mut T, value) }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { alloc::dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

impl Display for BufferError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            BufferError::InvalidLayout => write!(f, "invalid layout"),
            BufferError::AllocationFailed => write!(f, "allocation failed"),
        }
    }
}
