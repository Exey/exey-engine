//! Buffer wrapper. A [`Buffer`] owns a `vk::Buffer` plus the `vk::DeviceMemory`
//! backing it. Two factory paths cover M2's needs:
//!
//! - [`Buffer::host_visible`] — host-visible + coherent. Map, write, unmap.
//!   Used for the M2 vertex/index data, the staging buffer that uploads
//!   texture pixels, and (M6) the persistently-mapped streaming buffers
//!   in `BigBufferRenderer` — see [`Buffer::map_persistent`] /
//!   [`Buffer::write_at_offset`].
//! - [`Buffer::device_local`] — GPU-local memory. Cannot be mapped from the
//!   host; must be filled via a staging buffer + `vkCmdCopyBuffer`. Unused
//!   today but kept for when static GPU-only assets arrive.
//!
//! AS3 equivalent: `RenderUtil.vertexBuffer` / `RenderUtil.indexBuffer`
//! (`Stage3D` did the host→device transfer transparently behind
//! `uploadFromByteArray`; on Vulkan we have to be explicit).

use anyhow::{Context, Result};
use std::ptr;
use vulkanalia::prelude::v1_0::*;

use super::{Device, Instance, memory};

pub struct Buffer {
    pub handle: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    /// Set by [`Buffer::map_persistent`]. While non-null, the host has the
    /// memory mapped — [`Buffer::write_at_offset`] writes through this
    /// pointer without a per-write map/unmap cycle. M6's `BigBufferRenderer`
    /// uses this to stream vertex / index data each frame.
    ///
    /// Stored as a raw `*mut u8`; the pointer is alive for as long as the
    /// `Buffer` (we unmap in [`Buffer::destroy`]). The `unsafe` contract is
    /// that the caller of `write_at_offset` doesn't keep references to the
    /// mapped range across a queue submit — for HOST_COHERENT memory, that
    /// is the spec's only requirement.
    mapped: *mut u8,
}

// Buffer holds a raw pointer to its mapped GPU memory but does not give the
// rest of the engine any shared access to that pointer; `write_at_offset`
// takes `&self` and writes through it exclusively. From the rest of the
// engine's perspective the Buffer behaves like an owned, single-threaded
// resource — which is how it's used. Sync/Send mirror what we'd get if the
// mapped pointer were instead returned per call.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    /// Host-visible + coherent buffer. "Coherent" means we don't have to
    /// `flush_mapped_memory_ranges` after writing — the spec guarantees the
    /// GPU sees our writes by the next pipeline barrier or queue submit.
    pub fn host_visible(
        instance: &Instance,
        device: &Device,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        Self::create(
            instance,
            device,
            size,
            usage,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }

    /// Device-local buffer. Fastest for the GPU to read; must be filled via
    /// a staging buffer copy.
    pub fn device_local(
        instance: &Instance,
        device: &Device,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        Self::create(
            instance,
            device,
            size,
            usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
    }

    fn create(
        instance: &Instance,
        device: &Device,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        props: vk::MemoryPropertyFlags,
    ) -> Result<Self> {
        let info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let handle = unsafe { device.logical.create_buffer(&info, None) }
            .context("vkCreateBuffer failed")?;

        let req = unsafe { device.logical.get_buffer_memory_requirements(handle) };
        let mem_type =
            memory::find_memory_type(instance, device, req.memory_type_bits, props)?;
        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(req.size)
            .memory_type_index(mem_type);
        let memory = unsafe { device.logical.allocate_memory(&alloc_info, None) }
            .context("vkAllocateMemory failed (buffer)")?;
        unsafe { device.logical.bind_buffer_memory(handle, memory, 0) }?;

        Ok(Self {
            handle,
            memory,
            size,
            mapped: ptr::null_mut(),
        })
    }

    /// Write `bytes` starting at offset 0. Caller's responsibility:
    /// - the buffer must be host-visible (the only case where mapping is legal),
    /// - `bytes.len()` must be ≤ the allocated size,
    /// - the buffer must NOT already be persistently mapped via
    ///   [`Self::map_persistent`] — pick one strategy per buffer.
    /// HOST_COHERENT means we don't need an explicit flush.
    pub fn write_bytes(&self, device: &Device, bytes: &[u8]) -> Result<()> {
        debug_assert!(
            self.mapped.is_null(),
            "write_bytes called on a persistently-mapped buffer; use write_at_offset instead"
        );
        debug_assert!(
            bytes.len() as vk::DeviceSize <= self.size,
            "write_bytes overflow: {} > {}",
            bytes.len(),
            self.size
        );
        unsafe {
            let dst = device.logical.map_memory(
                self.memory,
                0,
                bytes.len() as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )?;
            ptr::copy_nonoverlapping(bytes.as_ptr(), dst.cast::<u8>(), bytes.len());
            device.logical.unmap_memory(self.memory);
        }
        Ok(())
    }

    /// Map the entire buffer once and keep the pointer for the buffer's
    /// lifetime. Subsequent [`Self::write_at_offset`] calls write through
    /// the stored pointer without a map/unmap cycle each time. The buffer
    /// must be host-visible.
    ///
    /// `destroy` unmaps automatically. Calling this twice is a logic error
    /// and we debug-assert; in release builds the second call simply
    /// overwrites the previous pointer (the spec forbids double-mapping,
    /// so the new vkMapMemory would itself error).
    pub fn map_persistent(&mut self, device: &Device) -> Result<()> {
        debug_assert!(
            self.mapped.is_null(),
            "map_persistent called twice on the same buffer"
        );
        let ptr = unsafe {
            device
                .logical
                .map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty())
                .context("vkMapMemory failed (persistent)")?
        };
        self.mapped = ptr.cast::<u8>();
        Ok(())
    }

    /// Write `bytes` at `offset`. The buffer must have been persistently
    /// mapped via [`Self::map_persistent`]; bytes-plus-offset must fit
    /// within the buffer.
    ///
    /// HOST_COHERENT means writes are visible to the GPU before the next
    /// queue submit — no flush needed. Calling this concurrently from
    /// multiple threads on overlapping ranges is the caller's problem
    /// (the engine itself is single-threaded on the render path).
    pub fn write_at_offset(&self, offset: vk::DeviceSize, bytes: &[u8]) {
        debug_assert!(
            !self.mapped.is_null(),
            "write_at_offset on a non-persistently-mapped buffer; call map_persistent first"
        );
        debug_assert!(
            offset + bytes.len() as vk::DeviceSize <= self.size,
            "write_at_offset overflow: {} + {} > {}",
            offset,
            bytes.len(),
            self.size,
        );
        unsafe {
            let dst = self.mapped.add(offset as usize);
            ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
        }
    }

    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            if !self.mapped.is_null() {
                device.logical.unmap_memory(self.memory);
                self.mapped = ptr::null_mut();
            }
            if self.handle != vk::Buffer::null() {
                device.logical.destroy_buffer(self.handle, None);
            }
            if self.memory != vk::DeviceMemory::null() {
                device.logical.free_memory(self.memory, None);
            }
        }
        self.handle = vk::Buffer::null();
        self.memory = vk::DeviceMemory::null();
    }
}
