//! Buffer wrapper. A [`Buffer`] owns a `vk::Buffer` plus the `vk::DeviceMemory`
//! backing it. Two factory paths cover M2's needs:
//!
//! - [`Buffer::host_visible`] — host-visible + coherent. Map, write, unmap.
//!   Used for the M2 vertex/index data and for the staging buffer that
//!   uploads texture pixels.
//! - [`Buffer::device_local`] — GPU-local memory. Cannot be mapped from the
//!   host; must be filled via a staging buffer + `vkCmdCopyBuffer`. Not used
//!   by M2's tiny single-quad path but lives here for M6.
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
}

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
        })
    }

    /// Write `bytes` starting at offset 0. Caller's responsibility:
    /// - the buffer must be host-visible (the only case where mapping is legal),
    /// - `bytes.len()` must be ≤ the allocated size.
    /// HOST_COHERENT means we don't need an explicit flush.
    pub fn write_bytes(&self, device: &Device, bytes: &[u8]) -> Result<()> {
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

    pub fn destroy(&mut self, device: &Device) {
        unsafe {
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
