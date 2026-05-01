//! GPU memory allocation helpers.
//!
//! Vulkan splits memory into "memory types", each with a heap and a property
//! mask (host-visible? device-local? coherent? cached?). Allocating any
//! resource boils down to: find a type whose `memory_type_bits` is set in
//! the resource's requirements AND whose properties contain everything we
//! need.
//!
//! M2 allocates one [`vk::DeviceMemory`] per buffer and one per image.
//! That's wasteful in the large (drivers cap allocation count somewhere
//! around 4096), but for an engine with one quad it's fine. M6 will revisit
//! when the BigBuffer renderer wants persistent device-local pools.

use anyhow::{Result, anyhow};
use vulkanalia::prelude::v1_0::*;

use super::{Device, Instance};

/// Pick a memory type matching `type_bits` (from a buffer's or image's
/// memory requirements) AND containing all `required_props`. Returns the
/// type index suitable for `vk::MemoryAllocateInfo::memory_type_index`.
pub fn find_memory_type(
    instance: &Instance,
    device: &Device,
    type_bits: u32,
    required_props: vk::MemoryPropertyFlags,
) -> Result<u32> {
    let props = unsafe {
        instance
            .instance
            .get_physical_device_memory_properties(device.physical)
    };
    for i in 0..props.memory_type_count {
        let bit = 1u32 << i;
        let supported = (type_bits & bit) != 0;
        let has_props = props.memory_types[i as usize]
            .property_flags
            .contains(required_props);
        if supported && has_props {
            return Ok(i);
        }
    }
    Err(anyhow!(
        "no memory type matches bits={type_bits:#x} props={required_props:?}"
    ))
}
