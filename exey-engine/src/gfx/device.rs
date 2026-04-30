//! Physical device selection and logical device creation.
//!
//! Splits out what the AS3 `Stage3DProxy` did under the hood: pick a GPU,
//! find queue families that can present to our surface, build a logical
//! device with the features we need (Vulkan 1.3 dynamic rendering +
//! synchronization2).

use anyhow::{Context, Result, anyhow};
use std::collections::HashSet;
use vulkanalia::prelude::v1_0::*;
// vulkanalia 0.31+ split the extension traits. We need the instance-level
// half of KHR_surface for `get_physical_device_surface_support_khr` (used in
// queue-family selection).
use vulkanalia::vk::KhrSurfaceExtensionInstanceCommands;

use super::Instance;

/// Device extensions we always want — the swapchain extension is mandatory
/// because dynamic rendering still presents through `VK_KHR_swapchain`.
const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];

#[derive(Copy, Clone, Debug)]
pub struct QueueFamilyIndices {
    /// Index of a queue family that supports graphics + compute + transfer
    /// AND can present to our surface. We deliberately pick a single family
    /// so the renderer doesn't have to cross-queue ownership-transfer
    /// every frame.
    pub graphics_present: u32,
}

pub struct Device {
    pub physical: vk::PhysicalDevice,
    pub physical_props: vk::PhysicalDeviceProperties,
    pub logical: vulkanalia::Device,
    pub queue: vk::Queue,
    pub queues: QueueFamilyIndices,
}

impl Device {
    pub fn new(instance: &Instance) -> Result<Self> {
        let (physical, queues) = pick_physical_device(instance)?;
        let physical_props = unsafe {
            instance.instance.get_physical_device_properties(physical)
        };
        log_device_info(instance, physical, &physical_props);

        // Single graphics+present queue, priority 1.0.
        let priorities = [1.0f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queues.graphics_present)
            .queue_priorities(&priorities)
            .build()];

        // Required Vulkan 1.3 features for dynamic rendering and the new
        // synchronization API. Both are baseline 1.3 — any 1.3-capable GPU
        // exposes them.
        let mut features_13 = vk::PhysicalDeviceVulkan13Features::builder()
            .dynamic_rendering(true)
            .synchronization2(true);

        let extensions = DEVICE_EXTENSIONS
            .iter()
            .map(|e| e.as_ptr())
            .collect::<Vec<_>>();

        // Enable common 1.0 features we expect most pixel-art workloads to
        // want. `sampler_anisotropy` is needed once we add textures in M2,
        // but enabling it here is free — we just don't use it yet.
        let device_features = vk::PhysicalDeviceFeatures::builder()
            .sampler_anisotropy(true);

        let info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&extensions)
            .enabled_features(&device_features)
            .push_next(&mut features_13);

        let logical = unsafe { instance.instance.create_device(physical, &info, None) }
            .context("vkCreateDevice failed — does this GPU support Vulkan 1.3?")?;

        let queue = unsafe { logical.get_device_queue(queues.graphics_present, 0) };

        Ok(Self {
            physical,
            physical_props,
            logical,
            queue,
            queues,
        })
    }

    pub fn wait_idle(&self) {
        unsafe { self.logical.device_wait_idle() }.ok();
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.logical.destroy_device(None);
        }
    }
}

/// Picks the first physical device that:
///   - supports Vulkan 1.3,
///   - has the swapchain extension,
///   - has a queue family that supports graphics AND can present to our surface.
/// Prefers discrete GPUs over integrated ones if multiple match.
fn pick_physical_device(instance: &Instance) -> Result<(vk::PhysicalDevice, QueueFamilyIndices)> {
    let physicals = unsafe { instance.instance.enumerate_physical_devices() }?;
    if physicals.is_empty() {
        return Err(anyhow!("no Vulkan physical devices found"));
    }

    let mut best: Option<(vk::PhysicalDevice, QueueFamilyIndices, u32)> = None;
    for &p in &physicals {
        let Some(qf) = find_queue_families(instance, p)? else {
            continue;
        };
        if !supports_extensions(instance, p, DEVICE_EXTENSIONS)? {
            continue;
        }
        let props = unsafe { instance.instance.get_physical_device_properties(p) };
        if vk::version_major(props.api_version) < 1
            || (vk::version_major(props.api_version) == 1
                && vk::version_minor(props.api_version) < 3)
        {
            continue;
        }
        let score = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 100,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 10,
            _ => 1,
        };
        if best.map(|(_, _, s)| s < score).unwrap_or(true) {
            best = Some((p, qf, score));
        }
    }
    best.map(|(p, q, _)| (p, q))
        .ok_or_else(|| anyhow!("no suitable GPU found (need Vulkan 1.3 + swapchain + present queue)"))
}

fn find_queue_families(
    instance: &Instance,
    device: vk::PhysicalDevice,
) -> Result<Option<QueueFamilyIndices>> {
    let families = unsafe {
        instance
            .instance
            .get_physical_device_queue_family_properties(device)
    };
    for (idx, fam) in families.iter().enumerate() {
        let idx_u32 = idx as u32;
        if !fam.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            continue;
        }
        let presents = unsafe {
            instance.instance.get_physical_device_surface_support_khr(
                device,
                idx_u32,
                instance.surface,
            )
        }?;
        if presents {
            return Ok(Some(QueueFamilyIndices {
                graphics_present: idx_u32,
            }));
        }
    }
    Ok(None)
}

fn supports_extensions(
    instance: &Instance,
    device: vk::PhysicalDevice,
    required: &[vk::ExtensionName],
) -> Result<bool> {
    let available = unsafe {
        instance
            .instance
            .enumerate_device_extension_properties(device, None)
    }?
    .iter()
    .map(|e| e.extension_name)
    .collect::<HashSet<_>>();
    Ok(required.iter().all(|r| available.contains(r)))
}

fn log_device_info(
    _instance: &Instance,
    _device: vk::PhysicalDevice,
    props: &vk::PhysicalDeviceProperties,
) {
    let name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    log::info!(
        "Picked GPU: {name}  (Vulkan {}.{}.{}, type {:?})",
        vk::version_major(props.api_version),
        vk::version_minor(props.api_version),
        vk::version_patch(props.api_version),
        props.device_type,
    );
}
