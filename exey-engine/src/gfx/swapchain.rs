//! Swapchain. Owns the swapchain, its images, and per-image views.
//!
//! Resize handling: drop the old swapchain (after device-wait-idle in the
//! caller) and call `recreate`. Doing this idiomatically in Rust meant
//! splitting from the AS3 design — there it was a `configureBackBuffer`
//! call on the proxy.

use anyhow::{Context, Result, anyhow};
use vulkanalia::prelude::v1_0::*;
// Split-trait names for vulkanalia 0.31+. Surface queries are instance-level;
// swapchain create/destroy/get-images are device-level.
use vulkanalia::vk::{KhrSurfaceExtensionInstanceCommands, KhrSwapchainExtensionDeviceCommands};

use super::{Device, Instance};

pub struct Swapchain {
    pub handle: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

impl Swapchain {
    pub fn new(instance: &Instance, device: &Device, window_size: (u32, u32)) -> Result<Self> {
        Self::create(instance, device, window_size, vk::SwapchainKHR::null())
    }

    pub fn recreate(
        &mut self,
        instance: &Instance,
        device: &Device,
        window_size: (u32, u32),
    ) -> Result<()> {
        // Spec note: passing the OLD swapchain to oldSwapchain lets the driver
        // reuse resources and is recommended for resize.
        let old = self.handle;
        let new = Self::create(instance, device, window_size, old)?;
        // Destroy old views and old swapchain only AFTER new one is in.
        self.destroy_views_and_chain(device);
        *self = new;
        Ok(())
    }

    fn create(
        instance: &Instance,
        device: &Device,
        (req_w, req_h): (u32, u32),
        old: vk::SwapchainKHR,
    ) -> Result<Self> {
        let caps = unsafe {
            instance.instance.get_physical_device_surface_capabilities_khr(
                device.physical,
                instance.surface,
            )
        }?;
        let formats = unsafe {
            instance
                .instance
                .get_physical_device_surface_formats_khr(device.physical, instance.surface)
        }?;
        let present_modes = unsafe {
            instance
                .instance
                .get_physical_device_surface_present_modes_khr(device.physical, instance.surface)
        }?;
        if formats.is_empty() || present_modes.is_empty() {
            return Err(anyhow!(
                "surface has no formats or present modes — driver bug?"
            ));
        }

        let format = pick_format(&formats);
        let present_mode = pick_present_mode(&present_modes);
        let extent = pick_extent(&caps, req_w, req_h);

        // image_count: triple-buffer when we can. Spec says clamp into [min, max].
        let mut image_count = caps.min_image_count + 1;
        if caps.max_image_count > 0 && image_count > caps.max_image_count {
            image_count = caps.max_image_count;
        }

        let info = vk::SwapchainCreateInfoKHR::builder()
            .surface(instance.surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old);

        let handle = unsafe { device.logical.create_swapchain_khr(&info, None) }
            .context("vkCreateSwapchainKHR failed")?;
        let images = unsafe { device.logical.get_swapchain_images_khr(handle) }?;
        let views = images
            .iter()
            .map(|&img| create_view(device, img, format.format))
            .collect::<Result<Vec<_>>>()?;

        log::info!(
            "swapchain: {}x{}, {} images, format {:?}, present_mode {:?}",
            extent.width,
            extent.height,
            images.len(),
            format.format,
            present_mode
        );

        Ok(Self {
            handle,
            images,
            views,
            format: format.format,
            extent,
        })
    }

    fn destroy_views_and_chain(&self, device: &Device) {
        unsafe {
            for &v in &self.views {
                device.logical.destroy_image_view(v, None);
            }
            device.logical.destroy_swapchain_khr(self.handle, None);
        }
    }

    pub fn destroy(&mut self, device: &Device) {
        self.destroy_views_and_chain(device);
        self.handle = vk::SwapchainKHR::null();
        self.views.clear();
        self.images.clear();
    }
}

fn pick_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    // Prefer 8-bit BGRA / sRGB. Fall back to whatever's first if the driver
    // doesn't advertise our preference.
    formats
        .iter()
        .copied()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or(formats[0])
}

fn pick_present_mode(modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    // MAILBOX is the gamer's choice — non-blocking, low-latency, no tearing.
    // FIFO is guaranteed by spec, used as the safe fallback (== vsync).
    if modes.contains(&vk::PresentModeKHR::MAILBOX) {
        vk::PresentModeKHR::MAILBOX
    } else {
        vk::PresentModeKHR::FIFO
    }
}

fn pick_extent(caps: &vk::SurfaceCapabilitiesKHR, req_w: u32, req_h: u32) -> vk::Extent2D {
    // u32::MAX is the spec's "use the window size" sentinel.
    if caps.current_extent.width != u32::MAX {
        return caps.current_extent;
    }
    vk::Extent2D {
        width: req_w.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
        height: req_h.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
    }
}

fn create_view(device: &Device, image: vk::Image, format: vk::Format) -> Result<vk::ImageView> {
    let info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::_2D)
        .format(format)
        .components(vk::ComponentMapping::default())
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    Ok(unsafe { device.logical.create_image_view(&info, None) }?)
}
