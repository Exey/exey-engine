//! Texture: a 2D `vk::Image` plus its memory, view, and a default sampler.
//!
//! The original AS3 engine treated textures as named entries in a manager
//! (`textureName:String` keys lookup); the Rust port keeps the texture
//! itself as a self-contained owning struct and lets a higher layer (M3+)
//! own the name→`Arc<Texture>` map.
//!
//! Upload path mirrors what `Stage3D` did for us under the hood:
//!   1) RGBA8 bytes → host-visible staging buffer
//!   2) device-local `R8G8B8A8_SRGB` image
//!   3) UNDEFINED → TRANSFER_DST_OPTIMAL barrier
//!   4) `vkCmdCopyBufferToImage`
//!   5) TRANSFER_DST_OPTIMAL → SHADER_READ_ONLY_OPTIMAL barrier
//!
//! sRGB is deliberate: the swapchain is `B8G8R8A8_SRGB`, so storing pixel-art
//! source bytes as sRGB and sampling them as sRGB makes the round trip
//! identity. Treating the texture as linear UNORM would gamma-shift sprites.
//!
//! The sampler uses NEAREST filtering — sprite-art correctness, matches
//! the AS3 engine's `MIPNEAREST` / `NEAREST` choices.

use anyhow::{Context, Result};
use vulkanalia::prelude::v1_0::*;

use super::{Device, Instance, buffer::Buffer, memory};

pub struct Texture {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
    pub width: u32,
    pub height: u32,
}

impl Texture {
    /// Create a 2D texture from a tightly-packed RGBA8 byte slice
    /// (`width * height * 4` bytes, row-major, top-left origin).
    pub fn from_rgba(
        instance: &Instance,
        device: &Device,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> Result<Self> {
        let expected = (width as usize) * (height as usize) * 4;
        anyhow::ensure!(
            rgba.len() == expected,
            "Texture::from_rgba: got {} bytes, expected {expected} ({width}x{height} RGBA8)",
            rgba.len()
        );
        log::info!(
            "Texture::from_rgba — uploading {width}x{height} RGBA ({} bytes) as R8G8B8A8_SRGB",
            rgba.len()
        );

        // 1) staging buffer.
        let mut staging = Buffer::host_visible(
            instance,
            device,
            rgba.len() as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_SRC,
        )?;
        staging.write_bytes(device, rgba)?;

        // 2) device-local image.
        let format = vk::Format::R8G8B8A8_SRGB;
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
        let img_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::_2D)
            .format(format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.logical.create_image(&img_info, None) }
            .context("vkCreateImage failed")?;

        let req = unsafe { device.logical.get_image_memory_requirements(image) };
        let mem_type = memory::find_memory_type(
            instance,
            device,
            req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(req.size)
            .memory_type_index(mem_type);
        let memory_handle = unsafe { device.logical.allocate_memory(&alloc_info, None) }
            .context("vkAllocateMemory failed (image)")?;
        unsafe { device.logical.bind_image_memory(image, memory_handle, 0) }?;

        // 3-5) one-shot command buffer to do the layout transitions + copy.
        upload_pixels(device, &staging, image, width, height)
            .context("texture pixel upload failed")?;

        staging.destroy(device);

        // 6) view + sampler.
        let view = create_view(device, image, format)?;
        let sampler = create_sampler(device)?;

        log::info!(
            "Texture::from_rgba — done.  image={:?}  view={:?}  sampler={:?}",
            image, view, sampler,
        );

        Ok(Self {
            image,
            memory: memory_handle,
            view,
            sampler,
            width,
            height,
        })
    }

    /// Decode a PNG byte stream and upload. Convenience wrapper for assets
    /// loaded from disk; M2's demo uses [`Self::from_rgba`] with a
    /// procedural checkerboard, so this is here for M3+.
    pub fn from_png_bytes(
        instance: &Instance,
        device: &Device,
        bytes: &[u8],
    ) -> Result<Self> {
        let img = image::load_from_memory(bytes)
            .context("failed to decode PNG")?
            .to_rgba8();
        let (w, h) = (img.width(), img.height());
        Self::from_rgba(instance, device, w, h, img.as_raw())
    }

    /// Load an image file (PNG, JPEG, …) from disk, promote near-black
    /// pixels to fully transparent, and upload as RGBA.
    ///
    /// Why the alpha key: the wolf spritesheet ships as a JPEG with a
    /// black background and no alpha channel. The sprite shader does
    /// alpha blending, so we need actual transparency for the wolves
    /// to read as cut-out characters on the tiles. A hard
    /// `rgb == (0,0,0) → alpha 0` test would leave a 1–2px grey halo
    /// from JPEG compression around every silhouette; a luma threshold
    /// with a small ramp band hides that halo.
    ///
    /// Thresholds (in 0..255 luma space, ITU-R BT.601):
    /// * `luma <= LUMA_KEY_LOW`  → alpha 0   (fully transparent)
    /// * `luma >= LUMA_KEY_HIGH` → alpha 255 (fully opaque)
    /// * in between → linearly ramped
    ///
    /// For pure-PNG inputs with an existing alpha channel, this path
    /// is wrong (it would re-key already-transparent pixels and corrupt
    /// the alpha). Use [`Self::from_png_bytes`] for those. We pick the
    /// loader based on the *call site* in the demo — the user picks the
    /// asset and the demo picks the loader.
    pub fn from_image_file_with_luma_key(
        instance: &Instance,
        device: &Device,
        path: &std::path::Path,
    ) -> Result<Self> {
        const LUMA_KEY_LOW: u32 = 8;
        const LUMA_KEY_HIGH: u32 = 16;

        log::info!("Texture::from_image_file_with_luma_key — opening {}", path.display());
        let img = image::open(path)
            .with_context(|| format!("failed to open image file: {}", path.display()))?
            .to_rgba8();
        let (w, h) = (img.width(), img.height());
        let mut rgba = img.into_raw();

        // Re-key alpha based on luma. Iterate as chunks of 4 bytes (RGBA).
        // The existing alpha (255 from `to_rgba8` on a JPEG) is ignored;
        // for PNGs with real alpha, prefer `from_png_bytes`.
        let mut transparent = 0usize;
        let mut ramp = 0usize;
        let band = (LUMA_KEY_HIGH - LUMA_KEY_LOW).max(1);
        for px in rgba.chunks_exact_mut(4) {
            // BT.601 luma ≈ 0.299 R + 0.587 G + 0.114 B. Integer
            // approximation with a /1000 divisor — accurate to ~0.1
            // of a unit, far below our threshold of 8.
            let luma = (299 * px[0] as u32 + 587 * px[1] as u32 + 114 * px[2] as u32) / 1000;
            if luma <= LUMA_KEY_LOW {
                px[3] = 0;
                transparent += 1;
            } else if luma >= LUMA_KEY_HIGH {
                px[3] = 255;
            } else {
                // Ramp from 0 at LOW to 255 at HIGH.
                let a = ((luma - LUMA_KEY_LOW) * 255) / band;
                px[3] = a.min(255) as u8;
                ramp += 1;
            }
        }
        let total = (w as usize) * (h as usize);
        log::info!(
            "Texture::from_image_file_with_luma_key — {w}x{h}  {} fully transparent  {} in ramp  {} opaque  (luma keys {LUMA_KEY_LOW}..{LUMA_KEY_HIGH})",
            transparent, ramp, total - transparent - ramp,
        );

        Self::from_rgba(instance, device, w, h, &rgba)
    }

    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            if self.sampler != vk::Sampler::null() {
                device.logical.destroy_sampler(self.sampler, None);
            }
            if self.view != vk::ImageView::null() {
                device.logical.destroy_image_view(self.view, None);
            }
            if self.image != vk::Image::null() {
                device.logical.destroy_image(self.image, None);
            }
            if self.memory != vk::DeviceMemory::null() {
                device.logical.free_memory(self.memory, None);
            }
        }
        self.sampler = vk::Sampler::null();
        self.view = vk::ImageView::null();
        self.image = vk::Image::null();
        self.memory = vk::DeviceMemory::null();
    }
}

/// Issues a one-shot command buffer that does:
///   barrier(UNDEFINED -> TRANSFER_DST) → copy_buffer_to_image → barrier(TRANSFER_DST -> SHADER_READ_ONLY)
/// then waits idle. Synchronous and slow — only used for asset upload, never
/// per-frame. M3+ may queue uploads via a transfer queue, but for now we
/// ride the graphics queue and block.
fn upload_pixels(
    device: &Device,
    staging: &Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Result<()> {
    // Allocate a one-shot command buffer. We use a fresh transient pool so
    // we don't have to coordinate with the per-frame pools.
    let pool_info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
        .queue_family_index(device.queues.graphics_present);
    let pool = unsafe { device.logical.create_command_pool(&pool_info, None) }?;

    let result = (|| -> Result<()> {
        let alloc = vk::CommandBufferAllocateInfo::builder()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cb = unsafe { device.logical.allocate_command_buffers(&alloc) }?[0];

        let begin = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.logical.begin_command_buffer(cb, &begin) }?;

        // UNDEFINED -> TRANSFER_DST_OPTIMAL
        image_barrier(
            device,
            cb,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        );

        let region = vk::BufferImageCopy::builder()
            .buffer_offset(0)
            .buffer_row_length(0)   // 0 = tightly packed
            .buffer_image_height(0) // 0 = tightly packed
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });
        unsafe {
            device.logical.cmd_copy_buffer_to_image(
                cb,
                staging.handle,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }

        // TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
        image_barrier(
            device,
            cb,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        );

        unsafe { device.logical.end_command_buffer(cb) }?;

        // Submit and wait — texture upload is one-shot at engine startup,
        // we don't need to overlap it with frame work.
        let cbs = [cb];
        let submit = vk::SubmitInfo::builder().command_buffers(&cbs);
        unsafe {
            device.logical.queue_submit(device.queue, &[submit], vk::Fence::null())?;
            device.logical.queue_wait_idle(device.queue)?;
        }
        Ok(())
    })();

    unsafe { device.logical.destroy_command_pool(pool, None) };
    result
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

fn create_sampler(device: &Device) -> Result<vk::Sampler> {
    // NEAREST filtering — pixel art, no smoothing. clamp_to_edge so seams on
    // a tile sheet don't bleed. Anisotropy off; we don't have mips.
    let info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::NEAREST)
        .min_filter(vk::Filter::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .anisotropy_enable(false)
        .max_anisotropy(1.0)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .mip_lod_bias(0.0)
        .min_lod(0.0)
        .max_lod(0.0);
    Ok(unsafe { device.logical.create_sampler(&info, None) }?)
}

#[allow(clippy::too_many_arguments)]
fn image_barrier(
    device: &Device,
    cb: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
) {
    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(src_access)
        .dst_access_mask(dst_access);
    unsafe {
        device.logical.cmd_pipeline_barrier(
            cb,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );
    }
}
