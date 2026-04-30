//! Per-frame resources — command pool / buffers and the sync primitives.
//!
//! With dynamic rendering (Vulkan 1.3) there is no render pass to handle
//! the swapchain image's layout transitions for us, so we issue the two
//! barriers explicitly:
//!
//!   UNDEFINED      → COLOR_ATTACHMENT_OPTIMAL  (before begin_rendering)
//!   COLOR_ATTACH.. → PRESENT_SRC_KHR           (after end_rendering, before queue_present)

use anyhow::{Context, Result};
use vulkanalia::prelude::v1_0::*;
// Dynamic rendering (cmd_begin_rendering / cmd_end_rendering) is core in
// Vulkan 1.3, so the methods live on the v1.3 device trait — NOT on a KHR
// extension trait. Swapchain ops (acquire_next_image_khr, queue_present_khr,
// etc.) are device-level extension methods.
use vulkanalia::vk::{DeviceV1_3, KhrSwapchainExtensionDeviceCommands};

use super::{Device, Swapchain};

/// How many frames we let the CPU stay ahead of the GPU. Two is the standard
/// "double-buffered" tradeoff between CPU/GPU overlap and memory.
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

/// Per in-flight-frame: one command buffer + the three sync primitives.
pub struct FrameContext {
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    /// Signalled when the swapchain image is ready to be drawn into.
    pub image_available: vk::Semaphore,
    /// Signalled when our rendering work is done — the present op waits on it.
    pub render_finished: vk::Semaphore,
    /// CPU-side fence: signalled when this frame's GPU work has finished, so
    /// we can safely reset and re-record its command buffer.
    pub in_flight: vk::Fence,
}

pub struct FramesInFlight {
    pub frames: Vec<FrameContext>,
    pub current: usize,
}

impl FramesInFlight {
    pub fn new(device: &Device) -> Result<Self> {
        let mut frames = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            frames.push(FrameContext::new(device)?);
        }
        Ok(Self { frames, current: 0 })
    }

    pub fn destroy(&mut self, device: &Device) {
        for f in self.frames.drain(..) {
            f.destroy(device);
        }
    }

    pub fn advance(&mut self) {
        self.current = (self.current + 1) % self.frames.len();
    }

    pub fn current(&self) -> &FrameContext {
        &self.frames[self.current]
    }
}

impl FrameContext {
    pub fn new(device: &Device) -> Result<Self> {
        let pool_info = vk::CommandPoolCreateInfo::builder()
            // RESET_COMMAND_BUFFER lets us call vkResetCommandBuffer per frame
            // without resetting the whole pool.
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(device.queues.graphics_present);
        let command_pool =
            unsafe { device.logical.create_command_pool(&pool_info, None) }?;

        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer =
            unsafe { device.logical.allocate_command_buffers(&alloc_info) }?[0];

        let sem_info = vk::SemaphoreCreateInfo::builder();
        let image_available =
            unsafe { device.logical.create_semaphore(&sem_info, None) }?;
        let render_finished =
            unsafe { device.logical.create_semaphore(&sem_info, None) }?;

        // Fence starts SIGNALLED so the very first frame doesn't block on it.
        let fence_info =
            vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        let in_flight = unsafe { device.logical.create_fence(&fence_info, None) }?;

        Ok(Self {
            command_pool,
            command_buffer,
            image_available,
            render_finished,
            in_flight,
        })
    }

    pub fn destroy(self, device: &Device) {
        unsafe {
            device.logical.destroy_fence(self.in_flight, None);
            device.logical.destroy_semaphore(self.render_finished, None);
            device.logical.destroy_semaphore(self.image_available, None);
            device.logical.destroy_command_pool(self.command_pool, None);
        }
    }
}

/// Result of `acquire`. `Suboptimal` and `OutOfDate` mean the caller must
/// recreate the swapchain before the next frame.
pub enum AcquireResult {
    Ok(u32),
    Recreate,
}

/// Acquire the next swapchain image, waiting on this frame's fence first.
pub fn acquire(
    device: &Device,
    swapchain: &Swapchain,
    frame: &FrameContext,
) -> Result<AcquireResult> {
    unsafe {
        device
            .logical
            .wait_for_fences(&[frame.in_flight], true, u64::MAX)?;
    }
    let acquired = unsafe {
        device.logical.acquire_next_image_khr(
            swapchain.handle,
            u64::MAX,
            frame.image_available,
            vk::Fence::null(),
        )
    };
    match acquired {
        Ok((index, vk::SuccessCode::SUBOPTIMAL_KHR)) => {
            let _ = index;
            Ok(AcquireResult::Recreate)
        }
        Ok((index, _)) => {
            // Reset only AFTER successful acquire — otherwise we'd starve on
            // a swapchain that needs recreation.
            unsafe { device.logical.reset_fences(&[frame.in_flight])? };
            Ok(AcquireResult::Ok(index))
        }
        Err(vk::ErrorCode::OUT_OF_DATE_KHR) => Ok(AcquireResult::Recreate),
        Err(e) => Err(e).context("vkAcquireNextImageKHR failed"),
    }
}

/// Records: barrier → begin_rendering → (optional draw) → end_rendering → barrier.
/// In M1 there is no draw; this is the canvas the future renderers paint on.
pub fn record_clear(
    device: &Device,
    swapchain: &Swapchain,
    frame: &FrameContext,
    image_index: u32,
    clear_color: [f32; 4],
) -> Result<()> {
    let cb = frame.command_buffer;
    unsafe {
        device.logical.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
    }

    let begin = vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { device.logical.begin_command_buffer(cb, &begin) }?;

    let image = swapchain.images[image_index as usize];
    let view = swapchain.views[image_index as usize];

    // 1) UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
    image_barrier(
        device,
        cb,
        image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
    );

    // 2) Begin dynamic rendering
    let clear = vk::ClearValue {
        color: vk::ClearColorValue { float32: clear_color },
    };
    let color_attachment = vk::RenderingAttachmentInfo::builder()
        .image_view(view)
        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .clear_value(clear);

    let color_attachments = [color_attachment.build()];
    let render_info = vk::RenderingInfo::builder()
        .render_area(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain.extent,
        })
        .layer_count(1)
        .color_attachments(&color_attachments);

    unsafe {
        device.logical.cmd_begin_rendering(cb, &render_info);
        // M2+: bind pipeline, vertex/index buffers, descriptors, draw.
        device.logical.cmd_end_rendering(cb);
    }

    // 3) COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR
    image_barrier(
        device,
        cb,
        image,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        vk::ImageLayout::PRESENT_SRC_KHR,
        vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        vk::AccessFlags::empty(),
        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
    );

    unsafe { device.logical.end_command_buffer(cb) }?;
    Ok(())
}

/// Submits this frame's command buffer and queues a present.
pub fn submit_and_present(
    device: &Device,
    swapchain: &Swapchain,
    frame: &FrameContext,
    image_index: u32,
) -> Result<bool /* needs_recreate */> {
    let wait_semaphores = [frame.image_available];
    let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
    let cmd_buffers = [frame.command_buffer];
    let signal_semaphores = [frame.render_finished];

    let submit = vk::SubmitInfo::builder()
        .wait_semaphores(&wait_semaphores)
        .wait_dst_stage_mask(&wait_stages)
        .command_buffers(&cmd_buffers)
        .signal_semaphores(&signal_semaphores);

    unsafe {
        device
            .logical
            .queue_submit(device.queue, &[submit], frame.in_flight)?;
    }

    let swapchains = [swapchain.handle];
    let image_indices = [image_index];
    let present = vk::PresentInfoKHR::builder()
        .wait_semaphores(&signal_semaphores)
        .swapchains(&swapchains)
        .image_indices(&image_indices);

    let result = unsafe { device.logical.queue_present_khr(device.queue, &present) };
    match result {
        Ok(vk::SuccessCode::SUBOPTIMAL_KHR) => Ok(true),
        Ok(_) => Ok(false),
        Err(vk::ErrorCode::OUT_OF_DATE_KHR) => Ok(true),
        Err(e) => Err(e).context("vkQueuePresentKHR failed"),
    }
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
