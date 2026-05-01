//! [`SpritePipeline`] — the textured-quad graphics pipeline used by every
//! renderer in M2.
//!
//! Owns:
//!   * vertex + fragment shader modules
//!   * descriptor set layout (single combined image sampler at set=0 binding=0)
//!   * pipeline layout (the layout above + a 32-byte push-constant range)
//!   * the [`vk::Pipeline`] itself, configured for dynamic rendering with
//!     the swapchain's color format
//!   * a descriptor pool large enough for one set per loaded texture (M2
//!     pre-allocates [`MAX_TEXTURES`] slots; M3 will switch to per-texture
//!     allocation when the asset manager arrives)
//!
//! Push-constant layout (matches `shaders/sprite.vert`):
//!
//! ```text
//!   offset  size  field
//!   ------  ----  -----
//!     0      8    vec2 scale     // 2.0/extent.{w,h}, screen → NDC scale
//!     8      8    vec2 offset    // (-1, -1), shift origin to top-left
//!    16     16    vec4 tint      // per-draw color modulator (default [1;4])
//! ```
//!
//! Dynamic state covers viewport and scissor so we don't rebuild the pipeline
//! on resize. Color attachment format must match the swapchain at pipeline
//! creation time. Today we read the swapchain format up front (see
//! `pick_format` in `gfx::swapchain`); if a future code path could change
//! the format, we'd need to recreate this pipeline.

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use vulkanalia::prelude::v1_0::*;

use crate::draw::Vertex2D;
use crate::gfx::{Device, Swapchain, Texture};

/// Compiled SPIR-V committed under `exey-engine/shaders/spv/`. See
/// `tools/compile_shaders.sh` for the rebuild step.
const VERT_SPV: &[u8] = include_bytes!("../../shaders/spv/sprite.vert.spv");
const FRAG_SPV: &[u8] = include_bytes!("../../shaders/spv/sprite.frag.spv");

/// Cap on simultaneously loaded textures for M2 / M3. The descriptor pool is
/// sized once at engine startup. Bumping this is cheap (descriptors are tiny).
pub const MAX_TEXTURES: u32 = 64;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SpritePushConstants {
    pub scale: [f32; 2],
    pub offset: [f32; 2],
    pub tint: [f32; 4],
}

// Layout note: GLSL push constants follow std430-ish rules where `vec2` is
// 8-byte-aligned and `vec4` is 16-byte-aligned. With the field order above
// (vec2, vec2, vec4) the natural sequential offsets (0, 8, 16) happen to
// match what GLSL expects, so a `repr(C)` Rust struct serializes correctly
// without manual padding. If you reorder the fields, recheck — e.g. `vec3`
// followed by `f32` packs differently in std430 than in repr(C).

impl SpritePushConstants {
    /// Build the screen → NDC mapping for a given framebuffer extent.
    /// Pixel coords with origin top-left map to clip space [-1, 1] with +Y
    /// down (the natural Vulkan convention; we don't flip).
    pub fn for_extent(extent: vk::Extent2D, tint: [f32; 4]) -> Self {
        let w = extent.width.max(1) as f32;
        let h = extent.height.max(1) as f32;
        Self {
            scale: [2.0 / w, 2.0 / h],
            offset: [-1.0, -1.0],
            tint,
        }
    }
}

pub struct SpritePipeline {
    pub layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    /// The format the pipeline was built against; recreating the swapchain
    /// with a different format would require rebuilding this pipeline.
    pub color_format: vk::Format,
}

impl SpritePipeline {
    pub fn new(device: &Device, swapchain: &Swapchain) -> Result<Self> {
        let descriptor_set_layout = create_descriptor_set_layout(device)?;
        let layout = create_pipeline_layout(device, descriptor_set_layout)?;
        let pipeline = create_pipeline(device, layout, swapchain.format)?;
        let descriptor_pool = create_descriptor_pool(device)?;

        Ok(Self {
            layout,
            pipeline,
            descriptor_set_layout,
            descriptor_pool,
            color_format: swapchain.format,
        })
    }

    /// Allocate a descriptor set bound to one texture. Caller keeps the
    /// returned set as long as the texture lives.
    pub fn allocate_descriptor(
        &self,
        device: &Device,
        texture: &Texture,
    ) -> Result<vk::DescriptorSet> {
        let layouts = [self.descriptor_set_layout];
        let alloc = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        let set = unsafe { device.logical.allocate_descriptor_sets(&alloc) }
            .context("vkAllocateDescriptorSets failed")?[0];

        let image_info = vk::DescriptorImageInfo::builder()
            .sampler(texture.sampler)
            .image_view(texture.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let image_infos = [image_info.build()];
        let write = vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_infos);
        unsafe {
            device.logical.update_descriptor_sets(
                &[write.build()],
                &[] as &[vk::CopyDescriptorSet],
            );
        }
        Ok(set)
    }

    /// Bind pipeline + dynamic state (viewport, scissor) for the current
    /// framebuffer extent. The renderer calls this once before issuing draws.
    pub fn bind(&self, device: &Device, cb: vk::CommandBuffer, extent: vk::Extent2D) {
        unsafe {
            device.logical.cmd_bind_pipeline(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            // Vulkan's default viewport puts the origin top-left and +Y down,
            // which matches our Vertex2D coordinate convention. No Y flip.
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            device.logical.cmd_set_viewport(cb, 0, &[viewport]);

            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            };
            device.logical.cmd_set_scissor(cb, 0, &[scissor]);
        }
    }

    /// Push the screen→clip mapping + tint for the next draw call.
    pub fn push_constants(
        &self,
        device: &Device,
        cb: vk::CommandBuffer,
        pc: &SpritePushConstants,
    ) {
        unsafe {
            device.logical.cmd_push_constants(
                cb,
                self.layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::bytes_of(pc),
            );
        }
    }

    /// Bind a texture's descriptor set at set=0.
    pub fn bind_texture(
        &self,
        device: &Device,
        cb: vk::CommandBuffer,
        descriptor: vk::DescriptorSet,
    ) {
        unsafe {
            device.logical.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                &[descriptor],
                &[],
            );
        }
    }

    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            if self.descriptor_pool != vk::DescriptorPool::null() {
                // Resetting the pool implicitly frees all descriptor sets
                // allocated from it; destroying the pool itself does the
                // same. We don't need to track sets individually.
                device
                    .logical
                    .destroy_descriptor_pool(self.descriptor_pool, None);
            }
            if self.pipeline != vk::Pipeline::null() {
                device.logical.destroy_pipeline(self.pipeline, None);
            }
            if self.layout != vk::PipelineLayout::null() {
                device.logical.destroy_pipeline_layout(self.layout, None);
            }
            if self.descriptor_set_layout != vk::DescriptorSetLayout::null() {
                device
                    .logical
                    .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            }
        }
        self.descriptor_pool = vk::DescriptorPool::null();
        self.pipeline = vk::Pipeline::null();
        self.layout = vk::PipelineLayout::null();
        self.descriptor_set_layout = vk::DescriptorSetLayout::null();
    }
}

fn create_descriptor_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout> {
    let binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);
    let bindings = [binding.build()];
    let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
    Ok(unsafe { device.logical.create_descriptor_set_layout(&info, None) }?)
}

fn create_pipeline_layout(
    device: &Device,
    set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout> {
    let push = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(std::mem::size_of::<SpritePushConstants>() as u32);
    let set_layouts = [set_layout];
    let push_ranges = [push.build()];
    let info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_ranges);
    Ok(unsafe { device.logical.create_pipeline_layout(&info, None) }?)
}

fn create_descriptor_pool(device: &Device) -> Result<vk::DescriptorPool> {
    let pool_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(MAX_TEXTURES);
    let pool_sizes = [pool_size.build()];
    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&pool_sizes)
        .max_sets(MAX_TEXTURES);
    Ok(unsafe { device.logical.create_descriptor_pool(&info, None) }?)
}

fn create_shader_module(device: &Device, code: &[u8]) -> Result<vk::ShaderModule> {
    // SPIR-V is a stream of u32 words; the byte slice from `include_bytes!`
    // is only 1-aligned. vulkanalia's `Bytecode` helper copies into an
    // internal aligned buffer for us and exposes both the u32 slice and the
    // byte size — matching what `vk::ShaderModuleCreateInfo` wants.
    let bytecode = vulkanalia::bytecode::Bytecode::new(code)
        .map_err(|e| anyhow::anyhow!("invalid SPIR-V: {e}"))?;
    let info = vk::ShaderModuleCreateInfo::builder()
        .code_size(bytecode.code_size())
        .code(bytecode.code());
    Ok(unsafe { device.logical.create_shader_module(&info, None) }?)
}

fn create_pipeline(
    device: &Device,
    layout: vk::PipelineLayout,
    color_format: vk::Format,
) -> Result<vk::Pipeline> {
    let vert = create_shader_module(device, VERT_SPV)?;
    let frag = create_shader_module(device, FRAG_SPV)?;

    // We must always destroy these, even on early-return error paths.
    let result = (|| -> Result<vk::Pipeline> {
        let main_name = b"main\0";

        let stages = [
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert)
                .name(main_name)
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag)
                .name(main_name)
                .build(),
        ];

        // Vertex input — one binding, three attributes (pos, color, uv) at
        // shader locations 0, 1, 2 to match `sprite.vert`.
        let vertex_binding = vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(Vertex2D::STRIDE)
            .input_rate(vk::VertexInputRate::VERTEX);
        let vertex_bindings = [vertex_binding.build()];
        let attr_pos = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(memoffset_of_pos());
        let attr_color = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(memoffset_of_color());
        let attr_uv = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(memoffset_of_uv());
        let vertex_attrs = [attr_pos.build(), attr_color.build(), attr_uv.build()];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&vertex_bindings)
            .vertex_attribute_descriptions(&vertex_attrs);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        // Viewport + scissor are dynamic; we still need to declare counts.
        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            // Cull NONE: our quads are 2D and we don't care about winding.
            // Avoids the classic "blank screen, was it CW or CCW" debugging
            // session at the cost of nothing measurable.
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            .line_width(1.0);

        let multisample = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::_1);

        // Standard non-premultiplied alpha blend:
        //   out.rgb = src.rgb * src.a + dst.rgb * (1 - src.a)
        //   out.a   = src.a + dst.a * (1 - src.a)
        // M3 may switch to PMA when we wire `Sprite2D.alpha`.
        let color_write_mask = vk::ColorComponentFlags::R
            | vk::ColorComponentFlags::G
            | vk::ColorComponentFlags::B
            | vk::ColorComponentFlags::A;
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(color_write_mask)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD);
        let attachments = [color_blend_attachment.build()];
        let color_blend = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&dynamic_states);

        // Dynamic rendering: instead of a vk::RenderPass we hand the pipeline
        // a `PipelineRenderingCreateInfo` listing the formats it'll target.
        // This *must* be chained as pNext on the graphics pipeline create info.
        let color_formats = [color_format];
        let mut rendering_info = vk::PipelineRenderingCreateInfo::builder()
            .color_attachment_formats(&color_formats);
        // No depth/stencil for the M2 sprite path.

        let info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic)
            .layout(layout)
            // render_pass = null is required when using dynamic rendering.
            .render_pass(vk::RenderPass::null())
            .subpass(0)
            .push_next(&mut rendering_info);

        let (pipelines, _success) = unsafe {
            device.logical.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[info],
                None,
            )
        }
        .map_err(|(_, e)| anyhow::anyhow!("vkCreateGraphicsPipelines failed: {e}"))?;
        Ok(pipelines[0])
    })();

    unsafe {
        device.logical.destroy_shader_module(frag, None);
        device.logical.destroy_shader_module(vert, None);
    }
    result
}

// We can't use the `memoffset` crate here because it'd be one more dep for
// three offsets. `Vertex2D` is `repr(C)` so the layout is fixed and we just
// hard-code the offsets — keeping them as functions so the symbolism is
// obvious to a reader.
const fn memoffset_of_pos() -> u32 {
    0
}
const fn memoffset_of_color() -> u32 {
    8 // 2 floats
}
const fn memoffset_of_uv() -> u32 {
    24 // 2 + 4 floats
}
