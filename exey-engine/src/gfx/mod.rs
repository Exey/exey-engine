//! Vulkan layer. Wraps `vulkanalia` into engine-shaped abstractions.
//!
//! Mirrors the original `stage3d/` package: `Stage3DManager` and
//! `Stage3DProxy` are split into [`instance`], [`device`], and [`swapchain`].
//! The original `RenderUtil` (vertex/index buffer scratchpads) becomes
//! [`buffer`] in later milestones.

pub mod device;
pub mod frame;
pub mod instance;
pub mod swapchain;

pub use device::Device;
pub use frame::{FrameContext, FramesInFlight};
pub use instance::Instance;
pub use swapchain::Swapchain;
