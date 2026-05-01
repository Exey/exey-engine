//! Vulkan instance + surface.
//!
//! Mirrors the original `Stage3DManager`: a single owner that hands out the
//! surface handle the rest of the engine binds to. The `Entry`/`Loader` pair
//! corresponds to the AS3 `getInstance(stage)` static.

use anyhow::{Context, Result, anyhow};
use std::collections::HashSet;
use std::ffi::CStr;
use vulkanalia::loader::{LIBRARY, LibloadingLoader};
use vulkanalia::prelude::v1_0::*;
// In vulkanalia 0.31+ the extension trait got split into instance-level and
// device-level halves. We need the instance-level halves of both KHR_surface
// (for destroy_surface_khr, get_physical_device_surface_*_khr) and
// EXT_debug_utils (for create/destroy_debug_utils_messenger_ext).
use vulkanalia::vk::{ExtDebugUtilsExtensionInstanceCommands, KhrSurfaceExtensionInstanceCommands};
use vulkanalia::window as vk_window;
use winit::window::Window;

/// Whether to enable Khronos validation layers. On by default in debug builds.
pub const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

/// The standard Khronos validation layer name.
pub const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

/// Owns the entry point, the `vk::Instance`, the surface and the optional
/// debug-utils messenger. Drop order matters: surface and messenger before
/// instance.
pub struct Instance {
    /// `vulkanalia::Entry`. Holds the dynamically loaded Vulkan library.
    /// Stored even though we don't access it after init — its lifetime
    /// must outlive the instance.
    _entry: Entry,
    pub instance: vulkanalia::Instance,
    pub surface: vk::SurfaceKHR,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

/// The minimum Vulkan SDK version that started requiring the portability
/// subset extension on macOS (MoltenVK). Below this version the portability
/// flags must NOT be passed; at or above this version they are required.
/// We hard-code the threshold the vulkanalia tutorial uses.
const PORTABILITY_MACOS_VERSION: u32 = vk::make_version(1, 3, 216);

impl Instance {
    pub fn new(window: &Window, app_name: &str) -> Result<Self> {
        // Step 1 — load libvulkan dynamically.
        let loader = unsafe { LibloadingLoader::new(LIBRARY) }
            .map_err(|e| anyhow!("failed to load Vulkan loader: {e}"))?;
        let entry = unsafe { Entry::new(loader) }
            .map_err(|e| anyhow!("failed to create Vulkan entry: {e}"))?;

        // Step 2 — collect required instance extensions: WSI extensions for
        // the platform (from vulkanalia's window helper), plus debug-utils
        // when validation is on. The helper takes any &dyn HasDisplayHandle,
        // and `winit::window::Window` implements that.
        let mut extensions = vk_window::get_required_instance_extensions(window)
            .iter()
            .map(|e| e.as_ptr())
            .collect::<Vec<_>>();
        if VALIDATION_ENABLED {
            extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
        }

        // macOS portability: from Vulkan SDK 1.3.216 onward MoltenVK is treated
        // as a non-conformant ("portability") implementation. To use it we
        // must add VK_KHR_portability_enumeration to the instance and pass
        // the ENUMERATE_PORTABILITY_KHR flag. On other platforms this branch
        // is dead code.
        let entry_version = entry.version().unwrap_or(vk::make_version(1, 0, 0));
        let portability_required =
            cfg!(target_os = "macos") && entry_version >= PORTABILITY_MACOS_VERSION;
        if portability_required {
            extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
            // Also required by some MoltenVK builds for device feature queries.
            extensions.push(vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION.name.as_ptr());
        }

        // Step 3 — collect requested layers. Validation layer is opt-in.
        let available_layers = unsafe { entry.enumerate_instance_layer_properties() }?
            .iter()
            .map(|l| l.layer_name)
            .collect::<HashSet<_>>();
        let mut layers: Vec<*const i8> = Vec::new();
        if VALIDATION_ENABLED {
            if !available_layers.contains(&VALIDATION_LAYER) {
                log::warn!(
                    "validation layer requested but not available — install the \
                     Vulkan SDK (LunarG) and try again. Continuing without it."
                );
            } else {
                layers.push(VALIDATION_LAYER.as_ptr());
            }
        }

        // Step 4 — application info. Vulkan 1.3 because we use dynamic rendering.
        let app_name_c = std::ffi::CString::new(app_name)?;
        let engine_name_c = std::ffi::CString::new("ExeyEngine")?;
        let app_info = vk::ApplicationInfo::builder()
            .application_name(app_name_c.as_bytes_with_nul())
            .application_version(vk::make_version(0, 1, 0))
            .engine_name(engine_name_c.as_bytes_with_nul())
            .engine_version(vk::make_version(0, 1, 0))
            .api_version(vk::make_version(1, 3, 0));

        let flags = if portability_required {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::empty()
        };

        let info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layers)
            .flags(flags);

        let instance = unsafe { entry.create_instance(&info, None) }
            .context("vkCreateInstance failed — is the Vulkan loader installed?")?;

        // Step 5 — debug messenger (validation output -> log crate).
        let debug_messenger = if VALIDATION_ENABLED && !layers.is_empty() {
            let info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .user_callback(Some(debug_callback));
            Some(unsafe { instance.create_debug_utils_messenger_ext(&info, None) }?)
        } else {
            None
        };

        // Step 6 — create the surface using the platform-agnostic helper.
        // `winit::window::Window` implements both HasDisplayHandle and
        // HasWindowHandle so we pass it for both args.
        let surface = unsafe { vk_window::create_surface(&instance, window, window) }
            .context("vk_window::create_surface failed")?;

        Ok(Self {
            _entry: entry,
            instance,
            surface,
            debug_messenger,
        })
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_surface_khr(self.surface, None);
            if let Some(m) = self.debug_messenger {
                self.instance.destroy_debug_utils_messenger_ext(m, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

/// Routes Vulkan validation messages to the `log` crate. Severity maps roughly:
/// ERROR -> error!, WARNING -> warn!, INFO -> info!, VERBOSE -> debug!.
extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _kind: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let data = unsafe { &*data };
    let msg = unsafe { CStr::from_ptr(data.message) }
        .to_string_lossy()
        .into_owned();
    match severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::error!("[vk] {msg}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::warn!("[vk] {msg}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::info!("[vk] {msg}"),
        _ => log::debug!("[vk] {msg}"),
    }
    vk::FALSE
}
