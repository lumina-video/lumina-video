use super::*;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_video::{
    kCVPixelBufferIOSurfacePropertiesKey, kCVPixelBufferMetalCompatibilityKey,
    kCVPixelBufferPixelFormatTypeKey, kCVPixelFormatType_32BGRA,
};
use objc2_foundation::{NSCopying, NSMutableDictionary, NSNumber, NSString};
use parking_lot::Mutex as ParkingMutex;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};

// ==========================================================================
// Raw FFI declarations for VideoToolbox and CoreMedia
//
// Using raw FFI because objc2 0.6 generated bindings changed from raw pointers
// to Option<&T>/NonNull which requires significant code restructuring.
// Raw FFI is more stable across objc2 versions.
// ==========================================================================

/// CMTime structure (matches CoreMedia layout)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CMTime {
    pub value: i64,
    pub timescale: i32,
    pub flags: u32,
    pub epoch: i64,
}

impl CMTime {
    pub const fn invalid() -> Self {
        Self {
            value: 0,
            timescale: 0,
            flags: 0,
            epoch: 0,
        }
    }

    pub const fn new(value: i64, timescale: i32) -> Self {
        Self {
            value,
            timescale,
            flags: 1,
            epoch: 0,
        } // flags=1 is kCMTimeFlags_Valid
    }
}

/// CMSampleTimingInfo structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CMSampleTimingInfo {
    pub duration: CMTime,
    pub presentation_time_stamp: CMTime,
    pub decode_time_stamp: CMTime,
}

/// VTDecompressionOutputCallback function pointer type.
/// Field names match Apple's C API naming convention.
#[allow(non_snake_case)]
pub type VTDecompressionOutputCallback = extern "C" fn(
    decompressionOutputRefCon: *mut c_void,
    sourceFrameRefCon: *mut c_void,
    status: i32,
    infoFlags: u32,
    imageBuffer: *mut c_void,
    presentationTimeStamp: CMTime,
    presentationDuration: CMTime,
);

/// VTDecompressionOutputCallbackRecord structure.
/// Field names match Apple's C API naming convention.
#[repr(C)]
#[allow(non_snake_case)]
pub struct VTDecompressionOutputCallbackRecord {
    pub decompressionOutputCallback: Option<VTDecompressionOutputCallback>,
    pub decompressionOutputRefCon: *mut c_void,
}

// Raw FFI declarations - split by framework for proper linking

// CoreVideo framework
#[link(name = "CoreVideo", kind = "framework")]
extern "C" {
    fn CVPixelBufferGetIOSurface(pixelBuffer: *const c_void) -> *mut c_void;
    fn CVPixelBufferGetWidth(pixelBuffer: *const c_void) -> usize;
    fn CVPixelBufferGetHeight(pixelBuffer: *const c_void) -> usize;
}

// CoreMedia framework
#[link(name = "CoreMedia", kind = "framework")]
extern "C" {
    fn CMVideoFormatDescriptionCreateFromH264ParameterSets(
        allocator: *const c_void,
        parameterSetCount: usize,
        parameterSetPointers: *const *const u8,
        parameterSetSizes: *const usize,
        NALUnitHeaderLength: i32,
        formatDescriptionOut: *mut *mut c_void,
    ) -> i32;

    // Reserved for H.265 support
    #[allow(dead_code)]
    fn CMVideoFormatDescriptionCreateFromHEVCParameterSets(
        allocator: *const c_void,
        parameterSetCount: usize,
        parameterSetPointers: *const *const u8,
        parameterSetSizes: *const usize,
        NALUnitHeaderLength: i32,
        extensions: *const c_void,
        formatDescriptionOut: *mut *mut c_void,
    ) -> i32;

    fn CMBlockBufferCreateWithMemoryBlock(
        structureAllocator: *const c_void,
        memoryBlock: *mut c_void,
        blockLength: usize,
        blockAllocator: *const c_void,
        customBlockSource: *const c_void,
        offsetToData: usize,
        dataLength: usize,
        flags: u32,
        blockBufferOut: *mut *mut c_void,
    ) -> i32;

    fn CMSampleBufferCreate(
        allocator: *const c_void,
        dataBuffer: *mut c_void,
        dataReady: bool,
        makeDataReadyCallback: *const c_void,
        makeDataReadyRefcon: *mut c_void,
        formatDescription: *mut c_void,
        numSamples: i64,
        numSampleTimingEntries: i64,
        sampleTimingArray: *const CMSampleTimingInfo,
        numSampleSizeEntries: i64,
        sampleSizeArray: *const usize,
        sampleBufferOut: *mut *mut c_void,
    ) -> i32;

    fn CMSampleBufferGetSampleAttachmentsArray(
        sbuf: *mut c_void,
        createIfNecessary: bool,
    ) -> *const c_void;

    /// kCMSampleAttachmentKey_DependsOnOthers: kCFBooleanFalse = sync sample (keyframe)
    static kCMSampleAttachmentKey_DependsOnOthers: *const c_void;
    static kCMSampleAttachmentKey_NotSync: *const c_void;
}

// VideoToolbox framework
#[link(name = "VideoToolbox", kind = "framework")]
extern "C" {
    fn VTDecompressionSessionCreate(
        allocator: *const c_void,
        videoFormatDescription: *mut c_void,
        videoDecoderSpecification: *const c_void,
        destinationImageBufferAttributes: *const c_void,
        outputCallback: *const VTDecompressionOutputCallbackRecord,
        decompressionSessionOut: *mut *mut c_void,
    ) -> i32;

    fn VTDecompressionSessionDecodeFrame(
        session: *mut c_void,
        sampleBuffer: *mut c_void,
        decodeFlags: u32,
        sourceFrameRefCon: *mut c_void,
        infoFlagsOut: *mut u32,
    ) -> i32;

    fn VTDecompressionSessionWaitForAsynchronousFrames(session: *mut c_void) -> i32;

    fn VTDecompressionSessionInvalidate(session: *mut c_void);
}

// CoreFoundation framework
#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFRelease(cf: *const c_void);
    fn CFRetain(cf: *const c_void) -> *const c_void;
    fn CFArrayGetValueAtIndex(theArray: *const c_void, idx: isize) -> *const c_void;
    fn CFDictionarySetValue(theDict: *const c_void, key: *const c_void, value: *const c_void);
    /// Null allocator - performs no allocation/deallocation.
    /// Use this for caller-owned memory passed to CM functions.
    static kCFAllocatorNull: *const c_void;
    static kCFBooleanTrue: *const c_void;
    static kCFBooleanFalse: *const c_void;
}

/// Wrapper for CVPixelBuffer (raw pointer) that releases on drop.
struct PixelBufferWrapper(*mut c_void);

impl PixelBufferWrapper {
    /// Retains and wraps a CVPixelBuffer pointer.
    unsafe fn retain(ptr: *mut c_void) -> Self {
        if !ptr.is_null() {
            CFRetain(ptr);
        }
        Self(ptr)
    }
}

impl Drop for PixelBufferWrapper {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { CFRelease(self.0) };
        }
    }
}

impl std::fmt::Debug for PixelBufferWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PixelBufferWrapper")
            .field("ptr", &self.0)
            .finish()
    }
}

// SAFETY: CVPixelBuffer is safe to send between threads because:
// - The pixel data is immutable after creation
// - CoreFoundation reference counting is thread-safe
// - The IOSurface backing (if any) is also thread-safe
unsafe impl Send for PixelBufferWrapper {}
unsafe impl Sync for PixelBufferWrapper {}

/// A decoded frame from VTDecompressionSession, ready for rendering.
struct DecodedVTFrame {
    /// Presentation timestamp in microseconds
    pts_us: u64,
    /// The decoded CVPixelBuffer (retained)
    pixel_buffer: PixelBufferWrapper,
    /// True when callback had kVTDecodeInfo_RequiredFrameDropped.
    required_frame_dropped: bool,
}

/// Shared state for decoder callback to push decoded frames.
struct VTCallbackState {
    /// Queue of decoded frames (protected by mutex)
    decoded_frames: ParkingMutex<VecDeque<DecodedVTFrame>>,
    /// Error flag set by callback on decode failure
    decode_error: AtomicBool,
    /// OSStatus from last callback error (0 = no error)
    decode_error_status: AtomicI32,
    /// Frame counter for debugging
    frame_count: AtomicU32,
}

impl VTCallbackState {
    fn new() -> Self {
        Self {
            decoded_frames: ParkingMutex::new(VecDeque::with_capacity(8)),
            decode_error: AtomicBool::new(false),
            decode_error_status: AtomicI32::new(0),
            frame_count: AtomicU32::new(0),
        }
    }
}

/// VTDecompressionSession wrapper for zero-copy H.264/H.265 decoding.
///
/// Uses raw FFI pointers for VideoToolbox interop. The session and format_desc
/// are retained on creation and released on drop.
pub struct VTDecoder {
    /// The VideoToolbox decompression session (retained)
    session: *mut c_void,
    /// CMFormatDescription for the video stream (retained)
    format_desc: *mut c_void,
    /// Shared callback state (Arc for callback lifetime)
    callback_state: Arc<VTCallbackState>,
    /// Video dimensions (reserved for future use in frame validation)
    #[allow(dead_code)]
    pub width: u32,
    /// Video dimensions (reserved for future use in frame validation)
    #[allow(dead_code)]
    pub height: u32,
    /// Codec type (H.264 or H.265, reserved for H.265 support)
    #[allow(dead_code)]
    codec: VTCodec,
    /// True if NAL data is AVCC format (length-prefixed), false for Annex B (start codes).
    /// Set from catalog info to avoid heuristic detection that misclassifies
    /// AVCC frames with 256-511 byte NALs (length prefix [0,0,1,X] looks like Annex B).
    is_avcc: bool,
    /// Consecutive callbacks flagged with RequiredFrameDropped.
    required_drop_streak: u32,
}

/// If this many consecutive decoded callbacks require frame drops, treat as
/// a corruption storm and force session recovery.
const VT_REQUIRED_DROP_STORM_THRESHOLD: u32 = 36;

// SAFETY: VTDecompressionSession is designed for multi-threaded use.
// The session pointer is only accessed through synchronized methods.
// The callback_state uses interior mutability with proper synchronization.
unsafe impl Send for VTDecoder {}
unsafe impl Sync for VTDecoder {}

/// Supported codecs for VTDecompressionSession.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum VTCodec {
    /// H.264/AVC codec
    H264,
    /// H.265/HEVC codec (reserved for future support)
    H265,
}

impl VTDecoder {
    /// Creates a new VTDecoder for H.264 with the given SPS/PPS NAL units.
    ///
    /// # Arguments
    /// * `sps` - Sequence Parameter Set NAL unit (without start code)
    /// * `pps` - Picture Parameter Set NAL unit (without start code)
    /// * `width` - Video width (hint, may be overridden by SPS)
    /// * `height` - Video height (hint, may be overridden by SPS)
    pub fn new_h264(
        sps: &[u8],
        pps: &[u8],
        width: u32,
        height: u32,
        is_avcc: bool,
    ) -> Result<Self, VideoError> {
        tracing::info!(
            "VTDecoder: Creating H.264 decoder {}x{} (SPS: {} bytes, PPS: {} bytes, avcc={})",
            width,
            height,
            sps.len(),
            pps.len(),
            is_avcc,
        );

        // Create CMFormatDescription from SPS/PPS
        let format_desc = Self::create_h264_format_description(sps, pps)?;

        // Create decoder with format description
        Self::create_decoder(format_desc, width, height, VTCodec::H264, is_avcc)
    }

    /// Creates a new VTDecoder for H.265 with the given VPS/SPS/PPS NAL units.
    ///
    /// # Arguments
    /// * `vps` - Video Parameter Set NAL unit (without start code)
    /// * `sps` - Sequence Parameter Set NAL unit (without start code)
    /// * `pps` - Picture Parameter Set NAL unit (without start code)
    /// * `width` - Video width
    /// * `height` - Video height
    #[allow(dead_code)]
    pub fn new_h265(
        vps: &[u8],
        sps: &[u8],
        pps: &[u8],
        width: u32,
        height: u32,
        is_avcc: bool,
    ) -> Result<Self, VideoError> {
        tracing::info!(
            "VTDecoder: Creating H.265 decoder {}x{} (VPS: {} bytes, SPS: {} bytes, PPS: {} bytes, avcc={})",
            width,
            height,
            vps.len(),
            sps.len(),
            pps.len(),
            is_avcc,
        );

        // Create CMFormatDescription from VPS/SPS/PPS
        let format_desc = Self::create_h265_format_description(vps, sps, pps)?;

        // Create decoder with format description
        Self::create_decoder(format_desc, width, height, VTCodec::H265, is_avcc)
    }

    /// Creates CMVideoFormatDescription for H.264 from SPS/PPS.
    /// Returns a retained pointer that must be released with CFRelease.
    fn create_h264_format_description(
        sps: &[u8],
        pps: &[u8],
    ) -> Result<*mut c_void, VideoError> {
        // Prepare parameter set pointers and sizes
        let parameter_sets: [*const u8; 2] = [sps.as_ptr(), pps.as_ptr()];
        let parameter_set_sizes: [usize; 2] = [sps.len(), pps.len()];

        let mut format_desc_ptr: *mut c_void = ptr::null_mut();

        // Use 4-byte NAL length prefix (standard for Annex B to AVCC conversion)
        let nal_unit_header_length: i32 = 4;

        let status = unsafe {
            CMVideoFormatDescriptionCreateFromH264ParameterSets(
                ptr::null(),                  // allocator (NULL = default)
                2,                            // parameter set count
                parameter_sets.as_ptr(),      // parameter set pointers
                parameter_set_sizes.as_ptr(), // parameter set sizes
                nal_unit_header_length,       // NAL unit header length
                &mut format_desc_ptr,         // output format description
            )
        };

        if status != 0 || format_desc_ptr.is_null() {
            return Err(VideoError::DecoderInit(format!(
                "Failed to create H.264 format description: OSStatus {}",
                status
            )));
        }

        Ok(format_desc_ptr)
    }

    /// Creates CMVideoFormatDescription for H.265 from VPS/SPS/PPS.
    /// Returns a retained pointer that must be released with CFRelease.
    #[allow(dead_code)]
    fn create_h265_format_description(
        vps: &[u8],
        sps: &[u8],
        pps: &[u8],
    ) -> Result<*mut c_void, VideoError> {
        // Prepare parameter set pointers and sizes (VPS, SPS, PPS order)
        let parameter_sets: [*const u8; 3] = [vps.as_ptr(), sps.as_ptr(), pps.as_ptr()];
        let parameter_set_sizes: [usize; 3] = [vps.len(), sps.len(), pps.len()];

        let mut format_desc_ptr: *mut c_void = ptr::null_mut();

        // Use 4-byte NAL length prefix
        let nal_unit_header_length: i32 = 4;

        let status = unsafe {
            CMVideoFormatDescriptionCreateFromHEVCParameterSets(
                ptr::null(),                  // allocator (NULL = default)
                3,                            // parameter set count
                parameter_sets.as_ptr(),      // parameter set pointers
                parameter_set_sizes.as_ptr(), // parameter set sizes
                nal_unit_header_length,       // NAL unit header length
                ptr::null(),                  // extensions (NULL for default)
                &mut format_desc_ptr,         // output format description
            )
        };

        if status != 0 || format_desc_ptr.is_null() {
            return Err(VideoError::DecoderInit(format!(
                "Failed to create H.265 format description: OSStatus {}",
                status
            )));
        }

        Ok(format_desc_ptr)
    }

    /// Creates the VTDecompressionSession with IOSurface-compatible output.
    fn create_decoder(
        format_desc: *mut c_void,
        width: u32,
        height: u32,
        codec: VTCodec,
        is_avcc: bool,
    ) -> Result<Self, VideoError> {
        // Create output pixel buffer attributes for IOSurface + Metal compatibility
        let destination_attributes = Self::create_output_attributes()?;

        // Create callback state for receiving decoded frames
        let callback_state = Arc::new(VTCallbackState::new());

        // Create the decompression session
        let session =
            Self::create_session(format_desc, &destination_attributes, &callback_state)?;

        tracing::info!(
            "VTDecoder: Created {:?} session with IOSurface+Metal output (avcc={})",
            codec,
            is_avcc,
        );

        Ok(Self {
            session,
            format_desc,
            callback_state,
            width,
            height,
            codec,
            is_avcc,
            required_drop_streak: 0,
        })
    }

    /// Creates pixel buffer attributes dictionary for IOSurface + Metal output.
    ///
    /// This uses NSMutableDictionary (same pattern as macos_video.rs) to configure:
    /// - kCVPixelBufferPixelFormatTypeKey = kCVPixelFormatType_32BGRA
    /// - kCVPixelBufferIOSurfacePropertiesKey = {} (empty dict enables IOSurface)
    /// - kCVPixelBufferMetalCompatibilityKey = true
    fn create_output_attributes(
    ) -> Result<Retained<NSMutableDictionary<NSString, AnyObject>>, VideoError> {
        unsafe {
            let dict: Retained<NSMutableDictionary<NSString, AnyObject>> =
                NSMutableDictionary::new();

            // Set pixel format to BGRA (matches MacOSVideoDecoder)
            let key_cfstring = kCVPixelBufferPixelFormatTypeKey;
            let pixel_format = NSNumber::numberWithUnsignedInt(kCVPixelFormatType_32BGRA);

            let key_ptr = key_cfstring as *const _ as *const NSString;
            let key: &NSString = &*key_ptr;
            let key_copying: &ProtocolObject<dyn NSCopying> = ProtocolObject::from_ref(key);

            let value_ptr = Retained::as_ptr(&pixel_format) as *mut AnyObject;
            let value: &AnyObject = &*value_ptr;

            dict.setObject_forKey(value, key_copying);

            // Set IOSurface properties (empty dictionary enables IOSurface backing)
            let iosurface_key_cfstring = kCVPixelBufferIOSurfacePropertiesKey;
            let iosurface_key_ptr = iosurface_key_cfstring as *const _ as *const NSString;
            let iosurface_key: &NSString = &*iosurface_key_ptr;
            let iosurface_key_copying: &ProtocolObject<dyn NSCopying> =
                ProtocolObject::from_ref(iosurface_key);
            let iosurface_props: Retained<NSMutableDictionary<NSString, AnyObject>> =
                NSMutableDictionary::new();
            let iosurface_value_ptr = Retained::as_ptr(&iosurface_props) as *mut AnyObject;
            let iosurface_value: &AnyObject = &*iosurface_value_ptr;
            dict.setObject_forKey(iosurface_value, iosurface_key_copying);

            // Set Metal compatibility
            let metal_key_cfstring = kCVPixelBufferMetalCompatibilityKey;
            let metal_key_ptr = metal_key_cfstring as *const _ as *const NSString;
            let metal_key: &NSString = &*metal_key_ptr;
            let metal_key_copying: &ProtocolObject<dyn NSCopying> =
                ProtocolObject::from_ref(metal_key);
            let metal_value = NSNumber::numberWithBool(true);
            let metal_value_ptr = Retained::as_ptr(&metal_value) as *mut AnyObject;
            let metal_value: &AnyObject = &*metal_value_ptr;
            dict.setObject_forKey(metal_value, metal_key_copying);

            tracing::debug!(
                "VTDecoder: Configured output with IOSurface + Metal compatibility"
            );

            Ok(dict)
        }
    }

    /// Creates the VTDecompressionSession with callback.
    /// Returns a retained session pointer.
    fn create_session(
        format_desc: *mut c_void,
        destination_attributes: &NSMutableDictionary<NSString, AnyObject>,
        callback_state: &Arc<VTCallbackState>,
    ) -> Result<*mut c_void, VideoError> {
        // Create callback record with our decompression output handler
        // The callback_state is passed as refcon (reference context) to the callback
        let callback_state_ptr = Arc::into_raw(Arc::clone(callback_state)) as *mut c_void;

        let callback_record = VTDecompressionOutputCallbackRecord {
            decompressionOutputCallback: Some(vt_decode_callback),
            decompressionOutputRefCon: callback_state_ptr,
        };

        let mut session_ptr: *mut c_void = ptr::null_mut();

        // Get raw pointer to the NSDictionary for FFI
        let dest_attrs_ptr = destination_attributes
            as *const NSMutableDictionary<NSString, AnyObject>
            as *const c_void;

        let status = unsafe {
            VTDecompressionSessionCreate(
                ptr::null(),      // allocator
                format_desc,      // video format description
                ptr::null(),      // decoder specification (NULL = auto)
                dest_attrs_ptr,   // destination attributes
                &callback_record, // output callback record
                &mut session_ptr, // output session
            )
        };

        if status != 0 || session_ptr.is_null() {
            // Clean up the Arc we created for the callback
            unsafe { Arc::from_raw(callback_state_ptr as *const VTCallbackState) };
            return Err(VideoError::DecoderInit(format!(
                "VTDecompressionSessionCreate failed: OSStatus {}",
                status
            )));
        }

        Ok(session_ptr)
    }

    /// Decodes a frame from encoded NAL unit data.
    ///
    /// The NAL unit can be in:
    /// - AVCC format (length-prefixed NALs) - used by MoQ/hang
    /// - Annex B format (start code prefixed) - used by raw H.264 streams
    ///
    /// Returns the decoded VideoFrame with IOSurface-backed GPU surface.
    pub fn decode_frame(
        &mut self,
        nal_data: &[u8],
        pts_us: u64,
        is_keyframe: bool,
    ) -> Result<Option<VideoFrame>, VideoError> {
        // Log first few bytes to debug format
        let preview: Vec<u8> = nal_data.iter().take(16).copied().collect();
        tracing::debug!(
            "VTDecoder::decode_frame: {} bytes, keyframe={}, first 16 bytes: {:02x?}",
            nal_data.len(),
            is_keyframe,
            preview
        );

        // Use known format from catalog/init context instead of heuristic detection.
        // The heuristic is_avcc_format() misclassifies AVCC frames with 256-511 byte
        // NAL lengths: the prefix [0x00,0x00,0x01,X] looks like an Annex B start code.
        let is_avcc = self.is_avcc;
        tracing::debug!(
            "VTDecoder: detected format: {}",
            if is_avcc { "AVCC" } else { "Annex B" }
        );

        let avcc_data = if is_avcc {
            // Already in AVCC format, use as-is
            tracing::debug!("VTDecoder: copying {} bytes AVCC data", nal_data.len());
            nal_data.to_vec()
        } else {
            // Annex B format, convert to AVCC
            let converted = Self::annex_b_to_avcc(nal_data);
            tracing::debug!(
                "VTDecoder: converted Annex B to AVCC: {} -> {} bytes",
                nal_data.len(),
                converted.len()
            );
            converted
        };

        tracing::debug!("VTDecoder: avcc_data ready, {} bytes", avcc_data.len());

        if avcc_data.is_empty() {
            return Err(VideoError::DecodeFailed(
                "Empty AVCC data after conversion".to_string(),
            ));
        }

        tracing::debug!("VTDecoder: creating CMBlockBuffer");

        // Create CMBlockBuffer from the AVCC data
        let mut block_buffer_ptr: *mut c_void = ptr::null_mut();

        let status = unsafe {
            CMBlockBufferCreateWithMemoryBlock(
                ptr::null(),                       // allocator for CMBlockBuffer structure
                avcc_data.as_ptr() as *mut c_void, // memory block
                avcc_data.len(),                   // block length
                kCFAllocatorNull, // block allocator: kCFAllocatorNull = caller owns memory, don't free
                ptr::null(),      // custom block source
                0,                // offset into block
                avcc_data.len(),  // data length
                0,                // flags
                &mut block_buffer_ptr, // output block buffer
            )
        };

        if status != 0 || block_buffer_ptr.is_null() {
            return Err(VideoError::DecodeFailed(format!(
                "CMBlockBufferCreate failed: OSStatus {}",
                status
            )));
        }

        tracing::debug!("VTDecoder: CMBlockBuffer created successfully");

        // Create CMSampleBuffer from block buffer
        let mut sample_buffer_ptr: *mut c_void = ptr::null_mut();

        // Create timing info for this frame
        let pts = CMTime::new(pts_us as i64, 1_000_000); // microseconds
        tracing::debug!("VTDecoder: creating timing info, pts_us={}", pts_us);

        let timing_info = CMSampleTimingInfo {
            duration: CMTime::invalid(),
            presentation_time_stamp: pts,
            decode_time_stamp: CMTime::invalid(),
        };

        let sample_size = avcc_data.len();
        tracing::debug!(
            "VTDecoder: creating CMSampleBuffer, sample_size={}, format_desc={:?}",
            sample_size,
            self.format_desc
        );

        let status = unsafe {
            CMSampleBufferCreate(
                ptr::null(),            // allocator
                block_buffer_ptr,       // data buffer
                true,                   // data ready
                ptr::null(),            // make data ready callback
                ptr::null_mut(),        // make data ready refcon
                self.format_desc,       // format description
                1,                      // num samples
                1,                      // num sample timing entries
                &timing_info,           // sample timing array
                1,                      // num sample size entries
                &sample_size,           // sample size array
                &mut sample_buffer_ptr, // output sample buffer
            )
        };
        tracing::debug!("VTDecoder: CMSampleBufferCreate returned status={}", status);

        // Release block buffer (sample buffer retains it if needed)
        unsafe { CFRelease(block_buffer_ptr) };
        tracing::debug!("VTDecoder: released block buffer");

        if status != 0 || sample_buffer_ptr.is_null() {
            return Err(VideoError::DecodeFailed(format!(
                "CMSampleBufferCreate failed: OSStatus {}",
                status
            )));
        }

        // SAFETY: Mutate the sample attachment dictionary to set keyframe flags.
        //
        // - `sample_buffer_ptr` is a valid CMSampleBufferRef created by
        //   `CMSampleBufferCreate` above with `num_samples = 1`, so sample
        //   index 0 is the only valid index.
        // - `CMSampleBufferGetSampleAttachmentsArray(sample_buffer_ptr, true)`
        //   returns a retained CFArrayRef with one entry per sample (i.e. one
        //   mutable CFDictionaryRef at index 0), or null on failure.
        // - `CFArrayGetValueAtIndex(attachments, 0)` yields a non-null
        //   CFDictionaryRef that we may mutate via `CFDictionarySetValue`
        //   because the array was obtained with `createIfNecessary = true`.
        // - Both pointers are checked for null before use.
        // - `is_keyframe` guards which keys are set: keyframes get
        //   DependsOnOthers=false + NotSync=false; non-keyframes get
        //   DependsOnOthers=true.
        unsafe {
            let attachments = CMSampleBufferGetSampleAttachmentsArray(sample_buffer_ptr, true);
            if !attachments.is_null() {
                let dict = CFArrayGetValueAtIndex(attachments, 0);
                if !dict.is_null() {
                    if is_keyframe {
                        CFDictionarySetValue(
                            dict,
                            kCMSampleAttachmentKey_DependsOnOthers,
                            kCFBooleanFalse,
                        );
                        CFDictionarySetValue(
                            dict,
                            kCMSampleAttachmentKey_NotSync,
                            kCFBooleanFalse,
                        );
                    } else {
                        CFDictionarySetValue(
                            dict,
                            kCMSampleAttachmentKey_DependsOnOthers,
                            kCFBooleanTrue,
                        );
                    }
                }
            }
        }

        tracing::debug!(
            "VTDecoder: CMSampleBuffer created (keyframe={}), calling VTDecompressionSessionDecodeFrame",
            is_keyframe
        );

        // Decode the frame synchronously for MoQ live streams
        // Use flag 0 to request synchronous decode
        let decode_flags: u32 = 0;

        let mut info_flags_out: u32 = 0;

        let status = unsafe {
            VTDecompressionSessionDecodeFrame(
                self.session,
                sample_buffer_ptr,
                decode_flags,
                ptr::null_mut(),     // source frame refcon
                &mut info_flags_out, // info flags out
            )
        };
        tracing::debug!(
            "VTDecoder: VTDecompressionSessionDecodeFrame returned status={}, info_flags={}",
            status,
            info_flags_out
        );

        if status != 0 {
            // Release sample buffer before returning error
            unsafe { CFRelease(sample_buffer_ptr) };
            return Err(VideoError::DecodeFailed(format!(
                "VTDecompressionSessionDecodeFrame failed: OSStatus {}",
                status
            )));
        }

        tracing::debug!("VTDecoder: waiting for async frames");

        // Wait for decode to complete BEFORE releasing sample buffer
        // This ensures the memory backing avcc_data stays valid
        let wait_status =
            unsafe { VTDecompressionSessionWaitForAsynchronousFrames(self.session) };
        tracing::debug!("VTDecoder: wait completed, status={}", wait_status);

        // Now safe to release sample buffer - decode is complete
        // Note: avcc_data is still valid here because we used kCFAllocatorNull,
        // so CMBlockBuffer never tried to free it. Rust will drop it at scope end.
        unsafe { CFRelease(sample_buffer_ptr) };
        tracing::debug!("VTDecoder: released sample buffer");

        let status = wait_status;

        if status != 0 {
            tracing::warn!(
                "VTDecompressionSessionWaitForAsynchronousFrames: OSStatus {}",
                status
            );
        }

        // Check for decode errors from callback
        if self
            .callback_state
            .decode_error
            .swap(false, Ordering::AcqRel)
        {
            let cb_status = self
                .callback_state
                .decode_error_status
                .swap(0, Ordering::Relaxed);
            return Err(VideoError::DecodeFailed(format!(
                "VT decode callback error: OSStatus {}",
                cb_status
            )));
        }

        // Pop decoded frame from callback queue
        let queue_len = self.callback_state.decoded_frames.lock().len();
        tracing::debug!("VTDecoder: checking callback queue, length={}", queue_len);
        let decoded = self.callback_state.decoded_frames.lock().pop_front();

        match decoded {
            Some(frame) => {
                if frame.required_frame_dropped {
                    self.required_drop_streak = self.required_drop_streak.saturating_add(1);
                    tracing::debug!(
                        "VTDecoder: callback flagged RequiredFrameDropped on decoded frame (streak={})",
                        self.required_drop_streak
                    );
                    if self.required_drop_streak >= VT_REQUIRED_DROP_STORM_THRESHOLD {
                        let storm_streak = self.required_drop_streak;
                        // Reset so we don't immediately retrigger every frame if
                        // callback flags remain noisy for a short window.
                        self.required_drop_streak = 0;
                        return Err(VideoError::DecodeFailed(format!(
                            "VT required-frame-drop storm (streak={})",
                            storm_streak
                        )));
                    }
                } else if self.required_drop_streak > 0 {
                    tracing::debug!(
                        "VTDecoder: required-frame-drop streak cleared at {}",
                        self.required_drop_streak
                    );
                    self.required_drop_streak = 0;
                }

                tracing::debug!("VTDecoder: got frame from queue, calling create_gpu_frame");
                // Create MacOSGpuSurface from CVPixelBuffer
                let video_frame = self.create_gpu_frame(frame)?;
                tracing::debug!("VTDecoder: create_gpu_frame succeeded");
                Ok(Some(video_frame))
            }
            None => {
                // No frame available yet (async decode may not have completed)
                tracing::debug!("VTDecoder: queue was empty, returning None");
                Ok(None)
            }
        }
    }

    /// Checks if NAL data is in AVCC format (4-byte length prefix) vs Annex B (start codes).
    ///
    /// AVCC format: first 4 bytes are NAL length (big-endian), followed by NAL data
    /// Annex B format: starts with 0x00 0x00 0x00 0x01 or 0x00 0x00 0x01
    ///
    /// NOTE: This heuristic has known false negatives for AVCC frames with 256-511 byte
    /// NALs (length prefix [0,0,1,X] looks like Annex B). Prefer using VTDecoder::is_avcc
    /// field which is set from catalog info.
    #[allow(dead_code)]
    fn is_avcc_format(data: &[u8]) -> bool {
        if data.len() < 5 {
            return false;
        }

        // Check for Annex B start codes first
        if data[0] == 0 && data[1] == 0 {
            if data[2] == 1 {
                return false; // 3-byte start code
            }
            if data[2] == 0 && data[3] == 1 {
                return false; // 4-byte start code
            }
        }

        // Check if first 4 bytes make sense as AVCC length
        let nal_len = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;

        // AVCC length should be reasonable (not zero, not larger than data)
        // and the NAL type byte should be valid (0x01-0x1F for H.264)
        if nal_len > 0 && nal_len <= data.len() - 4 {
            let nal_type = data[4] & 0x1F;
            // Valid H.264 NAL types are 1-23
            if (1..=23).contains(&nal_type) {
                return true;
            }
        }

        false
    }

    /// Converts Annex B NAL data (start code prefixed) to AVCC format (length prefixed).
    fn annex_b_to_avcc(nal_data: &[u8]) -> Vec<u8> {
        // Find NAL unit boundaries (0x00 0x00 0x00 0x01 or 0x00 0x00 0x01)
        let mut result = Vec::with_capacity(nal_data.len() + 4);
        let mut i = 0;

        while i < nal_data.len() {
            // Find start code
            let start_code_len = if i + 4 <= nal_data.len()
                && nal_data[i] == 0
                && nal_data[i + 1] == 0
                && nal_data[i + 2] == 0
                && nal_data[i + 3] == 1
            {
                4
            } else if i + 3 <= nal_data.len()
                && nal_data[i] == 0
                && nal_data[i + 1] == 0
                && nal_data[i + 2] == 1
            {
                3
            } else {
                // No start code at this position, check next byte
                i += 1;
                continue;
            };

            // Find end of this NAL unit (next start code or end of data)
            let nal_start = i + start_code_len;
            let mut nal_end = nal_data.len();

            for j in nal_start..nal_data.len().saturating_sub(2) {
                if nal_data[j] == 0 && nal_data[j + 1] == 0 {
                    if j + 2 < nal_data.len() && nal_data[j + 2] == 1 {
                        nal_end = j;
                        break;
                    }
                    if j + 3 < nal_data.len() && nal_data[j + 2] == 0 && nal_data[j + 3] == 1 {
                        nal_end = j;
                        break;
                    }
                }
            }

            // Write NAL unit with 4-byte length prefix
            let nal_len = nal_end - nal_start;
            result.extend_from_slice(&(nal_len as u32).to_be_bytes());
            result.extend_from_slice(&nal_data[nal_start..nal_end]);

            i = nal_end;
        }

        // If no start codes found, assume raw NAL unit
        if result.is_empty() && !nal_data.is_empty() {
            result.extend_from_slice(&(nal_data.len() as u32).to_be_bytes());
            result.extend_from_slice(nal_data);
        }

        result
    }

    /// Creates a VideoFrame with MacOSGpuSurface from a decoded CVPixelBuffer.
    fn create_gpu_frame(&self, frame: DecodedVTFrame) -> Result<VideoFrame, VideoError> {
        tracing::debug!(
            "VTDecoder: create_gpu_frame called, pts_us={}",
            frame.pts_us
        );
        let pb_ptr = frame.pixel_buffer.0;
        tracing::debug!("VTDecoder: pixel_buffer ptr={:?}", pb_ptr);
        let width = unsafe { CVPixelBufferGetWidth(pb_ptr) } as u32;
        let height = unsafe { CVPixelBufferGetHeight(pb_ptr) } as u32;
        tracing::debug!("VTDecoder: got dimensions {}x{}", width, height);

        // Check for IOSurface availability (confirms hardware decode)
        let io_surface = unsafe { CVPixelBufferGetIOSurface(pb_ptr) };

        if io_surface.is_null() {
            // Fallback: IOSurface not available, would need CPU copy
            // For now, return error as we require zero-copy
            return Err(VideoError::DecodeFailed(
                "VTDecoder: IOSurface not available (software decode?)".to_string(),
            ));
        }

        // Create owner wrapper to keep CVPixelBuffer alive
        let owner: Arc<dyn std::any::Any + Send + Sync> = Arc::new(frame.pixel_buffer);

        // Create MacOSGpuSurface for zero-copy rendering
        let gpu_surface = unsafe {
            MacOSGpuSurface::new(
                io_surface,
                width,
                height,
                PixelFormat::Bgra,
                None, // No CPU fallback - zero-copy only
                owner,
            )
        };

        let pts = Duration::from_micros(frame.pts_us);

        tracing::trace!(
            "VTDecoder: decoded frame {}x{} pts={:?} (zero-copy)",
            width,
            height,
            pts
        );

        Ok(VideoFrame::new(pts, DecodedFrame::MacOS(gpu_surface)))
    }

    /// Returns the codec type (reserved for H.265 support).
    #[allow(dead_code)]
    pub fn codec(&self) -> VTCodec {
        self.codec
    }

    /// Returns the number of frames decoded so far.
    pub fn frame_count(&self) -> u32 {
        self.callback_state
            .frame_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Clears queued output and error flag before waiting for a fresh IDR.
    #[allow(dead_code)]
    pub fn prepare_for_idr_resync(&mut self) {
        // Wait for any pending async frames to complete
        let wait_status =
            unsafe { VTDecompressionSessionWaitForAsynchronousFrames(self.session) };
        if wait_status != 0 {
            tracing::debug!(
                "VTDecoder: wait during IDR resync returned OSStatus {}",
                wait_status
            );
        }
        // Clear the output queue and error flag
        self.callback_state.decoded_frames.lock().clear();
        self.callback_state
            .decode_error
            .store(false, Ordering::Release);
        tracing::debug!("VTDecoder: prepared for IDR resync (cleared queue and error flag)");
    }
}

impl Drop for VTDecoder {
    fn drop(&mut self) {
        let frame_count = self.callback_state.frame_count.load(Ordering::Relaxed);
        tracing::info!("VTDecoder: dropping after decoding {} frames", frame_count);

        if !self.session.is_null() {
            unsafe {
                // Drain any in-flight async decode callbacks before invalidating.
                // Without this, Invalidate can fire while a callback is still writing
                // to callback_state, and the hardware decoder may not fully quiesce.
                let wait_status = VTDecompressionSessionWaitForAsynchronousFrames(self.session);
                if wait_status != 0 {
                    tracing::warn!(
                        "VTDecoder: WaitForAsyncFrames in drop returned OSStatus {}",
                        wait_status
                    );
                }
                VTDecompressionSessionInvalidate(self.session);
                CFRelease(self.session);
            }
        }

        // Release format description
        if !self.format_desc.is_null() {
            unsafe { CFRelease(self.format_desc) };
        }
        tracing::info!("VTDecoder: dropped (session invalidated + released)");
    }
}

/// VTDecompressionSession output callback.
///
/// This is called by VideoToolbox when a frame has been decoded.
/// The decoded CVPixelBuffer is pushed to the callback state queue.
extern "C" fn vt_decode_callback(
    refcon: *mut c_void,
    _source_frame_refcon: *mut c_void,
    status: i32,
    info_flags: u32,
    image_buffer: *mut c_void, // CVImageBufferRef (same as CVPixelBufferRef for video)
    presentation_time_stamp: CMTime,
    _presentation_duration: CMTime,
) {
    tracing::debug!(
        "VT decode callback: status={}, info_flags={}, image_buffer={:?}",
        status,
        info_flags,
        image_buffer
    );

    // Recover the callback state from refcon
    let callback_state = unsafe { &*(refcon as *const VTCallbackState) };

    if status != 0 {
        tracing::error!("VT decode callback error: OSStatus {}", status);
        callback_state
            .decode_error_status
            .store(status, Ordering::Release);
        callback_state.decode_error.store(true, Ordering::Release);
        return;
    }

    // Check for dropped frames
    if info_flags & 0x1 != 0 {
        // kVTDecodeInfo_Asynchronous
        tracing::debug!("VT decode: async frame (info_flags=0x{:x})", info_flags);
    }
    if info_flags & 0x2 != 0 {
        // kVTDecodeInfo_FrameDropped
        tracing::warn!(
            "VT decode: frame dropped by VideoToolbox (info_flags=0x{:x})",
            info_flags
        );
        return;
    }
    if image_buffer.is_null() {
        if info_flags & 0x4 != 0 {
            // kVTDecodeInfo_RequiredFrameDropped with no image — genuine drop
            tracing::warn!(
                "VT decode: frame dropped (info_flags=0x{:x}, no image buffer)",
                info_flags
            );
        } else {
            tracing::warn!("VT decode callback: null image buffer");
        }
        return;
    }

    // info_flags bit 0x4 (RequiredFrameDropped) fires on 100% of frames on
    // macOS 15 / Apple Silicon even when status=0 and image_buffer is valid.
    // When VT successfully produces pixels, treat the frame as good — the flag
    // is informational, not an error signal.
    let required_frame_dropped = if info_flags & 0x4 != 0 && !image_buffer.is_null() {
        tracing::trace!(
            "VT decode: info_flags=0x{:x} with valid image — treating as successful",
            info_flags
        );
        false
    } else {
        false
    };

    // CVImageBuffer is the same as CVPixelBuffer for video frames
    // Retain the pixel buffer so it stays valid
    tracing::debug!(
        "VT decode callback: retaining image_buffer {:?}",
        image_buffer
    );
    let pixel_buffer = unsafe { PixelBufferWrapper::retain(image_buffer) };
    tracing::debug!("VT decode callback: retained, ptr={:?}", pixel_buffer.0);

    // Extract PTS from CMTime
    let pts_us = if presentation_time_stamp.timescale > 0 {
        ((presentation_time_stamp.value as f64 / presentation_time_stamp.timescale as f64)
            * 1_000_000.0) as u64
    } else {
        0
    };

    // Push to decoded frame queue
    let frame = DecodedVTFrame {
        pts_us,
        pixel_buffer,
        required_frame_dropped,
    };
    tracing::debug!("VT decode callback: acquiring lock on decoded_frames queue");
    let mut queue = callback_state.decoded_frames.lock();
    queue.push_back(frame);
    let queue_len = queue.len();
    drop(queue);
    let total_count = callback_state.frame_count.fetch_add(1, Ordering::Relaxed) + 1;

    tracing::debug!(
        "VT decode callback: pushed frame pts={}us, queue_len={}, total_count={}",
        pts_us,
        queue_len,
        total_count
    );
}
