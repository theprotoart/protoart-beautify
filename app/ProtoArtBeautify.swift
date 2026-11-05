// ProtoArt Beautify — native SwiftUI shell around a bundled Python engine.
// Build: see build.sh (compiled with swiftc; no Xcode project needed).

import AppKit
import CoreImage
import CryptoKit
import ImageIO
import SwiftUI
import UniformTypeIdentifiers
import Vision

// MARK: - Settings

struct EnhanceSettings: Equatable, Codable {
    var profile: String = ProcessInfo.processInfo.physicalMemory <= 9 * 1_073_741_824 ? "8gb" : "16gb"
    var prompt = "masterpiece, best quality, highres"
    var negativePrompt = "worst quality, low quality, normal quality"
    var seed = 42.0
    var upscaleFactor = 2.0
    var resemblance = 0.6      // controlnet_scale
    var resemblanceDecay = 1.0 // controlnet_decay
    var hdr = 6.0              // condition_scale
    var creativity = 0.35      // denoise_strength
    var steps = 18.0
    var solver = "DDIM"
    var mode = "detail"        // "detail" = AI re-detailing, "instant" = 4× upscaler only
    var turbo = false          // distilled LCM add-on: ~2 passes, no negative-prompt pass
    var likeness = 0.0         // 0 = off; IP-Adapter Plus strength (keeps the photo's people and things)
    var keepTones = true       // put the photo's own light and colour back after the AI (the new detail stays)
    var faceDetail = false     // Face Detail: redraw each face close up with the same AI (AI presets only)
    var faceDetailStrength = 0.15
    var faceDetailHair = false

    init() {}

    /// Presets saved by an older version may lack newer fields: keep what's there, default the rest.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func v<T: Decodable>(_ k: CodingKeys, _ d: T) -> T { (try? c.decodeIfPresent(T.self, forKey: k)) ?? d }
        let z = EnhanceSettings()
        profile = v(.profile, z.profile); prompt = v(.prompt, z.prompt); negativePrompt = v(.negativePrompt, z.negativePrompt)
        seed = v(.seed, z.seed); upscaleFactor = v(.upscaleFactor, z.upscaleFactor); resemblance = v(.resemblance, z.resemblance)
        resemblanceDecay = v(.resemblanceDecay, z.resemblanceDecay); hdr = v(.hdr, z.hdr); creativity = v(.creativity, z.creativity)
        steps = v(.steps, z.steps); solver = v(.solver, z.solver); mode = v(.mode, z.mode); turbo = v(.turbo, z.turbo)
        likeness = v(.likeness, z.likeness)
        keepTones = v(.keepTones, z.keepTones); faceDetail = v(.faceDetail, z.faceDetail)
        faceDetailStrength = v(.faceDetailStrength, z.faceDetailStrength); faceDetailHair = v(.faceDetailHair, z.faceDetailHair)
    }

    /// Face Detail only runs with the AI (not INSTANT).
    var faceDetailActive: Bool { faceDetail && mode == "detail" }

    var rateKey: String { "\(profile)-\(mode)\(turbo ? "-turbo" : "")" }  // speed memory for estimates

    func params(maxSide: Int) -> [String: Any] {
        var p: [String: Any] = [
            "profile": profile,
            "prompt": prompt,
            "negative_prompt": negativePrompt,
            "seed": Int(seed),
            "upscale_factor": upscaleFactor,
            "controlnet_scale": resemblance,
            "controlnet_decay": resemblanceDecay,
            "condition_scale": hdr,
            "denoise_strength": creativity,
            "num_inference_steps": Int(steps),
            "solver": solver,
            "mode": mode,
            "turbo": turbo,
            "low_memory": true,
            "keep_tones": keepTones,
        ]
        if maxSide > 0 { p["max_output_side"] = maxSide }
        if mode == "detail" && likeness > 0 { p["likeness"] = likeness }
        return p
    }

    /// Mirrors the engine's size logic so the app can estimate queue time.
    func predictedOutput(for input: CGSize, maxSide: Int) -> CGSize {
        var w = Double(input.width), h = Double(input.height), f = upscaleFactor
        guard w > 0, h > 0 else { return .zero }
        let shortCap: Double = profile == "8gb" ? 512 : 768
        let mpCap: Double = (profile == "8gb" ? 8 : 16) * 1_000_000
        if min(w, h) > shortCap { let s = shortCap / min(w, h); w *= s; h *= s }
        if maxSide > 0 {
            let long = max(w, h), cap = Double(maxSide)
            if long * f > cap {
                if long <= cap { f = cap / long } else {
                    f = max(1, min(f, 2)); let s = cap / f / long; w *= s; h *= s
                }
            }
        }
        if w * h * f * f > mpCap { f = (mpCap / (w * h)).squareRoot() }
        f = max(1, f)
        return CGSize(width: w * f, height: h * f)
    }
}

enum Fmt {
    static func duration(_ s: Double) -> String {
        let t = Int(s.rounded())
        if t < 60 { return "\(max(t, 1))s" }
        if t < 3600 { return "\(t / 60)m \(String(format: "%02d", t % 60))s" }
        return "\(t / 3600)h \(String(format: "%02d", (t % 3600) / 60))m"
    }
    static func size(_ s: CGSize) -> String { "\(Int(s.width))×\(Int(s.height))" }
}

// MARK: - Queue items

struct QueueItem: Identifiable, Equatable {
    enum State: Equatable {
        case queued, running, done, stopped, cancelled
        case failed(String)
    }

    let id = UUID()
    let url: URL
    var thumbnail: NSImage?
    var inputSize: CGSize = .zero
    var state: State = .queued
    var message = ""
    var fraction: Double?
    var eta: Double?
    var outputURL: URL?
    var outputSize: CGSize?
    var seconds: Double?
    var savedURL: URL?          // where the result was last saved (copy or overwritten original)
    var filtersBaked = false    // detail filters were applied before the AI ran (already in the output)
    var facesFixed: Int?        // Face Fix ran: how many faces it restored
    var job: JobSnapshot?       // + QUEUE: the settings this row was queued with (nil = use the current ones)

    var name: String { url.lastPathComponent }
    var isFinished: Bool {
        switch state { case .done, .stopped, .cancelled, .failed: return true; default: return false }
    }

    static func == (a: QueueItem, b: QueueItem) -> Bool {
        a.id == b.id && a.state == b.state && a.fraction == b.fraction && a.eta == b.eta
            && a.message == b.message && a.outputURL == b.outputURL && a.thumbnail === b.thumbnail
            && a.savedURL == b.savedURL && a.filtersBaked == b.filtersBaked && a.facesFixed == b.facesFixed
            && a.job == b.job
    }
}

/// Everything a run needs, frozen when the row was queued with + QUEUE (so you can try several settings).
struct JobSnapshot: Equatable {
    var settings: EnhanceSettings
    var maxSide: Int
    var faceFix: Bool
    var faceModel: String
    var faceStrength: Double
    var faceFidelity: Double
    var label: String
}

/// A preset you saved yourself (kept in the app's settings).
struct UserPreset: Codable, Identifiable, Equatable {
    var id = UUID()
    var name: String
    var settings: EnhanceSettings
    var maxSide: Int
    var faceFix: Bool
    var faceModel: String
    var faceStrength: Double
    var faceFidelity: Double
    var filters: DetailFilters?     // only when "include Adjust & filters" was ticked
    var filterStage: String?
}

enum ImageIOHelpers {
    static func pixelSize(_ url: URL) -> CGSize {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let w = props[kCGImagePropertyPixelWidth] as? Int,
              let h = props[kCGImagePropertyPixelHeight] as? Int else { return .zero }
        let orientation = props[kCGImagePropertyOrientation] as? Int ?? 1
        return orientation >= 5 ? CGSize(width: h, height: w) : CGSize(width: w, height: h)
    }

    /// Downsampled, orientation-corrected image: cheap on memory even for huge photos.
    static func image(_ url: URL, maxPixel: Int) -> NSImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let opts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixel,
        ]
        guard let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary) else { return nil }
        return NSImage(cgImage: cg, size: CGSize(width: cg.width, height: cg.height))
    }

    static func cgImage(_ url: URL, maxPixel: Int) -> CGImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let opts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixel,
        ]
        return CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary)
    }

    static func isImage(_ url: URL) -> Bool {
        guard let type = UTType(filenameExtension: url.pathExtension) else { return false }
        return type.conforms(to: .image)
    }

    /// Files are kept; folders are expanded (recursively) to the images inside.
    static func expand(_ urls: [URL]) -> [URL] {
        var out: [URL] = []
        for url in urls {
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else { continue }
            if isDir.boolValue {
                let e = FileManager.default.enumerator(at: url, includingPropertiesForKeys: nil,
                                                       options: [.skipsHiddenFiles, .skipsPackageDescendants])
                var found: [URL] = []
                while let f = e?.nextObject() as? URL { if isImage(f) { found.append(f) } }
                out += found.sorted { $0.path.localizedStandardCompare($1.path) == .orderedAscending }
            } else if isImage(url) {
                out.append(url)
            }
        }
        return out
    }
}

// MARK: - Detail filters (Nik Color Efex-style looks, rendered with Core Image on the GPU)

struct DetailFilters: Codable, Equatable {
    // Camera Raw-style basic adjustments (0 = no change). Light:
    var exposure = 0.0          // -2…2 EV
    var contrast = 0.0          // -1…1
    var highlights = 0.0        // -1…1
    var shadows = 0.0           // -1…1
    var whites = 0.0            // -1…1
    var blacks = 0.0            // -1…1
    // Color:
    var temperature = 0.0       // -1…1 (cooler … warmer)
    var tint = 0.0              // -1…1 (green … magenta)
    var vibrance = 0.0          // -1…1
    var saturation = 0.0        // -1…1
    // Presence (Camera Raw), from Protoart Studio's Adjust:
    var texture = 0.0           // -1…1 fine detail (skin, foliage), not edges
    var clarity = 0.0           // -1…1 mid-tone contrast over larger areas
    var dehaze = 0.0            // -1…1 + takes haze away, - adds a soft veil
    var sharpen = 0.0           // 0…1
    var vignette = 0.0          // -1…1 + darkens, - lightens the corners
    // Color Mixer (HSL): per band red, orange, yellow, green, aqua, blue, purple, magenta; -1…1 each
    var mixHue = [Double](repeating: 0, count: 8)
    var mixSat = [Double](repeating: 0, count: 8)
    var mixLum = [Double](repeating: 0, count: 8)
    // Color Grading: hue (degrees) and amount (0…1) for shadows, midtones, highlights; balance -1…1
    var cgShHue = 220.0, cgShSat = 0.0
    var cgMidHue = 30.0, cgMidSat = 0.0
    var cgHiHue = 40.0, cgHiSat = 0.0
    var cgBalance = 0.0
    // Bleach Bypass: the film look of skipping the bleach bath — silvery, contrasty, half desaturated.
    var bleachOn = false
    var bbBrightness = 0.0      // -1…1
    var bbSaturation = 0.0      // -1…1 (0 = the classic half-desaturated look)
    var bbContrast = 0.0        // -1…1
    var bbLocalContrast = 0.0   // 0…1
    // Dark Contrasts: deeper, moodier tones with crisp mid-tone detail.
    var darkOn = false
    var dcStrength = 0.5        // 0…1
    var dcBrightness = 0.0      // -1…1
    var dcSaturation = 0.0      // -1…1
    // Detail Extractor: pulls out texture everywhere (HDR-like).
    var extractOn = false
    var deAmount = 0.5          // 0…1
    var deContrast = 0.0        // -1…1
    var deSaturation = 0.0      // -1…1
    var deRadius = 1            // 0 fine · 1 normal · 2 coarse
    // Film Grain: photographic grain over the whole picture.
    var grainOn = false
    var grainAmount = 0.5       // 0…1 intensity
    var grainSize = 0.3         // 0…1 grain size
    var grainSoftness = 0.3     // 0…1 (hard … soft)
    var grainColor = false      // monochrome (false) or colour grain
    // Bloom (like Procreate's): the bright parts of the picture spill soft light around themselves.
    var bloomOn = false
    var blAmount = 0.6          // 0…1 how strong the glow is
    var blTransition = 0.4      // 0…1 how much of the picture counts as bright (low = only the brightest)
    var blSize = 0.4            // 0…1 how far the light spreads
    var blBurn = 0.0            // 0…1 screen (0) … adds light, can blow out (1)
    // Glamour Glow (Nik Color Efex): soft, flattering glow.
    var glowOn = false
    var ggGlow = 0.5            // 0…1
    var ggSaturation = 0.0      // -1…1
    var ggWarmth = 0.0          // -1…1
    // Blur: softens; Soft focus screens the blur over the picture.
    var blurOn = false
    var bzSize = 0.3            // 0…1
    var bzMix = 1.0             // 0…1
    var bzKind = 0              // 0 blur · 1 soft focus
    // Skin Texture: procedural pores, fine bumps and optional freckles (shading only).
    var skinOn = false
    var skAmount = 0.5          // 0…1.5
    var skSize = 0.35           // 0…1 pore size
    var skFine = 0.5            // 0…1 fine texture
    var skFreckles = 0.0        // 0…1
    var skSeed = 0.0            // pattern
    /// Stack order of the creative filters (top runs first). Light/Color always run before them.
    var order = DetailFilters.defaultOrder
    static let defaultOrder = ["extract", "dark", "bleach", "glow", "bloom", "blur", "skin", "grain"]  // as in Studio

    init() {}

    /// Settings saved by an older version lack newer filters: keep everything that's there, default the rest.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func v<T: Decodable>(_ k: CodingKeys, _ d: T) -> T { (try? c.decodeIfPresent(T.self, forKey: k)) ?? d }
        let z = DetailFilters()
        exposure = v(.exposure, z.exposure); contrast = v(.contrast, z.contrast)
        highlights = v(.highlights, z.highlights); shadows = v(.shadows, z.shadows)
        whites = v(.whites, z.whites); blacks = v(.blacks, z.blacks)
        temperature = v(.temperature, z.temperature); tint = v(.tint, z.tint)
        vibrance = v(.vibrance, z.vibrance); saturation = v(.saturation, z.saturation)
        bleachOn = v(.bleachOn, z.bleachOn); bbBrightness = v(.bbBrightness, z.bbBrightness)
        bbSaturation = v(.bbSaturation, z.bbSaturation); bbContrast = v(.bbContrast, z.bbContrast)
        bbLocalContrast = v(.bbLocalContrast, z.bbLocalContrast)
        darkOn = v(.darkOn, z.darkOn); dcStrength = v(.dcStrength, z.dcStrength)
        dcBrightness = v(.dcBrightness, z.dcBrightness); dcSaturation = v(.dcSaturation, z.dcSaturation)
        extractOn = v(.extractOn, z.extractOn); deAmount = v(.deAmount, z.deAmount)
        deContrast = v(.deContrast, z.deContrast); deSaturation = v(.deSaturation, z.deSaturation)
        deRadius = v(.deRadius, z.deRadius)
        grainOn = v(.grainOn, z.grainOn); grainAmount = v(.grainAmount, z.grainAmount)
        grainSize = v(.grainSize, z.grainSize); grainSoftness = v(.grainSoftness, z.grainSoftness)
        grainColor = v(.grainColor, z.grainColor)
        bloomOn = v(.bloomOn, z.bloomOn); blAmount = v(.blAmount, z.blAmount)
        blTransition = v(.blTransition, z.blTransition); blSize = v(.blSize, z.blSize); blBurn = v(.blBurn, z.blBurn)
        texture = v(.texture, z.texture); clarity = v(.clarity, z.clarity); dehaze = v(.dehaze, z.dehaze)
        sharpen = v(.sharpen, z.sharpen); vignette = v(.vignette, z.vignette)
        func eight(_ a: [Double]) -> [Double] { a.count == 8 ? a : [Double](repeating: 0, count: 8) }
        mixHue = eight(v(.mixHue, z.mixHue)); mixSat = eight(v(.mixSat, z.mixSat)); mixLum = eight(v(.mixLum, z.mixLum))
        cgShHue = v(.cgShHue, z.cgShHue); cgShSat = v(.cgShSat, z.cgShSat)
        cgMidHue = v(.cgMidHue, z.cgMidHue); cgMidSat = v(.cgMidSat, z.cgMidSat)
        cgHiHue = v(.cgHiHue, z.cgHiHue); cgHiSat = v(.cgHiSat, z.cgHiSat); cgBalance = v(.cgBalance, z.cgBalance)
        glowOn = v(.glowOn, z.glowOn); ggGlow = v(.ggGlow, z.ggGlow)
        ggSaturation = v(.ggSaturation, z.ggSaturation); ggWarmth = v(.ggWarmth, z.ggWarmth)
        blurOn = v(.blurOn, z.blurOn); bzSize = v(.bzSize, z.bzSize); bzMix = v(.bzMix, z.bzMix); bzKind = v(.bzKind, z.bzKind)
        skinOn = v(.skinOn, z.skinOn); skAmount = v(.skAmount, z.skAmount); skSize = v(.skSize, z.skSize)
        skFine = v(.skFine, z.skFine); skFreckles = v(.skFreckles, z.skFreckles); skSeed = v(.skSeed, z.skSeed)
        order = DetailFilters.fullOrder(v(.order, z.order))
    }

    /// Adds filters a saved order doesn't know yet (new ones go before Film Grain when grain is last).
    static func fullOrder(_ saved: [String]) -> [String] {
        var o = saved.filter { defaultOrder.contains($0) }
        o = o.reduce(into: [String]()) { if !$0.contains($1) { $0.append($1) } }
        for id in defaultOrder where !o.contains(id) {
            if o.last == "grain" { o.insert(id, at: o.count - 1) } else { o.append(id) }
        }
        return o
    }

    var lightOn: Bool { exposure != 0 || contrast != 0 || highlights != 0 || shadows != 0 || whites != 0 || blacks != 0 }
    var colorOn: Bool { temperature != 0 || tint != 0 || vibrance != 0 || saturation != 0 }
    var presenceOn: Bool { texture != 0 || clarity != 0 || dehaze != 0 || sharpen != 0 || vignette != 0 }
    var mixerOn: Bool { (mixHue + mixSat + mixLum).contains { $0 != 0 } }
    var gradingOn: Bool { cgShSat != 0 || cgMidSat != 0 || cgHiSat != 0 }
    var anyOn: Bool {
        lightOn || colorOn || presenceOn || mixerOn || gradingOn || bleachOn || darkOn || extractOn || grainOn
            || (bloomOn && blAmount > 0) || (glowOn && ggGlow > 0) || (blurOn && bzMix > 0) || (skinOn && skAmount > 0)
    }

    mutating func resetLight() { exposure = 0; contrast = 0; highlights = 0; shadows = 0; whites = 0; blacks = 0 }
    mutating func resetColor() { temperature = 0; tint = 0; vibrance = 0; saturation = 0 }
    mutating func resetPresence() { texture = 0; clarity = 0; dehaze = 0; sharpen = 0; vignette = 0 }
    mutating func resetMixer() {
        mixHue = [Double](repeating: 0, count: 8); mixSat = mixHue; mixLum = mixHue
    }
    mutating func resetGrading() {
        let z = DetailFilters()
        cgShHue = z.cgShHue; cgShSat = 0; cgMidHue = z.cgMidHue; cgMidSat = 0; cgHiHue = z.cgHiHue; cgHiSat = 0; cgBalance = 0
    }

    static func load() -> DetailFilters {
        guard let d = UserDefaults.standard.data(forKey: "detailFilters"),
              let f = try? JSONDecoder().decode(DetailFilters.self, from: d) else { return DetailFilters() }
        return f
    }

    func save() {
        if let d = try? JSONEncoder().encode(self) { UserDefaults.standard.set(d, forKey: "detailFilters") }
    }
}

/// Core Image kernels for the filters ported from Protoart Studio's Adjust (its WebGL shaders, same maths).
/// Inputs are in display gamma (sRGB-encoded). A kernel that fails to compile is nil: that step is skipped
/// (and logged) instead of crashing the app.
enum Kernels {
    private static func make(_ name: String, _ src: String) -> CIColorKernel? {
        let k = CIColorKernel(source: src)
        if k == nil { NSLog("ProtoArt Beautify: the %@ filter's kernel didn't compile; skipping it", name) }
        return k
    }
    private static let lumaFn = "float pabLuma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }\n"

    /// unsharp mask c + amount * (c - blur); midtone-weighted for Texture / Clarity
    static let local = make("local", lumaFn + """
    kernel vec4 pabLocal(__sample c, __sample b, float amount, float mid) {
        float w = 1.0;
        if (mid > 0.5) { float L = pabLuma(c.rgb); w = 1.0 - pow(abs(L - 0.5) * 2.0, 2.0); }
        return vec4(c.rgb + amount * w * (c.rgb - b.rgb), c.a);
    }
    """)

    /// Dehaze: haze guessed from the local darkest channel (dark-channel prior)
    static let dehaze = make("dehaze", lumaFn + """
    kernel vec4 pabDehaze(__sample s, __sample b, float amount) {
        vec3 c = s.rgb;
        float dark = min(b.r, min(b.g, b.b));
        float A = 0.95;
        if (amount > 0.0) {
            float t = max(1.0 - amount * 0.9 * dark / A, 0.35);
            vec3 r = (c - A * (1.0 - t)) / t;
            float L0 = pabLuma(c);
            float L1 = pabLuma(r);
            r = mix(vec3(L1), r, 1.0 + 0.25 * amount);
            c = mix(c, r, 0.85) + (L0 - L1) * 0.25 * amount;
        } else {
            float k = -amount * (0.35 + 0.45 * (1.0 - dark));
            c = mix(c, vec3(A) * 0.85 + c * 0.15, k * 0.7);
        }
        return vec4(max(c, vec3(0.0)), s.a);
    }
    """)

    static let vignette = make("vignette", """
    kernel vec4 pabVignette(__sample c, vec2 center, float scale, float amount) {
        vec2 d = (destCoord() - center) * scale;
        float r = length(d) / 1.4142;
        float v = smoothstep(0.35, 1.05, r);
        vec3 o = amount >= 0.0 ? c.rgb * (1.0 - amount * 0.85 * v) : mix(c.rgb, vec3(1.0), -amount * 0.7 * v);
        return vec4(o, c.a);
    }
    """)

    /// Color Mixer (HSL): bands centred at 0, 30, 60, 120, 180, 240, 270, 300 degrees
    static let mixer = make("mixer", lumaFn + """
    float pabBand(vec4 a, vec4 b, float h) {
        if (h < 30.0) return mix(a.x, a.y, h / 30.0);
        if (h < 60.0) return mix(a.y, a.z, (h - 30.0) / 30.0);
        if (h < 120.0) return mix(a.z, a.w, (h - 60.0) / 60.0);
        if (h < 180.0) return mix(a.w, b.x, (h - 120.0) / 60.0);
        if (h < 240.0) return mix(b.x, b.y, (h - 180.0) / 60.0);
        if (h < 270.0) return mix(b.y, b.z, (h - 240.0) / 30.0);
        if (h < 300.0) return mix(b.z, b.w, (h - 270.0) / 30.0);
        return mix(b.w, a.x, (h - 300.0) / 60.0);
    }
    vec3 pabHsl(vec3 c) {
        float mx = max(c.r, max(c.g, c.b));
        float mn = min(c.r, min(c.g, c.b));
        float l = (mx + mn) * 0.5;
        float d = mx - mn;
        float h = 0.0;
        float s = 0.0;
        if (d > 0.00001) {
            s = d / (1.0 - abs(2.0 * l - 1.0) + 0.00001);
            if (mx == c.r) { h = mod((c.g - c.b) / d, 6.0); }
            else if (mx == c.g) { h = (c.b - c.r) / d + 2.0; }
            else { h = (c.r - c.g) / d + 4.0; }
            h = h * 60.0;
        }
        return vec3(h, s, l);
    }
    vec3 pabRgb(vec3 h) {
        float c = (1.0 - abs(2.0 * h.z - 1.0)) * h.y;
        float x = c * (1.0 - abs(mod(h.x / 60.0, 2.0) - 1.0));
        float m = h.z - c * 0.5;
        vec3 r = vec3(c, 0.0, x);
        if (h.x < 60.0) { r = vec3(c, x, 0.0); }
        else if (h.x < 120.0) { r = vec3(x, c, 0.0); }
        else if (h.x < 180.0) { r = vec3(0.0, c, x); }
        else if (h.x < 240.0) { r = vec3(0.0, x, c); }
        else if (h.x < 300.0) { r = vec3(x, 0.0, c); }
        return r + m;
    }
    kernel vec4 pabMixer(__sample s, vec4 h0, vec4 h1, vec4 s0, vec4 s1, vec4 l0, vec4 l1) {
        vec3 c = clamp(s.rgb, 0.0, 1.0);
        vec3 h = pabHsl(c);
        float w = smoothstep(0.02, 0.18, h.y);
        float dh = pabBand(h0, h1, h.x);
        float ds = pabBand(s0, s1, h.x);
        float dl = pabBand(l0, l1, h.x);
        h.x = mod(h.x + dh * 30.0 * w + 360.0, 360.0);
        h.y = clamp(h.y * (1.0 + ds * w), 0.0, 1.0);
        vec3 r = pabRgb(h);
        r = r * max(0.0, 1.0 + dl * 0.6 * w * h.y);
        return vec4(r, s.a);
    }
    """)

    /// Color Grading: tint shadows / midtones / highlights; brightness kept
    static let grade = make("grade", lumaFn + """
    kernel vec4 pabGrade(__sample s, vec3 sh, vec3 mid, vec3 hi, float balance) {
        vec3 c = s.rgb;
        float L = clamp(pabLuma(c), 0.0, 1.0);
        float b = 0.5 - 0.35 * balance;
        float ws = 1.0 - smoothstep(0.0, b * 1.6, L);
        float wh = smoothstep(b * 0.6, 1.0, L);
        float wm = max(0.0, 1.0 - ws - wh);
        vec3 r = c + sh * ws + mid * wm + hi * wh;
        r = r * (L / max(pabLuma(r), 0.0001));
        return vec4(max(r, vec3(0.0)), s.a);
    }
    """)

    static let glow = make("glow", """
    kernel vec4 pabGlow(__sample s, __sample bl, float amount, float warmth) {
        vec3 c = clamp(s.rgb, 0.0, 1.0);
        vec3 b = clamp(bl.rgb, 0.0, 1.0);
        vec3 scr = 1.0 - (1.0 - c) * (1.0 - b);
        c = mix(c, scr, amount * 0.7);
        c = mix(c, c * 0.9 + 0.1 * b, amount * 0.3);
        c = c * vec3(1.0 + 0.06 * warmth, 1.0 + 0.01 * warmth, 1.0 - 0.06 * warmth);
        return vec4(c, s.a);
    }
    """)

    static let blurMix = make("blur", """
    kernel vec4 pabBlurMix(__sample s, __sample bl, float amount, float soft) {
        vec3 c = s.rgb;
        vec3 b = bl.rgb;
        vec3 t = b;
        if (soft > 0.5) { t = mix(c, 1.0 - (1.0 - c) * (1.0 - b), 0.6) * 0.5 + b * 0.5; }
        return vec4(mix(c, t, amount), s.a);
    }
    """)

    /// Skin Texture: cellular pores, fine bumps, optional freckles; shading only, mid-tone weighted
    static let skin = make("skin", lumaFn + """
    vec2 pabHash2(vec2 p, float seed) {
        p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
        return fract(sin(p + seed) * 43758.5453);
    }
    float pabHash1(vec2 p, float seed) { return fract(sin(dot(p, vec2(41.7, 289.3)) + seed) * 24634.6345); }
    float pabPore(vec2 x, float keep, float seed) {
        vec2 i = floor(x);
        vec2 f = fract(x);
        float best = 0.0;
        for (int y = -1; y <= 1; y++) {
            for (int k = -1; k <= 1; k++) {
                vec2 g = vec2(float(k), float(y));
                vec2 id = i + g;
                float h = pabHash1(id + 7.3, seed);
                if (h <= keep) {
                    vec2 r = g + pabHash2(id, seed) - f;
                    float rad = 0.12 + 0.16 * pabHash1(id + 3.1, seed);
                    float depth = 0.5 + 0.5 * pabHash1(id + 5.7, seed);
                    best = max(best, depth * (1.0 - smoothstep(rad * 0.25, rad, length(r))));
                }
            }
        }
        return best;
    }
    float pabNoise(vec2 x, float seed) {
        vec2 i = floor(x);
        vec2 f = fract(x);
        f = f * f * (3.0 - 2.0 * f);
        return mix(mix(pabHash1(i, seed), pabHash1(i + vec2(1.0, 0.0), seed), f.x),
                   mix(pabHash1(i + vec2(0.0, 1.0), seed), pabHash1(i + vec2(1.0, 1.0), seed), f.x), f.y);
    }
    kernel vec4 pabSkin(__sample s, float amount, float cell, float fine, float freckles, float seed) {
        vec3 c = s.rgb;
        vec2 px = destCoord();
        float dens = 0.35 + 0.35 * pabNoise(px / (cell * 12.0) + 5.0, seed);
        float pores = pabPore(px / cell, dens, seed) + 0.6 * pabPore(px / (cell * 1.8) + 13.0, dens * 0.5, seed);
        float bumps = (pabNoise(px / (cell * 0.6), seed) - 0.5) + 0.5 * (pabNoise(px / (cell * 0.3) + 9.0, seed) - 0.5)
                    + 0.6 * (pabNoise(px / (cell * 2.5) + 21.0, seed) - 0.5);
        float L = pabLuma(c);
        float mid = smoothstep(0.04, 0.3, L) * (1.0 - smoothstep(0.9, 1.0, L));
        float t = (-0.3 * pores + fine * 0.22 * bumps) * amount * mid;
        float lit = smoothstep(0.45, 0.85, L);
        c = c * (1.0 + t) + vec3(0.025 * fine * max(0.0, bumps) * amount * mid * lit);
        if (freckles > 0.0) {
            float cluster = smoothstep(0.4, 0.75, pabNoise(px / (cell * 40.0) + 3.0, seed));
            float spot = pabPore(px / (cell * 3.0) + 41.0, 0.3 + 0.4 * cluster, seed);
            c = mix(c, c * vec3(0.82, 0.66, 0.52), spot * freckles * 0.75 * mid);
        }
        return vec4(max(c, vec3(0.0)), s.a);
    }
    """)

    /// Color Grading tint: a colour of that hue centred on grey, scaled by the amount (as Studio's tintOf)
    static func tint(_ hue: Double, _ sat: Double) -> CIVector {
        let h = (hue.truncatingRemainder(dividingBy: 360) + 360).truncatingRemainder(dividingBy: 360)
        func f(_ n: Double) -> Double {
            let k = (n + h / 30).truncatingRemainder(dividingBy: 12)
            return 0.5 - 0.5 * max(-1, min(k - 3, 9 - k, 1))
        }
        let a = 0.35 * sat
        return CIVector(x: (f(0) - 0.5) * a, y: (f(8) - 0.5) * a, z: (f(4) - 0.5) * a)
    }
}

enum DetailFilterEngine {
    static let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!
    static let context = CIContext(options: [.cacheIntermediates: false])

    /// Chain order: Detail Extractor → Dark Contrasts → Bleach Bypass. Radii are relative to the image's
    /// long side, so the small live preview and the full-size export look the same.
    static func apply(_ f: DetailFilters, to input: CIImage) -> CIImage {
        let ext = input.extent
        let unit = max(ext.width, ext.height) / 1000
        var img = input
        func localContrast(_ i: CIImage, radius: Double, amount: Double) -> CIImage {
            guard amount > 0.001 else { return i }
            return i.clampedToExtent().applyingFilter("CIUnsharpMask", parameters: [
                kCIInputRadiusKey: radius * unit, kCIInputIntensityKey: amount]).cropped(to: ext)
        }
        // 1. Basic adjustments (Camera Raw "Light" and "Color"), before the creative filters.
        if f.exposure != 0 {
            img = img.applyingFilter("CIExposureAdjust", parameters: [kCIInputEVKey: f.exposure])
        }
        if f.shadows > 0 || f.highlights < 0 {
            // local recovery: lift shadows / pull down highlights using their surroundings
            img = img.applyingFilter("CIHighlightShadowAdjust", parameters: [
                "inputRadius": 12 * unit,
                "inputShadowAmount": max(0, f.shadows), "inputHighlightAmount": 1 + min(0, f.highlights)])
        }
        if f.shadows < 0 || f.highlights > 0 || f.whites != 0 || f.blacks != 0 {
            // tone curve: deepen shadows / brighten highlights, and move the white and black points
            let b0 = CIVector(x: 0.08 * max(0, -f.blacks), y: 0.08 * max(0, f.blacks))
            let b1 = CIVector(x: 0.25, y: 0.25 + 0.08 * min(0, f.shadows))
            let b2 = CIVector(x: 0.5, y: 0.5)
            let b3 = CIVector(x: 0.75, y: 0.75 + 0.08 * max(0, f.highlights))
            let b4 = CIVector(x: 1 - 0.08 * max(0, f.whites), y: 1 - 0.08 * max(0, -f.whites))
            img = img.applyingFilter("CIToneCurve", parameters: [
                "inputPoint0": b0, "inputPoint1": b1, "inputPoint2": b2, "inputPoint3": b3, "inputPoint4": b4])
        }
        if f.contrast != 0 || f.saturation != 0 {
            img = img.applyingFilter("CIColorControls", parameters: [
                kCIInputContrastKey: 1 + 0.5 * f.contrast, kCIInputSaturationKey: max(0, 1 + f.saturation),
                kCIInputBrightnessKey: 0])
        }
        if f.temperature != 0 || f.tint != 0 {
            img = img.applyingFilter("CITemperatureAndTint", parameters: [
                "inputNeutral": CIVector(x: 6500 + 3000 * f.temperature, y: 60 * f.tint),
                "inputTargetNeutral": CIVector(x: 6500, y: 0)])
        }
        if f.vibrance != 0 {
            img = img.applyingFilter("CIVibrance", parameters: ["inputAmount": f.vibrance])
        }
        // 1b. Presence, Color Mixer and Color Grading (Protoart Studio's Adjust maths, run in display gamma).
        if f.presenceOn || f.mixerOn || f.gradingOn {
            var g = img.applyingFilter("CILinearToSRGBToneCurve")
            func blurred(_ i: CIImage, _ r: Double) -> CIImage {
                i.clampedToExtent().applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: r]).cropped(to: ext)
            }
            func local(_ i: CIImage, radius: Double, amount: Double, midtones: Bool) -> CIImage {
                guard amount != 0, let k = Kernels.local else { return i }
                return k.apply(extent: ext, arguments: [i, blurred(i, radius * unit), amount, midtones ? 1.0 : 0.0]) ?? i
            }
            if f.dehaze != 0, let k = Kernels.dehaze {   // first, like Camera Raw: it works on the whole scene
                g = k.apply(extent: ext, arguments: [g, blurred(g, 18 * unit), f.dehaze]) ?? g
            }
            g = local(g, radius: 3.5, amount: 1.1 * f.texture, midtones: true)
            g = local(g, radius: 30, amount: 0.9 * f.clarity, midtones: true)
            if f.sharpen > 0 { g = local(g, radius: 0.9, amount: 1.6 * f.sharpen, midtones: false) }
            if f.vignette != 0, let k = Kernels.vignette {
                let long = Double(max(ext.width, ext.height))
                g = k.apply(extent: ext, arguments: [g, CIVector(x: ext.midX, y: ext.midY), 2 / long, f.vignette]) ?? g
            }
            if f.mixerOn, let k = Kernels.mixer {
                func v4(_ a: [Double], _ o: Int) -> CIVector { CIVector(x: a[o], y: a[o + 1], z: a[o + 2], w: a[o + 3]) }
                g = k.apply(extent: ext, arguments: [g, v4(f.mixHue, 0), v4(f.mixHue, 4), v4(f.mixSat, 0), v4(f.mixSat, 4),
                                                     v4(f.mixLum, 0), v4(f.mixLum, 4)]) ?? g
            }
            if f.gradingOn, let k = Kernels.grade {
                g = k.apply(extent: ext, arguments: [g, Kernels.tint(f.cgShHue, f.cgShSat), Kernels.tint(f.cgMidHue, f.cgMidSat),
                                                     Kernels.tint(f.cgHiHue, f.cgHiSat), f.cgBalance]) ?? g
            }
            img = g.applyingFilter("CISRGBToneCurveToLinear")
        }
        // 2. Creative filters, in the order the user stacked them (top of the list runs first).
        for id in f.order {
            switch id {
            case "extract":
                guard f.extractOn else { continue }
                let r = [2.0, 5.0, 14.0][min(2, max(0, f.deRadius))]
                img = img.applyingFilter("CIHighlightShadowAdjust", parameters: [
                    "inputShadowAmount": 0.5 * f.deAmount, "inputHighlightAmount": 1 - 0.3 * f.deAmount])
                img = localContrast(img, radius: r, amount: 1.6 * f.deAmount)
                img = img.applyingFilter("CIColorControls", parameters: [
                    kCIInputContrastKey: 1 + 0.3 * f.deContrast, kCIInputSaturationKey: max(0, 1 + f.deSaturation),
                    kCIInputBrightnessKey: 0])
            case "dark":
                guard f.darkOn else { continue }
                img = localContrast(img, radius: 5, amount: 1.4 * f.dcStrength)
                img = img.applyingFilter("CIGammaAdjust", parameters: [
                    "inputPower": max(0.4, 1 + 0.35 * f.dcStrength - 0.45 * f.dcBrightness)])
                // keep detail in the deepest shadows instead of crushing them to black
                img = img.applyingFilter("CIHighlightShadowAdjust", parameters: [
                    "inputShadowAmount": 0.4 * f.dcStrength, "inputHighlightAmount": 1])
                img = img.applyingFilter("CIColorControls", parameters: [
                    kCIInputContrastKey: 1 + 0.12 * f.dcStrength, kCIInputSaturationKey: max(0, 0.8 + 0.8 * f.dcSaturation),
                    kCIInputBrightnessKey: 0])
            case "bleach":
                guard f.bleachOn else { continue }
                let mono = img.applyingFilter("CIColorControls", parameters: [kCIInputSaturationKey: 0])
                let overlay = mono.applyingFilter("CIOverlayBlendMode", parameters: [kCIInputBackgroundImageKey: img])
                // 65% of the overlay: the silvery bleach-bypass bite without crushing the blacks
                img = img.applyingFilter("CIDissolveTransition", parameters: [
                    kCIInputTargetImageKey: overlay, kCIInputTimeKey: 0.65])
                img = img.applyingFilter("CIColorControls", parameters: [
                    kCIInputSaturationKey: max(0, 0.5 + 0.5 * f.bbSaturation),
                    kCIInputContrastKey: max(0.25, 1.0 + 0.45 * f.bbContrast),
                    kCIInputBrightnessKey: 0.2 * f.bbBrightness])
                img = localContrast(img, radius: 22, amount: 1.4 * f.bbLocalContrast)
            case "bloom":
                guard f.bloomOn, f.blAmount > 0 else { continue }
                img = bloom(img, f, unit: unit, extent: ext)
            case "glow":
                guard f.glowOn, f.ggGlow > 0, let k = Kernels.glow else { continue }
                let g = img.applyingFilter("CILinearToSRGBToneCurve")
                let b = g.clampedToExtent().applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: 10 * unit]).cropped(to: ext)
                var out = k.apply(extent: ext, arguments: [g, b, f.ggGlow, f.ggWarmth]) ?? g
                if f.ggSaturation != 0 {
                    out = out.applyingFilter("CIColorControls", parameters: [kCIInputSaturationKey: max(0, 1 + f.ggSaturation),
                                                                             kCIInputContrastKey: 1, kCIInputBrightnessKey: 0])
                }
                img = out.applyingFilter("CISRGBToneCurveToLinear")
            case "blur":
                guard f.blurOn, f.bzMix > 0, let k = Kernels.blurMix else { continue }
                let g = img.applyingFilter("CILinearToSRGBToneCurve")
                let r = (0.5 + 40 * pow(min(1, max(0, f.bzSize)), 1.6)) * unit
                let b = g.clampedToExtent().applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: r]).cropped(to: ext)
                img = (k.apply(extent: ext, arguments: [g, b, min(1, max(0, f.bzMix)), f.bzKind == 1 ? 1.0 : 0.0]) ?? g)
                    .applyingFilter("CISRGBToneCurveToLinear")
            case "skin":
                guard f.skinOn, f.skAmount > 0, let k = Kernels.skin else { continue }
                let g = img.applyingFilter("CILinearToSRGBToneCurve")
                let cell = (1.2 + 5.5 * min(1, max(0, f.skSize))) * unit
                img = (k.apply(extent: ext, arguments: [g, min(1.5, f.skAmount), cell, min(1, max(0, f.skFine)),
                                                        min(1, max(0, f.skFreckles)), f.skSeed]) ?? g)
                    .applyingFilter("CISRGBToneCurveToLinear")
            case "grain":
                guard f.grainOn else { continue }
                // Film grain: random noise blown up to the grain size (relative to the picture, so the
                // preview matches the export), softened, centred on mid-grey and overlaid.
                let cell = unit * (0.5 + 2.5 * f.grainSize)
                var noise = CIFilter(name: "CIRandomGenerator")!.outputImage!
                    .transformed(by: CGAffineTransform(scaleX: cell, y: cell))
                if f.grainSoftness > 0.01 {
                    noise = noise.applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: cell * 0.6 * f.grainSoftness])
                }
                noise = noise.applyingFilter("CIColorControls", parameters: [
                    kCIInputSaturationKey: f.grainColor ? 0.35 : 0, kCIInputContrastKey: 1, kCIInputBrightnessKey: 0])
                let a = 0.7 * f.grainAmount
                noise = noise.applyingFilter("CIColorMatrix", parameters: [
                    "inputRVector": CIVector(x: a, y: 0, z: 0, w: 0),
                    "inputGVector": CIVector(x: 0, y: a, z: 0, w: 0),
                    "inputBVector": CIVector(x: 0, y: 0, z: a, w: 0),
                    "inputBiasVector": CIVector(x: 0.5 * (1 - a), y: 0.5 * (1 - a), z: 0.5 * (1 - a), w: 0),
                ]).cropped(to: ext)
                img = noise.applyingFilter("CIOverlayBlendMode", parameters: [kCIInputBackgroundImageKey: img])
            default:
                continue
            }
        }
        return img.cropped(to: ext)
    }

    /// Bloom (like Procreate's, same recipe as Protoart Studio's Adjust). Core Image works in linear light, so:
    /// 1. pick the bright parts: each pixel's brightest channel through a soft threshold (Transition sets how
    ///    much of the picture counts as bright), times the pixel;  2. blur them twice, a tight core and a
    ///    3.5× wider halo (Size);  3. screen that light back on (never darkens), or with Burn add it, which
    ///    can blow out to white.
    static func bloom(_ img: CIImage, _ f: DetailFilters, unit: CGFloat, extent ext: CGRect) -> CIImage {
        let t = min(1, max(0, f.blTransition))
        let threshold = 0.92 - 0.7 * t, knee = 0.04 + 0.2 * t        // in display (gamma) values, as in Studio
        let lo = pow(max(0, threshold - knee), 2.2), hi = pow(min(1, threshold + knee), 2.2)  // → linear
        let span = max(1e-4, hi - lo)
        // mask = smoothstep(lo, hi, max(r, g, b))
        var mask = img.applyingFilter("CIMaximumComponent")
            .applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: 1 / span, y: 0, z: 0, w: 0),
                "inputGVector": CIVector(x: 0, y: 1 / span, z: 0, w: 0),
                "inputBVector": CIVector(x: 0, y: 0, z: 1 / span, w: 0),
                "inputBiasVector": CIVector(x: -lo / span, y: -lo / span, z: -lo / span, w: 0)])
            .applyingFilter("CIColorClamp", parameters: [
                "inputMinComponents": CIVector(x: 0, y: 0, z: 0, w: 1),
                "inputMaxComponents": CIVector(x: 1, y: 1, z: 1, w: 1)])
        mask = mask.applyingFilter("CIToneCurve", parameters: [   // smooth the ramp's ends (smoothstep)
            "inputPoint0": CIVector(x: 0, y: 0), "inputPoint1": CIVector(x: 0.25, y: 0.156),
            "inputPoint2": CIVector(x: 0.5, y: 0.5), "inputPoint3": CIVector(x: 0.75, y: 0.844),
            "inputPoint4": CIVector(x: 1, y: 1)])
        let pick = img.applyingFilter("CIMultiplyCompositing", parameters: [kCIInputBackgroundImageKey: mask])
            .clampedToExtent()
        let r = (4 + 56 * min(1, max(0, f.blSize))) * unit
        let burn = min(1, max(0, f.blBurn))
        let gain = f.blAmount * (1.6 + 4.0 * burn)
        func scaled(_ i: CIImage, _ k: Double) -> CIImage {
            i.applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: k, y: 0, z: 0, w: 0), "inputGVector": CIVector(x: 0, y: k, z: 0, w: 0),
                "inputBVector": CIVector(x: 0, y: 0, z: k, w: 0), "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1)])
        }
        let near = scaled(pick.applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: r]), 0.6 * gain)
        let far = scaled(pick.applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: r * 3.5]), 0.8 * gain)
        let light = near.applyingFilter("CIAdditionCompositing", parameters: [kCIInputBackgroundImageKey: far])
            .cropped(to: ext)
        let lightClamped = light.applyingFilter("CIColorClamp", parameters: [
            "inputMinComponents": CIVector(x: 0, y: 0, z: 0, w: 1),
            "inputMaxComponents": CIVector(x: 1, y: 1, z: 1, w: 1)])
        let screen = lightClamped.applyingFilter("CIScreenBlendMode", parameters: [kCIInputBackgroundImageKey: img])
        guard burn > 0.001 else { return screen.cropped(to: ext) }
        let added = light.applyingFilter("CIAdditionCompositing", parameters: [kCIInputBackgroundImageKey: img])
        return screen.applyingFilter("CIDissolveTransition", parameters: [
            kCIInputTargetImageKey: added, kCIInputTimeKey: burn]).cropped(to: ext)
    }

    private static func rgbSpace(_ space: CGColorSpace?) -> CGColorSpace {
        if let space, space.model == .rgb { return space }
        return sRGB
    }

    /// Colour-managed conversion to plain 8-bit sRGB.
    static func toSRGB(_ cg: CGImage) -> CGImage? {
        let ci = CIImage(cgImage: cg)
        return context.createCGImage(ci, from: ci.extent, format: .RGBA8, colorSpace: sRGB)
    }

    static func render(_ f: DetailFilters, _ cg: CGImage) -> CGImage? {
        let ci = CIImage(cgImage: cg)
        return context.createCGImage(apply(f, to: ci), from: ci.extent, format: .RGBA8,
                                     colorSpace: rgbSpace(cg.colorSpace))
    }

    /// Full-size, orientation-corrected pixels of a file.
    static func loadFull(_ url: URL) -> CGImage? {
        guard let ci = CIImage(contentsOf: url, options: [.applyOrientationProperty: true]) else { return nil }
        return context.createCGImage(ci, from: ci.extent, format: .RGBA8, colorSpace: rgbSpace(ci.colorSpace))
    }

    static func canWrite(_ type: UTType) -> Bool {
        let ids = (CGImageDestinationCopyTypeIdentifiers() as? [String]) ?? []
        return ids.contains(type.identifier)
    }

    static func write(_ cg: CGImage, to url: URL, type: UTType) throws {
        guard let dst = CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil) else {
            throw NSError(domain: "Save", code: 1, userInfo: [NSLocalizedDescriptionKey:
                "This Mac can't write .\(url.pathExtension) files."])
        }
        CGImageDestinationAddImage(dst, cg, [kCGImageDestinationLossyCompressionQuality: 0.95] as CFDictionary)
        guard CGImageDestinationFinalize(dst) else {
            throw NSError(domain: "Save", code: 2, userInfo: [NSLocalizedDescriptionKey: "Writing the file failed."])
        }
    }

    /// The pixels to save for an item: the enhanced result (or the original when only filters were used),
    /// with the detail filters applied unless they were already applied before the AI ran.
    static func finalImage(source: URL, enhanced: Bool, baked: Bool, filters: DetailFilters) -> CGImage? {
        guard let cg = loadFull(source) else { return nil }
        let needs = filters.anyOn && !(enhanced && baked)
        return needs ? render(filters, cg) : cg
    }
}

/// Thread-safe "latest request" counter so stale preview renders are skipped.
final class LatestToken: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0
    func next() -> Int { lock.lock(); defer { lock.unlock() }; value += 1; return value }
    func isCurrent(_ v: Int) -> Bool { lock.lock(); defer { lock.unlock() }; return v == value }
}

/// Target for the file-format pop-up in the Save As panel.
final class FormatPopupTarget: NSObject {
    let onChange: (Int) -> Void
    init(_ onChange: @escaping (Int) -> Void) { self.onChange = onChange }
    @objc func changed(_ sender: NSPopUpButton) { onChange(sender.indexOfSelectedItem) }
}

// MARK: - Engine process

final class LineBuffer { var data = Data() }

final class Engine {
    private var process: Process?
    private var stdin: FileHandle?
    var onMessage: (@MainActor ([String: Any]) -> Void)?
    var onExit: (@MainActor (Int32) -> Void)?

    static var logURL: URL {
        let dir = FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Logs", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("ProtoArt Beautify.log")
    }

    var isRunning: Bool { process?.isRunning ?? false }

    func start() throws {
        if isRunning { return }
        let res = Bundle.main.resourceURL!
        // Always the bundled interpreter and engine (no environment overrides in release builds).
        let python = res.appendingPathComponent("runtime/bin/python3")
        let server = res.appendingPathComponent("engine/server.py")
        guard FileManager.default.isExecutableFile(atPath: python.path) else {
            throw NSError(domain: "Engine", code: 1, userInfo: [NSLocalizedDescriptionKey:
                "Bundled Python not found at \(python.path). Rebuild the app with build.sh."])
        }

        let p = Process()
        p.executableURL = python
        p.arguments = ["-I", "-u", server.path]   // -I: ignore PYTHON* variables and user site-packages
        // Minimal, known environment: nothing inherited (no DYLD_*, PYTHON*, proxies or tokens from the shell).
        let env = ProcessInfo.processInfo.environment
        var penv: [String: String] = [
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": NSHomeDirectory(),
            "TMPDIR": NSTemporaryDirectory(),
            "LANG": "en_US.UTF-8",
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        ]
        if let user = env["USER"] { penv["USER"] = user }
        p.environment = penv

        let inPipe = Pipe(), outPipe = Pipe()
        p.standardInput = inPipe
        p.standardOutput = outPipe
        FileManager.default.createFile(atPath: Engine.logURL.path, contents: nil)
        p.standardError = try FileHandle(forWritingTo: Engine.logURL)

        let buffer = LineBuffer() // per-process; only touched on the pipe's reader thread
        outPipe.fileHandleForReading.readabilityHandler = { [weak self] h in
            let chunk = h.availableData
            guard let self, !chunk.isEmpty else { return }
            buffer.data.append(chunk)
            while let nl = buffer.data.firstIndex(of: 0x0A) {
                let line = buffer.data.subdata(in: buffer.data.startIndex..<nl)
                buffer.data.removeSubrange(buffer.data.startIndex...nl)
                if let obj = try? JSONSerialization.jsonObject(with: line) as? [String: Any] {
                    DispatchQueue.main.async { MainActor.assumeIsolated { self.onMessage?(obj) } }
                }
            }
        }
        p.terminationHandler = { [weak self] proc in
            outPipe.fileHandleForReading.readabilityHandler = nil
            let code = proc.terminationStatus
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    // Ignore exits of a process we already stopped on purpose (Cancel).
                    guard let self, self.process === proc else { return }
                    self.process = nil
                    self.stdin = nil
                    self.onExit?(code)
                }
            }
        }
        try p.run()
        process = p
        stdin = inPipe.fileHandleForWriting
    }

    func send(_ obj: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: obj) else { return }
        // throwing API: a dead engine must not crash the app (SIGPIPE is ignored at launch)
        try? stdin?.write(contentsOf: data + Data([0x0A]))
    }

    func stop() {
        guard let p = process else { return }
        if p.isRunning { p.terminate() }
        process = nil
        stdin = nil
    }
}


// MARK: - Model downloads (native, so they go out under the app's own network identity)

struct ModelFile {
    let key: String
    let repo: String
    let file: String
    let rev: String
    let fp16: Bool   // the engine converts these to fp16 once and keeps only that copy
    let mb: Int
    var optional = false  // fetched only when a feature needs it (face fix), not at first launch

    var url: URL { URL(string: "https://huggingface.co/\(repo)/resolve/\(rev)/\(file)")! }
    var rawName: String { key + "." + (file as NSString).pathExtension }
}

enum ModelStore {
    /// Same checkpoints and pinned revisions as the finegrain-image-enhancer Space.
    static let files: [ModelFile] = [
        ModelFile(key: "unet", repo: "refiners/juggernaut.reborn.sd1_5.unet", file: "model.safetensors",
                  rev: "347d14c3c782c4959cc4d1bb1e336d19f7dda4d2", fp16: true, mb: 3440),
        ModelFile(key: "clip_text_encoder", repo: "refiners/juggernaut.reborn.sd1_5.text_encoder", file: "model.safetensors",
                  rev: "744ad6a5c0437ec02ad826df9f6ede102bb27481", fp16: true, mb: 490),
        ModelFile(key: "lda", repo: "refiners/juggernaut.reborn.sd1_5.autoencoder", file: "model.safetensors",
                  rev: "3c1aae3fc3e03e4a2b7e0fa42b62ebb64f1a4c19", fp16: true, mb: 330),
        ModelFile(key: "controlnet_tile", repo: "refiners/controlnet.sd1_5.tile", file: "model.safetensors",
                  rev: "48ced6ff8bfa873a8976fa467c3629a240643387", fp16: true, mb: 1450),
        ModelFile(key: "esrgan", repo: "philz1337x/upscaler", file: "4x-UltraSharp.pth",
                  rev: "011deacac8270114eb7d2eeff4fe6fa9a837be70", fp16: false, mb: 67),
        ModelFile(key: "negative_embedding", repo: "philz1337x/embeddings", file: "JuggernautNegative-neg.pt",
                  rev: "203caa7e9cc2bc225031a4021f6ab1ded283454a", fp16: false, mb: 1),
        ModelFile(key: "lora_more_details", repo: "philz1337x/loras", file: "more_details.safetensors",
                  rev: "a3802c0280c0d00c2ab18d37454a8744c44e474e", fp16: false, mb: 10),
        ModelFile(key: "lora_sdxl_render", repo: "philz1337x/loras", file: "SDXLrender_v2.0.safetensors",
                  rev: "a3802c0280c0d00c2ab18d37454a8744c44e474e", fp16: false, mb: 10),
        // Turbo: distilled few-step add-on (Latent Consistency LoRA for SD 1.5).
        ModelFile(key: "lcm_lora", repo: "latent-consistency/lcm-lora-sdv1-5", file: "pytorch_lora_weights.safetensors",
                  rev: "cf2fced511dbe7e26c8d1d397e728fbab875db4b", fp16: false, mb: 135),
        // Face fix, downloaded the first time it's used. Byte-identical to the official GitHub releases
        // (TencentARC/GFPGAN v1.3.4 RestoreFormer.pth, sczhou/CodeFormer v0.1.0 codeformer.pth).
        ModelFile(key: "restoreformer", repo: "fofr/comfyui", file: "facerestore_models/RestoreFormer.pth",
                  rev: "173115bf2afcefcc7d6b27aedc681a5f5c9c8b0b", fp16: false, mb: 278, optional: true),
        ModelFile(key: "codeformer", repo: "fofr/comfyui", file: "facerestore_models/codeformer.pth",
                  rev: "173115bf2afcefcc7d6b27aedc681a5f5c9c8b0b", fp16: false, mb: 360, optional: true),
        // Likeness lock, downloaded the first time it's used: IP-Adapter Plus SD 1.5 and the OpenCLIP ViT-H
        // image encoder, as converted and published by refiners.
        ModelFile(key: "ip_plus", repo: "refiners/sd15.ip_adapter.plus", file: "model.safetensors",
                  rev: "2516bc6c2cbd933e7949f7c2c0f076790ff3bf35", fp16: false, mb: 94, optional: true),
        ModelFile(key: "clip_image_h", repo: "refiners/sd21.unclip.image_encoder", file: "model.safetensors",
                  rev: "4a36eec8c6fcfa9911a9b3290c707b8ad7591cc3", fp16: false, mb: 1206, optional: true),
    ]

    /// SHA-256 of every file that is kept exactly as downloaded (checked against Hugging Face's own LFS
    /// hashes). A download that doesn't match is thrown away. The big safetensors files are verified by
    /// HTTPS + pinned commit and converted to fp16 locally.
    static let pinnedSHA256: [String: String] = [
        "esrgan": "a5812231fc936b42af08a5edba784195495d303d5b3248c24489ef0c4021fe01",
        "negative_embedding": "979ea73200c7c29ba884e642e544c54abfaad4c9a82582c68f3212d390d41397",
        "lora_more_details": "d9cf2f88dec53a8dd8c09aba2b1d67963b265ebdf8255660f5bcd7f1c8edba33",
        "lora_sdxl_render": "672fcccf2ce6ca4ef71762a3ec1b8c8afd79328067e13c8f9756260ec94d1608",
        "lcm_lora": "8f90d840e075ff588a58e22c6586e2ae9a6f7922996ee6649a7f01072333afe4",
        "restoreformer": "07404d446d62ca3d5ed38b1de09a947a1e77d46dbccec961a74d713a8f24ace0",
        "codeformer": "1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7",
        "ip_plus": "6eae5d2098fa83e3b8bf416fb46dd77a921ad044650f13890e8d41d7c84a71d2",
        "clip_image_h": "c9f43e359e06f1a237324c4c11734d6acd7fbddbfd3b1ed4f1b525267bedb812",
    ]

    static func sha256(of url: URL) -> String? {
        guard let h = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? h.close() }
        var hasher = SHA256()
        while let chunk = try? h.read(upToCount: 4 << 20), !chunk.isEmpty { hasher.update(data: chunk) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static var root: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("ProtoArt Beautify", isDirectory: true)
    }
    static var downloads: URL { root.appendingPathComponent("downloads", isDirectory: true) }
    static var fp16Dir: URL { root.appendingPathComponent("models-fp16", isDirectory: true) }

    static func isPresent(_ f: ModelFile) -> Bool {
        let fm = FileManager.default
        if f.fp16, fm.fileExists(atPath: fp16Dir.appendingPathComponent("\(f.key).fp16.safetensors").path) { return true }
        return fm.fileExists(atPath: downloads.appendingPathComponent(f.rawName).path)
    }

    static var missing: [ModelFile] { files.filter { !$0.optional && !isPresent($0) } }
    static func faceModel(_ kind: String) -> ModelFile? { files.first { $0.key == kind } }
}

/// One-click CodeFormer settings (the same as Protoart Studio's FACE_PRESETS).
struct FacePreset: Identifiable {
    let id: String, label: String, fidelity: Double, strength: Double, tip: String
    static let all: [FacePreset] = [
        FacePreset(id: "balanced", label: "BALANCED", fidelity: 0.5, strength: 1.0, tip: "Good default for most portraits"),
        FacePreset(id: "identity", label: "LIKENESS", fidelity: 0.8, strength: 1.0, tip: "Keeps the likeness of a real person"),
        FacePreset(id: "heavy", label: "HEAVY", fidelity: 0.3, strength: 1.0, tip: "Fixes badly distorted or tiny faces (small faces in wide shots)"),
        FacePreset(id: "subtle", label: "SUBTLE", fidelity: 0.7, strength: 0.6, tip: "Light clean-up blended with the original, so skin texture survives"),
        FacePreset(id: "old", label: "OLD PHOTO", fidelity: 0.4, strength: 1.0, tip: "Scanned or worn photographs"),
        FacePreset(id: "gentle", label: "GENTLE", fidelity: 0.85, strength: 1.0, tip: "Faces that are already clear and just need a polish"),
    ]
}

/// Finds faces with Apple's Vision framework and returns five landmarks per face (the two eyes, the nose tip
/// and the two mouth corners, in the order the face-restore template expects) as fractions of the image size,
/// origin top-left.
enum FaceFinder {
    static func landmarks(_ cg: CGImage) -> [[[Double]]] {
        let request = VNDetectFaceLandmarksRequest()
        let handler = VNImageRequestHandler(cgImage: cg, options: [:])
        guard (try? handler.perform([request])) != nil else { return [] }
        let W = Double(cg.width), H = Double(cg.height)
        let size = CGSize(width: W, height: H)
        func pts(_ r: VNFaceLandmarkRegion2D?) -> [CGPoint] {
            (r?.pointsInImage(imageSize: size) ?? []).map { CGPoint(x: $0.x, y: H - $0.y) } // to top-left origin
        }
        func centre(_ p: [CGPoint]) -> CGPoint? {
            p.isEmpty ? nil : CGPoint(x: p.map(\.x).reduce(0, +) / CGFloat(p.count),
                                      y: p.map(\.y).reduce(0, +) / CGFloat(p.count))
        }
        var out: [[[Double]]] = []
        for face in request.results ?? [] {
            guard let lm = face.landmarks,
                  var e1 = centre(pts(lm.leftPupil)) ?? centre(pts(lm.leftEye)),
                  var e2 = centre(pts(lm.rightPupil)) ?? centre(pts(lm.rightEye)) else { continue }
            let lips = pts(lm.outerLips)
            guard let mouth = centre(lips) else { continue }
            // Order the eyes like the template: going from eye 1 to eye 2, the mouth is on the clockwise side.
            let eyeMid = CGPoint(x: (e1.x + e2.x) / 2, y: (e1.y + e2.y) / 2)
            if (e2.x - e1.x) * (mouth.y - eyeMid.y) - (e2.y - e1.y) * (mouth.x - eyeMid.x) < 0 { swap(&e1, &e2) }
            let ux = e2.x - e1.x, uy = e2.y - e1.y
            // Mouth corners: the lip points furthest along the eye line in each direction.
            func along(_ p: CGPoint) -> CGFloat { (p.x - mouth.x) * ux + (p.y - mouth.y) * uy }
            guard let m1 = lips.min(by: { along($0) < along($1) }),
                  let m2 = lips.max(by: { along($0) < along($1) }) else { continue }
            // Nose tip: the nose-crest point furthest toward the mouth, else the nose's centre.
            let dx = mouth.x - eyeMid.x, dy = mouth.y - eyeMid.y
            let crest = pts(lm.noseCrest)
            guard let nose = crest.max(by: { ($0.x * dx + $0.y * dy) < ($1.x * dx + $1.y * dy) })
                    ?? centre(pts(lm.nose)) else { continue }
            out.append([e1, e2, nose, m1, m2].map { [Double($0.x) / W, Double($0.y) / H] })
        }
        return out
    }
}

/// Downloads the missing model files one by one with byte progress, ETA, retries and resume.
final class ModelDownloader: NSObject, URLSessionDownloadDelegate {
    var onProgress: (@MainActor (_ message: String, _ fraction: Double, _ eta: Double?) -> Void)?
    var onFinish: (@MainActor (_ error: String?) -> Void)?
    private(set) var isRunning = false

    private lazy var session = URLSession(configuration: .default, delegate: self, delegateQueue: .main)
    private var queue: [ModelFile] = []
    private var current: ModelFile?
    private var index = 0
    private var count = 0
    private var totalBytes: Double = 1
    private var finishedBytes: Double = 0
    private var currentBytes: Double = 0
    private var bytesThisRun: Double = 0
    private var runStart = Date()
    private var retries = 0
    private var hashMismatch = false
    private var lastReport = Date.distantPast

    func start(_ files: [ModelFile]) {
        guard !isRunning, !files.isEmpty else { return }
        isRunning = true
        queue = files
        count = files.count
        index = 0
        totalBytes = Double(files.reduce(0) { $0 + $1.mb }) * 1_048_576
        finishedBytes = 0
        bytesThisRun = 0
        runStart = Date()
        try? FileManager.default.createDirectory(at: ModelStore.downloads, withIntermediateDirectories: true)
        next()
    }

    private func next(resumeData: Data? = nil) {
        if resumeData == nil {
            guard !queue.isEmpty else { finish(nil); return }
            current = queue.removeFirst()
            index += 1
            retries = 0
        }
        currentBytes = 0
        let task = resumeData.map { session.downloadTask(withResumeData: $0) }
            ?? session.downloadTask(with: current!.url)
        task.resume()
        report(force: true)
    }

    private func finish(_ error: String?) {
        isRunning = false
        current = nil
        let cb = onFinish
        MainActor.assumeIsolated { cb?(error) }
    }

    private func report(force: Bool = false) {
        guard let f = current else { return }
        let now = Date()
        guard force || now.timeIntervalSince(lastReport) > 0.25 else { return }
        lastReport = now
        let done = finishedBytes + currentBytes
        let frac = min(0.999, done / max(totalBytes, 1))
        let elapsed = now.timeIntervalSince(runStart)
        let speed = elapsed > 2 ? bytesThisRun / elapsed : 0
        let eta: Double? = speed > 0 ? max(0, totalBytes - done) / speed : nil
        let gb = { (b: Double) in String(format: "%.2f", b / 1_073_741_824) }
        let name = f.key.uppercased().replacingOccurrences(of: "_", with: " ")
        let msg = "Downloading \(name) — \(index)/\(count) · \(gb(done)) of \(gb(totalBytes)) GB"
        let cb = onProgress
        MainActor.assumeIsolated { cb?(msg, frac, eta) }
    }

    // MARK: URLSessionDownloadDelegate (delivered on the main queue)

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        currentBytes = Double(totalBytesWritten)
        bytesThisRun += Double(bytesWritten)
        report()
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        guard let f = current else { return }
        let status = (downloadTask.response as? HTTPURLResponse)?.statusCode ?? 0
        guard status == 200 else { return } // reported as an error in didCompleteWithError
        if let want = ModelStore.pinnedSHA256[f.key], ModelStore.sha256(of: location) != want {
            hashMismatch = true // corrupted or tampered: never keep it (it is retried, then reported)
            return
        }
        let dest = ModelStore.downloads.appendingPathComponent(f.rawName)
        try? FileManager.default.removeItem(at: dest)
        do { try FileManager.default.moveItem(at: location, to: dest) } catch { return }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let f = current else { return }
        let status = (task.response as? HTTPURLResponse)?.statusCode ?? 0
        let saved = FileManager.default.fileExists(atPath: ModelStore.downloads.appendingPathComponent(f.rawName).path)
        if error == nil, status == 200, saved {
            finishedBytes += max(currentBytes, Double(f.mb) * 1_048_576 * 0.01)
            next()
            return
        }
        retries += 1
        let why = hashMismatch ? "the downloaded file failed its security check"
            : (error?.localizedDescription ?? "HTTP \(status)")
        hashMismatch = false
        guard retries <= 5 else {
            finish("Couldn't download \(f.key) from huggingface.co (\(why)). Check your internet connection or firewall.")
            return
        }
        let resume = (error as NSError?)?.userInfo[NSURLSessionDownloadTaskResumeData] as? Data
        DispatchQueue.main.asyncAfter(deadline: .now() + Double(min(2 << retries, 30))) { [weak self] in
            self?.next(resumeData: resume)
        }
    }
}

// MARK: - App model

@MainActor
final class AppModel: ObservableObject {
    // Settings (persisted)
    @Published var settings = EnhanceSettings() { didSet { settingsChanged() } }
    @Published var batchMode: Bool { didSet { defaults.set(batchMode, forKey: "batchMode"); modeChanged() } }
    @Published var batchMaxSide: Int { didSet { defaults.set(batchMaxSide, forKey: "batchMaxSide"); settingsChanged() } }
    @Published var singleMaxSide: Int { didSet { defaults.set(singleMaxSide, forKey: "singleMaxSide"); settingsChanged() } }
    /// Default save folder (quick Save, Save As starting point and batch results).
    @Published var outputFolder: URL { didSet { defaults.set(outputFolder.path, forKey: "outputFolder") } }
    /// Save format: "png", "jpg" or "tif".
    @Published var outputFormat: String { didSet { defaults.set(outputFormat, forKey: "outputFormat") } }
    /// "copy" = save a brand-new file in the save folder; "overwrite" = replace the original photo.
    @Published var saveMode: String { didSet { defaults.set(saveMode, forKey: "saveMode") } }
    /// Monitor view: "single", "split" or "side".
    @Published var viewMode: String { didSet { defaults.set(viewMode, forKey: "viewMode") } }
    /// Save every finished result straight to the save folder (copy mode), so nothing waits in a temp folder.
    @Published var autoSaveResults: Bool { didSet { defaults.set(autoSaveResults, forKey: "autoSaveResults") } }
    /// Your own presets (Presets → MY PRESETS).
    @Published var userPresets: [UserPreset] = [] {
        didSet { if let d = try? JSONEncoder().encode(userPresets) { defaults.set(d, forKey: "userPresets") } }
    }

    // Face fix (runs after the enhance): RestoreFormer (default) or CodeFormer.
    @Published var faceFix: Bool { didSet { defaults.set(faceFix, forKey: "faceFix"); if faceFix { prefetchFaceModel() } } }
    @Published var faceModel: String { didSet { defaults.set(faceModel, forKey: "faceModel"); if faceFix { prefetchFaceModel() } } }
    @Published var faceStrength: Double { didSet { defaults.set(faceStrength, forKey: "faceStrength") } }
    @Published var faceFidelity: Double { didSet { defaults.set(faceFidelity, forKey: "faceFidelity") } }
    private var faceWaiters: [(String?) -> Void] = []

    // Detail filters (Bleach Bypass, Dark Contrasts, Detail Extractor)
    @Published var filters = DetailFilters.load() { didSet { if filters != oldValue { filters.save(); refreshFiltered() } } }
    /// "after" = applied on top of the AI result; "before" = applied to the photo, then the AI enhances that.
    @Published var filterStage: String { didSet { defaults.set(filterStage, forKey: "filterStage") } }
    @Published var filteredPreview: NSImage?
    private var previewInputCG: CGImage?
    private var previewOutputCG: CGImage?
    private var previewInputFull: CGImage?
    private var previewOutputFull: CGImage?
    private let filterToken = LatestToken()
    private let filterQueue = DispatchQueue(label: "detail-filters", qos: .userInitiated)

    // Presets: the lit key stays lit until a knob it set is moved (no flicker while turning knobs).
    @Published var activePreset: String?
    @Published var showHelp = false
    /// Filters panel beside the picture (toggled by the FILTERS key in the meter bridge).
    @Published var showFilters: Bool { didSet { defaults.set(showFilters, forKey: "showFilters") } }
    /// Adjust panel folded down to a slim tab.
    @Published var adjustCollapsed: Bool { didSet { defaults.set(adjustCollapsed, forKey: "adjustCollapsed") } }
    @Published var splitPosition: CGFloat = 0.5   // compare divider, kept while switching images/filters
    @Published var zoomScale: CGFloat?             // nil = fit; else screen points per image pixel
    @Published var zoomPan: CGSize = .zero
    private var presetSnapshot: EnhanceSettings?
    private var presetMaxSide: Int?
    private var applyingPreset = false

    // Queue
    @Published var items: [QueueItem] = []
    @Published var selectedID: QueueItem.ID? = nil { didSet { if selectedID != oldValue { loadPreview() } } }
    @Published var running = false
    @Published private(set) var runTotal = 0
    @Published private(set) var runCompleted = 0

    // Engine
    @Published var engineReady = false
    @Published var engineStatus = "Starting engine…"
    @Published var engineFraction: Double?

    // Preview of the selected item (downsampled to keep memory low)
    @Published var previewInput: NSImage?
    @Published var previewOutput: NSImage?
    @Published var errorMessage: String?

    private let defaults = UserDefaults.standard
    private let engine = Engine()
    private var warmupID: String?
    private var setupAlertShown = false
    private let downloader = ModelDownloader()
    @Published var setupETA: Double?
    private var jobID: String?
    private var jobItemID: QueueItem.ID?
    private var jobStart = Date()
    private var secondsPerMP: [String: Double] = [:] // learned per memory profile

    init() {
        let d = UserDefaults.standard
        batchMode = d.bool(forKey: "batchMode")
        batchMaxSide = d.object(forKey: "batchMaxSide") as? Int ?? 768
        singleMaxSide = d.object(forKey: "singleMaxSide") as? Int ?? 0
        outputFormat = d.string(forKey: "outputFormat") ?? "png"
        saveMode = d.string(forKey: "saveMode") ?? "copy"
        viewMode = d.string(forKey: "viewMode") ?? "split"
        filterStage = d.string(forKey: "filterStage") ?? "after"
        showFilters = d.object(forKey: "showFilters") as? Bool ?? true
        adjustCollapsed = d.bool(forKey: "adjustCollapsed")
        faceFix = d.bool(forKey: "faceFix")
        faceModel = d.string(forKey: "faceModel") ?? "codeformer"
        // 0% would skip every face (and looked like "no faces found"): the knob starts at 10%, like Studio's slider.
        faceStrength = max(0.1, d.object(forKey: "faceStrength") as? Double ?? 0.8)
        if (d.object(forKey: "faceStrength") as? Double ?? 0.8) < 0.1 { faceStrength = 0.8 }
        faceFidelity = d.object(forKey: "faceFidelity") as? Double ?? 0.5
        let pictures = FileManager.default.urls(for: .picturesDirectory, in: .userDomainMask)[0]
        // Default save folder: the Desktop, where results are easy to find (change it with the Save To tape).
        outputFolder = d.string(forKey: "outputFolder").map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask).first
            ?? pictures.appendingPathComponent("ProtoArt Beautify", isDirectory: true)
        autoSaveResults = d.object(forKey: "autoSaveResults") as? Bool ?? true
        if let data = d.data(forKey: "userPresets"), let list = try? JSONDecoder().decode([UserPreset].self, from: data) {
            userPresets = list
        }
        engine.onMessage = { [weak self] msg in self?.handle(msg) }
        engine.onExit = { [weak self] code in self?.engineExited(code) }
        #if arch(x86_64)
        // Intel Macs run the AI on the processor (no Apple GPU): start them on the fast preset.
        if !d.bool(forKey: "intelDefaultsApplied"), let turbo = Preset.all.first(where: { $0.id == "TURBO" }) {
            turbo.apply(to: self)
            d.set(true, forKey: "intelDefaultsApplied")
        }
        #endif
        syncPreset()
        AppModel.cleanTemp()
        warmup()
    }

    // MARK: Presets

    func applyPreset(_ p: Preset) {
        applyingPreset = true
        p.apply(to: self)
        applyingPreset = false
        activePreset = p.id
        presetSnapshot = settings
        presetMaxSide = maxSide
    }

    /// Light whichever preset matches the current settings (at launch and when switching batch mode).
    func syncPreset() {
        activePreset = Preset.all.first { $0.matches(self) }?.id
        presetSnapshot = settings
        presetMaxSide = maxSide
    }

    private func settingsChanged() {
        if settings.likeness > 0 && settings.mode == "detail" { prefetchOptional() }
        guard !applyingPreset, let snap = presetSnapshot else { return }
        let s = settings
        let aiMoved = s.creativity != snap.creativity || s.resemblance != snap.resemblance || s.hdr != snap.hdr
            || s.steps != snap.steps || s.solver != snap.solver || s.resemblanceDecay != snap.resemblanceDecay
        if aiMoved && s.mode == "instant" { settings.mode = "detail" } // turning an AI knob leaves Instant
        // Upscale is never part of a preset; Max Size only for presets that set it.
        let presetSetsSize = Preset.all.first { $0.id == activePreset }?.maxSide != nil
        if activePreset != nil && (aiMoved || (presetSetsSize && maxSide != presetMaxSide)) {
            activePreset = nil
        }
    }

    // MARK: Derived

    var maxSide: Int {
        get { batchMode ? batchMaxSide : singleMaxSide }
        set { if batchMode { batchMaxSide = newValue } else { singleMaxSide = newValue } }
    }
    var selected: QueueItem? { items.first { $0.id == selectedID } }

    /// Something to save: an enhanced result, or the original with detail filters on it.
    var canSave: Bool {
        guard let item = selected, item.state != .running else { return false }
        return item.outputURL != nil || filters.anyOn
    }

    /// What the monitor shows as "after".
    var displayAfter: NSImage? {
        guard let item = selected else { return nil }
        if let out = previewOutput {
            if item.filtersBaked || !filters.anyOn { return out }
            return filteredPreview ?? out
        }
        if item.outputURL != nil || item.state == .running { return nil }
        return filters.anyOn ? filteredPreview : nil
    }

    var saveType: UTType { outputFormat == "jpg" ? .jpeg : outputFormat == "tif" ? .tiff : .png }
    var saveExt: String { outputFormat == "jpg" ? "jpg" : outputFormat == "tif" ? "tif" : "png" }
    var current: QueueItem? { items.first { $0.id == jobItemID } }
    var queuedCount: Int { items.filter { $0.state == .queued }.count }
    var hasFinished: Bool { items.contains { $0.isFinished } }

    var canStart: Bool {
        if running { return false }
        if batchMode { return queuedCount > 0 }
        return selected != nil
    }

    /// Estimated seconds for an item that hasn't started, from what earlier runs took.
    func estimate(_ item: QueueItem) -> Double? {
        let s = item.job?.settings ?? settings, side = item.job?.maxSide ?? maxSide
        guard let rate = secondsPerMP[s.rateKey] else { return nil }
        let out = s.predictedOutput(for: item.inputSize, maxSide: side)
        return rate * Double(out.width * out.height) / 1_000_000
    }

    /// Time left for the whole run: current image's ETA + estimates for queued ones.
    var queueETA: Double? {
        guard running else { return nil }
        var total = 0.0
        if let cur = current {
            guard let eta = cur.eta else { return nil }
            total += eta
        }
        for item in items where item.state == .queued {
            guard let e = estimate(item) else { return nil }
            total += e
        }
        return total
    }

    // MARK: Face models (downloaded the first time Face Fix is used)

    /// Start fetching the chosen face model in the background as soon as Face Fix is switched on.
    private func prefetchFaceModel() { prefetchOptional() }

    /// Optional models the current settings need (face model, Likeness) that aren't on disk yet.
    private func missingOptional(face: Bool, faceKind: String, likeness: Double, mode: String) -> [ModelFile] {
        var keys: [String] = []
        if face { keys.append(faceKind) }
        if likeness > 0 && mode == "detail" { keys += ["ip_plus", "clip_image_h"] }
        return keys.compactMap { ModelStore.faceModel($0) }.filter { !ModelStore.isPresent($0) }
    }

    private func prefetchOptional() {
        guard engineReady else { return }
        let files = missingOptional(face: faceFix, faceKind: faceModel, likeness: settings.likeness, mode: settings.mode)
        if !files.isEmpty { ensureOptional(files) { _ in } }
    }

    private func ensureOptional(_ files: [ModelFile], _ done: @escaping (String?) -> Void) {
        let need = files.filter { !ModelStore.isPresent($0) }
        if need.isEmpty { done(nil); return }
        faceWaiters.append(done)
        guard !downloader.isRunning else { return }
        downloader.onProgress = { [weak self] msg, frac, eta in
            guard let self else { return }
            self.engineStatus = msg
            self.engineFraction = frac
            for i in self.items.indices where self.items[i].state == .queued && self.running {
                self.items[i].message = "Downloading an add-on model (first use only)…"
            }
        }
        downloader.onFinish = { [weak self] err in
            guard let self else { return }
            self.engineFraction = nil
            self.engineStatus = self.engineReady ? "Ready" : self.engineStatus
            let waiters = self.faceWaiters
            self.faceWaiters = []
            waiters.forEach { $0(err) }
        }
        downloader.start(need)
    }

    // MARK: Engine lifecycle

    func warmup() {
        // 1. Fetch any missing model files natively (byte progress + ETA, the app's own network identity).
        let missing = ModelStore.missing
        if !missing.isEmpty {
            engineReady = false
            guard !downloader.isRunning else { return }
            downloader.onProgress = { [weak self] msg, frac, eta in
                self?.engineStatus = msg
                self?.engineFraction = frac
                self?.setupETA = eta
            }
            downloader.onFinish = { [weak self] err in
                guard let self else { return }
                self.setupETA = nil
                if let err {
                    self.downloadFailed(err)
                } else {
                    self.warmup()
                }
            }
            downloader.start(missing)
            return
        }
        // 2. Everything is on disk: start the engine and load (first time it also converts to fp16).
        do {
            try engine.start()
            let id = UUID().uuidString
            warmupID = id
            engineReady = false
            engineStatus = "Preparing models (first launch downloads ~6 GB)…"
            engine.send(["id": id, "cmd": "warmup"])
        } catch {
            engineStatus = "Engine unavailable"
            errorMessage = error.localizedDescription
        }
    }

    /// Keep retrying quietly after the first alert (network blips, firewalls being sorted out).
    private func downloadFailed(_ message: String) {
        engineStatus = "Can't get models — retrying in 15 s"
        engineFraction = nil
        if !setupAlertShown {
            setupAlertShown = true
            errorMessage = message
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 15) { [weak self] in
            guard let self, !self.engineReady else { return }
            self.warmup()
        }
    }

    private func engineExited(_ code: Int32) {
        engineReady = false
        engineFraction = nil
        if let idx = index(of: jobItemID) {
            items[idx].state = .failed("Engine stopped (code \(code)) — likely out of memory. Try a smaller max size or the 8 GB profile.")
            items[idx].fraction = nil
            finishItem()
        }
        engineStatus = "Engine stopped — restarting…"
        warmup()
    }

    // MARK: Adding / removing

    func openPanel() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image, .folder]
        panel.canChooseDirectories = batchMode
        panel.allowsMultipleSelection = batchMode
        panel.message = batchMode ? "Choose images or folders to add to the queue" : "Choose an image"
        if panel.runModal() == .OK { add(panel.urls) }
    }

    func add(_ urls: [URL]) {
        var found = ImageIOHelpers.expand(urls)
        guard !found.isEmpty else {
            errorMessage = "No images found in what you dropped."
            return
        }
        if !batchMode {
            // Single mode: the new image replaces everything that isn't running.
            found = [found[0]]
            items.removeAll { $0.state != .running }
        }
        let existing = Set(items.filter { $0.state == .queued }.map { $0.url })
        let fresh = found.filter { !existing.contains($0) }.map { QueueItem(url: $0) }
        items += fresh
        if running { runTotal += fresh.count }
        if selectedID == nil || !batchMode { selectedID = fresh.first?.id ?? selectedID }
        for item in fresh { loadMetadata(item.id, url: item.url) }
        updateDockBadge()
        pump()
    }

    private func loadMetadata(_ id: QueueItem.ID, url: URL) {
        DispatchQueue.global(qos: .userInitiated).async {
            let size = ImageIOHelpers.pixelSize(url)
            let thumb = ImageIOHelpers.image(url, maxPixel: 120)
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    guard let i = self.index(of: id) else { return }
                    self.items[i].inputSize = size
                    self.items[i].thumbnail = thumb
                }
            }
        }
    }

    func remove(_ ids: Set<QueueItem.ID>) {
        let removable = ids.filter { $0 != jobItemID }
        let wasQueued = items.filter { removable.contains($0.id) && $0.state == .queued }.count
        items.removeAll { removable.contains($0.id) }
        if running { runTotal = max(runCompleted + (jobItemID == nil ? 0 : 1), runTotal - wasQueued) }
        if let s = selectedID, removable.contains(s) { selectedID = items.first?.id }
        updateDockBadge()
    }

    func clearFinished() {
        remove(Set(items.filter { $0.isFinished }.map { $0.id }))
    }

    func requeue(_ id: QueueItem.ID) {
        guard let i = index(of: id), items[i].state != .running else { return }
        items[i].state = .queued
        items[i].fraction = nil
        items[i].eta = nil
        items[i].message = ""
        if running { runTotal += 1 }
        updateDockBadge()
        pump()
    }

    // MARK: Running

    func start() {
        guard !running else { return }
        if !batchMode, let s = selectedID, let i = index(of: s), items[i].state != .queued {
            items[i].state = .queued // single mode: Enhance again with the current settings
        }
        runTotal = queuedCount
        runCompleted = 0
        guard runTotal > 0 else { return }
        if batchMode && saveMode == "overwrite" && !confirmOverwrite(count: runTotal, name: nil) { return }
        running = true
        if !engine.isRunning { warmup() }
        updateDockBadge()
        pump()
    }

    /// Cancel one image. Queued: skipped. Running: aborted, and the queue moves on.
    func cancel(_ id: QueueItem.ID) {
        guard let i = index(of: id) else { return }
        switch items[i].state {
        case .queued:
            items[i].state = .cancelled
            items[i].message = "Cancelled"
            if running { runTotal = max(runCompleted + (jobItemID == nil ? 0 : 1), runTotal - 1) }
            updateDockBadge()
        case .running:
            guard batchMode, queuedCount > 0 else {
                stop()
                if let j = index(of: id) { items[j].state = .cancelled; items[j].message = "Cancelled" }
                return
            }
            engine.stop() // the only way to interrupt the GPU work; models reload, then the queue resumes
            items[i].state = .cancelled
            items[i].fraction = nil
            items[i].eta = nil
            items[i].message = "Cancelled"
            jobID = nil
            jobItemID = nil
            runCompleted += 1
            updateDockBadge()
            warmup() // pump() runs again once the engine reports ready
        default:
            break
        }
    }

    func stop() {
        guard running else { return }
        running = false
        engine.stop() // models reload in the background
        if let i = index(of: jobItemID) {
            items[i].state = .stopped
            items[i].fraction = nil
            items[i].eta = nil
            items[i].message = "Stopped"
        }
        jobID = nil
        jobItemID = nil
        updateDockBadge()
        warmup()
    }

    /// Starts the next queued item if the engine is free.
    private func pump() {
        guard running, engineReady, jobID == nil else { return }
        guard let i = items.firstIndex(where: { $0.state == .queued }) else {
            let processed = runCompleted
            running = false
            updateDockBadge()
            if batchMode && processed > 1 { NSSound(named: "Glass")?.play() }
            return
        }
        let item = items[i]
        let job = item.job ?? snapshot(label: "")
        let needed = missingOptional(face: job.faceFix, faceKind: job.faceModel, likeness: job.settings.likeness,
                                     mode: job.settings.mode)
        if !needed.isEmpty {
            let itemID = item.id
            items[i].message = "Downloading an add-on model (first use only)…"
            ensureOptional(needed) { [weak self] err in
                guard let self else { return }
                if let err {
                    if let j = self.index(of: itemID) {
                        self.items[j].state = .failed("Couldn't download an add-on model: " + err)
                        self.items[j].message = ""
                    }
                    self.errorMessage = "Couldn't download an add-on model (Face Fix or Likeness): " + err
                }
                self.pump()
            }
            return
        }
        let out = outputPath(for: item)
        let id = UUID().uuidString
        jobID = id
        jobItemID = item.id
        items[i].state = .running
        items[i].fraction = 0
        items[i].eta = estimate(item)
        jobStart = Date()
        items[i].message = "Starting…"
        items[i].outputURL = nil
        items[i].savedURL = nil
        var params = job.settings.params(maxSide: job.maxSide)
        let face: [String: Any]? = job.faceFix
            ? ["model": job.faceModel, "strength": job.faceStrength, "fidelity": job.faceFidelity] : nil
        let detail: [String: Any]? = job.settings.faceDetailActive
            ? ["strength": job.settings.faceDetailStrength, "hair": job.settings.faceDetailHair] : nil
        let preFilter = filterStage == "before" && filters.anyOn
        items[i].filtersBaked = preFilter
        items[i].message = preFilter ? "Applying filters…" : "Preparing image…"
        // Security: the user's file is decoded only by macOS (ImageIO, kept patched by system updates) and
        // handed to the engine as a fresh sRGB PNG, so the bundled Python image libraries never parse
        // untrusted files. ≤3072 px is plenty: the engine works at ≤768 px on the short side.
        let f = filters, src = item.url, itemID = item.id
        DispatchQueue.global(qos: .userInitiated).async {
            var input: String?
            var faces: [[[Double]]] = []
            if let cg = ImageIOHelpers.cgImage(src, maxPixel: 3072),
               let clean = preFilter ? DetailFilterEngine.render(f, cg) : DetailFilterEngine.toSRGB(cg) {
                if face != nil || detail != nil { faces = FaceFinder.landmarks(clean) }
                let tmp = Self.tempDir.appendingPathComponent("input-\(UUID().uuidString.prefix(8)).png")
                if (try? DetailFilterEngine.write(clean, to: tmp, type: .png)) != nil { input = tmp.path }
            }
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    guard self.jobID == id else { return } // stopped or cancelled meanwhile
                    guard let input else {
                        if let j = self.index(of: itemID) {
                            self.items[j].state = .failed("Couldn't read this image (unsupported or damaged file).")
                        }
                        self.finishItem()
                        return
                    }
                    if var detail, !faces.isEmpty {
                        detail["faces"] = faces
                        params["face_detail"] = detail
                    }
                    if var face {
                        face["faces"] = faces
                        params["face_fix"] = face
                    }
                    if face != nil || detail != nil, let j = self.index(of: itemID) {
                        self.items[j].message = faces.isEmpty ? "No face found" : "Found \(faces.count) face\(faces.count == 1 ? "" : "s")"
                    }
                    self.engine.send(["id": id, "cmd": "enhance", "input": input, "output": out.path, "params": params])
                }
            }
        }
    }

    /// Remove temporary files (decoded inputs and unsaved results) from earlier sessions.
    nonisolated static func cleanTemp() {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("ProtoArtBeautify", isDirectory: true)
        try? FileManager.default.removeItem(at: dir)
    }

    nonisolated static var tempDir: URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("ProtoArtBeautify", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// The engine always writes a lossless PNG to a temp file; saving (copy/overwrite, any format) happens after.
    private func outputPath(for item: QueueItem) -> URL {
        let base = item.url.deletingPathExtension().lastPathComponent + "-beautify"
        return Self.tempDir.appendingPathComponent("\(base)-\(UUID().uuidString.prefix(6)).png")
    }

    private func finishItem() {
        jobID = nil
        jobItemID = nil
        runCompleted += 1
        if !batchMode { running = false } // single mode: one image per click
        updateDockBadge()
        pump()
    }

    // MARK: Engine messages

    private func handle(_ msg: [String: Any]) {
        let event = msg["event"] as? String
        let id = msg["id"] as? String

        if id != nil, id == warmupID {
            switch event {
            case "status":
                engineStatus = msg["message"] as? String ?? engineStatus
                engineFraction = msg["fraction"] as? Double
            case "done":
                engineReady = true
                engineFraction = nil
                engineStatus = "Ready"
                pump()
            case "error":
                downloadFailed(msg["message"] as? String ?? "Model setup failed")
            default: break
            }
            return
        }

        guard id != nil, id == jobID, let i = index(of: jobItemID) else {
            if event == "error", id == nil { errorMessage = msg["message"] as? String }
            return
        }
        switch event {
        case "status":
            items[i].message = msg["message"] as? String ?? items[i].message
            if let f = msg["fraction"] as? Double { items[i].fraction = f }
            if let e = msg["eta"] as? Double {
                items[i].eta = e
            } else if let f = items[i].fraction, f >= 0.05 {
                // Engine has no estimate yet: extrapolate from how fast the bar is moving.
                let elapsed = Date().timeIntervalSince(jobStart)
                items[i].eta = elapsed * (1 - f) / f
            }
        case "done":
            let path = msg["output"] as? String ?? ""
            let w = msg["width"] as? Double ?? 0, h = msg["height"] as? Double ?? 0
            let secs = msg["seconds"] as? Double ?? 0
            items[i].state = .done
            items[i].fraction = nil
            items[i].eta = nil
            items[i].outputURL = URL(fileURLWithPath: path)
            items[i].outputSize = CGSize(width: w, height: h)
            items[i].seconds = secs
            items[i].message = "Done in \(Fmt.duration(secs))"
            items[i].facesFixed = nil
            if let n = (msg["faces_detailed"] as? NSNumber)?.intValue, n > 0 {
                items[i].message += " · \(n) face\(n == 1 ? "" : "s") detailed"
            }
            if (items[i].job?.faceFix ?? faceFix), let n = (msg["faces_fixed"] as? NSNumber)?.intValue {
                items[i].facesFixed = n
                items[i].message += n == 0 ? " · no faces found" : " · \(n) face\(n == 1 ? "" : "s") fixed"
            }
            let mp = w * h / 1_000_000
            if mp > 0, secs > 0 {
                let r = secs / mp, old = secondsPerMP[settings.rateKey]
                secondsPerMP[settings.rateKey] = old.map { 0.5 * $0 + 0.5 * r } ?? r
            }
            if items[i].id == selectedID { loadPreview() }
            if batchMode || (autoSaveResults && saveMode == "copy") { autoSave(items[i].id) }
            finishItem()
        case "error":
            items[i].state = .failed(msg["message"] as? String ?? "Unknown error")
            items[i].fraction = nil
            items[i].eta = nil
            finishItem()
        default: break
        }
    }

    // MARK: Preview / saving

    private func loadPreview() {
        previewInput = nil
        previewOutput = nil
        previewInputCG = nil
        previewOutputCG = nil
        previewInputFull = nil
        previewOutputFull = nil
        filteredPreview = nil
        zoomScale = nil
        zoomPan = .zero
        guard let item = selected else { return }
        let inURL = item.url, outURL = item.outputURL, id = item.id
        DispatchQueue.global(qos: .userInitiated).async {
            // Up to 4096 px so zooming to 100% shows real detail (enhanced results are usually smaller,
            // i.e. shown at full resolution).
            let af = ImageIOHelpers.cgImage(inURL, maxPixel: 4096)
            let bf = outURL.flatMap { ImageIOHelpers.cgImage($0, maxPixel: 4096) }
            // Smaller copies for the live filter preview (keeps knob turns smooth).
            let ac = ImageIOHelpers.cgImage(inURL, maxPixel: 1600)
            let bc = outURL.flatMap { ImageIOHelpers.cgImage($0, maxPixel: 1600) }
            func ns(_ cg: CGImage?) -> NSImage? { cg.map { NSImage(cgImage: $0, size: CGSize(width: $0.width, height: $0.height)) } }
            let a = ns(af), b = ns(bf)
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    guard self.selectedID == id else { return }
                    self.previewInput = a
                    self.previewOutput = b
                    self.previewInputCG = ac
                    self.previewOutputCG = bc
                    self.previewInputFull = af
                    self.previewOutputFull = bf
                    self.refreshFiltered()
                }
            }
        }
    }

    /// Re-render the filtered "after" preview in the background; only the newest request is kept.
    func refreshFiltered() {
        let token = filterToken.next()
        let f = filters
        var source: CGImage?, full: CGImage?
        if let item = selected {
            if item.outputURL != nil {
                source = item.filtersBaked ? nil : previewOutputCG
                full = item.filtersBaked ? nil : previewOutputFull
            } else {
                source = previewInputCG
                full = previewInputFull
            }
        }
        guard f.anyOn, let src = source else { filteredPreview = nil; return }
        let tokenBox = filterToken
        // The size the preview is shown at (points = pixels) must stay the same for zoom to hold steady.
        let shownSize = full.map { CGSize(width: $0.width, height: $0.height) } ?? CGSize(width: src.width, height: src.height)
        func deliver(_ out: CGImage?) {
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    guard tokenBox.isCurrent(token), let out else { return }
                    self.filteredPreview = NSImage(cgImage: out, size: shownSize)
                }
            }
        }
        // Quick pass at 1600 px while knobs move; a full-resolution pass once they rest (for zooming in).
        filterQueue.async {
            guard tokenBox.isCurrent(token) else { return }
            deliver(DetailFilterEngine.render(f, src))
        }
        if let full, full.width > src.width {
            filterQueue.asyncAfter(deadline: .now() + 0.35) {
                guard tokenBox.isCurrent(token) else { return }
                deliver(DetailFilterEngine.render(f, full))
            }
        }
    }

    func chooseOutputFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.directoryURL = outputFolder
        panel.prompt = "Use Folder"
        panel.message = "Choose the default folder for saved images"
        if panel.runModal() == .OK, let url = panel.url { outputFolder = url }
    }

    private func uniqueURL(in folder: URL, base: String, ext: String) -> URL {
        var url = folder.appendingPathComponent("\(base).\(ext)")
        var n = 2
        while FileManager.default.fileExists(atPath: url.path) {
            url = folder.appendingPathComponent("\(base)-\(n).\(ext)")
            n += 1
        }
        return url
    }

    private func copyURL(for item: QueueItem) -> URL {
        try? FileManager.default.createDirectory(at: outputFolder, withIntermediateDirectories: true)
        var base = item.url.deletingPathExtension().lastPathComponent + "-beautify"
        if let label = item.job?.label, !label.isEmpty {
            let slug = label.lowercased().map { $0.isLetter || $0.isNumber ? String($0) : "-" }.joined()
                .split(separator: "-").joined(separator: "-")
            if !slug.isEmpty { base += "-" + String(slug.prefix(40)) }
        }
        return uniqueURL(in: outputFolder, base: base, ext: saveExt)
    }

    /// Ask before replacing originals (can be turned off with "Don't ask again").
    private func confirmOverwrite(count: Int, name: String?) -> Bool {
        if defaults.bool(forKey: "skipOverwriteConfirm") { return true }
        let alert = NSAlert()
        alert.messageText = count == 1 ? "Replace the original photo?" : "Replace \(count) original photos?"
        alert.informativeText = (name.map { "“\($0)” " } ?? (count == 1 ? "The original " : "Each original "))
            + "will be replaced by the enhanced version, keeping its name and file format. This can't be undone."
        alert.alertStyle = .warning
        alert.addButton(withTitle: "Replace")
        alert.addButton(withTitle: "Cancel")
        alert.showsSuppressionButton = true
        alert.suppressionButton?.title = "Don't ask again"
        let ok = alert.runModal() == .alertFirstButtonReturn
        if ok, alert.suppressionButton?.state == .on { defaults.set(true, forKey: "skipOverwriteConfirm") }
        return ok
    }

    /// Render (filters included) and write in the background.
    private func export(_ id: QueueItem.ID, to dest: URL, type: UTType, replacing: Bool) {
        guard let item = items.first(where: { $0.id == id }) else { return }
        let f = filters
        let source = item.outputURL ?? item.url
        let enhanced = item.outputURL != nil, baked = item.filtersBaked
        if let i = index(of: id) { items[i].message = "Saving…" }
        DispatchQueue.global(qos: .userInitiated).async {
            var failure: String?
            if let cg = DetailFilterEngine.finalImage(source: source, enhanced: enhanced, baked: baked, filters: f) {
                do {
                    if replacing {
                        let tmp = dest.deletingLastPathComponent()
                            .appendingPathComponent(".beautify-\(UUID().uuidString.prefix(8)).\(dest.pathExtension)")
                        try DetailFilterEngine.write(cg, to: tmp, type: type)
                        _ = try FileManager.default.replaceItemAt(dest, withItemAt: tmp)
                    } else {
                        try DetailFilterEngine.write(cg, to: dest, type: type)
                    }
                } catch {
                    failure = error.localizedDescription
                }
            } else {
                failure = "Couldn't read the image."
            }
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    guard let i = self.index(of: id) else { return }
                    if let failure {
                        self.errorMessage = "Couldn't save \(dest.lastPathComponent): \(failure)"
                        self.items[i].message = "Save failed"
                    } else {
                        self.items[i].savedURL = dest
                        self.items[i].message = (replacing ? "Replaced original" : "Saved") + " · " + dest.lastPathComponent
                        if replacing && dest == self.items[i].url { self.loadMetadata(id, url: dest) }
                    }
                }
            }
        }
    }

    private func overwriteType(for url: URL) -> UTType? {
        guard let t = UTType(filenameExtension: url.pathExtension), DetailFilterEngine.canWrite(t) else { return nil }
        return t
    }

    /// Batch: results save themselves as soon as each image finishes.
    private func autoSave(_ id: QueueItem.ID) {
        guard let item = items.first(where: { $0.id == id }) else { return }
        if saveMode == "overwrite" {
            guard let t = overwriteType(for: item.url) else {
                export(id, to: copyURL(for: item), type: saveType, replacing: false) // e.g. RAW: save a copy instead
                return
            }
            export(id, to: item.url, type: t, replacing: true)
        } else {
            export(id, to: copyURL(for: item), type: saveType, replacing: false)
        }
    }

    /// ⌘S: save with the Save module's settings (new file in the save folder, or replace the original).
    func quickSave() {
        guard canSave, let item = selected else { return }
        if saveMode == "overwrite" {
            guard let t = overwriteType(for: item.url) else {
                errorMessage = "Can't overwrite .\(item.url.pathExtension) files. Switch Save to COPY to save a new file."
                return
            }
            guard confirmOverwrite(count: 1, name: item.name) else { return }
            export(item.id, to: item.url, type: t, replacing: true)
        } else {
            export(item.id, to: copyURL(for: item), type: saveType, replacing: false)
        }
    }

    /// ⇧⌘S: choose the name, folder and format.
    func saveSelectedAs() {
        guard canSave, let item = selected else { return }
        let base = item.url.deletingPathExtension().lastPathComponent + "-beautify"
        let formats: [(String, UTType, String)] = [("PNG", .png, "png"), ("JPEG", .jpeg, "jpg"), ("TIFF", .tiff, "tif")]
        var chosen = formats.firstIndex { $0.2 == saveExt } ?? 0
        let panel = NSSavePanel()
        panel.canCreateDirectories = true
        panel.directoryURL = outputFolder
        panel.allowedContentTypes = [formats[chosen].1]
        panel.nameFieldStringValue = "\(base).\(formats[chosen].2)"
        let popup = NSPopUpButton(frame: .zero, pullsDown: false)
        popup.addItems(withTitles: formats.map { $0.0 })
        popup.selectItem(at: chosen)
        let target = FormatPopupTarget { i in
            chosen = i
            let name = (panel.nameFieldStringValue as NSString).deletingPathExtension
            panel.allowedContentTypes = [formats[i].1]
            panel.nameFieldStringValue = "\(name).\(formats[i].2)"
        }
        popup.target = target
        popup.action = #selector(FormatPopupTarget.changed(_:))
        let label = NSTextField(labelWithString: "Format:")
        let row = NSStackView(views: [label, popup])
        row.orientation = .horizontal
        row.edgeInsets = NSEdgeInsets(top: 8, left: 12, bottom: 8, right: 12)
        panel.accessoryView = row
        let ok = panel.runModal() == .OK
        withExtendedLifetime(target) {}
        guard ok, let dest = panel.url else { return }
        let replacing = dest.standardizedFileURL == item.url.standardizedFileURL
        if replacing && !confirmOverwrite(count: 1, name: item.name) { return }
        export(item.id, to: dest, type: formats[chosen].1, replacing: replacing || FileManager.default.fileExists(atPath: dest.path))
    }

    func reveal(_ id: QueueItem.ID? = nil) {
        let item = id.flatMap { i in items.first { $0.id == i } } ?? selected
        if let url = item?.savedURL ?? item?.outputURL { NSWorkspace.shared.activateFileViewerSelecting([url]) }
    }

    func openOutputFolder() {
        try? FileManager.default.createDirectory(at: outputFolder, withIntermediateDirectories: true)
        NSWorkspace.shared.open(outputFolder)
    }

    // MARK: Trying settings (+ QUEUE) and your own presets

    /// What the current knobs would run, frozen.
    func snapshot(label: String) -> JobSnapshot {
        JobSnapshot(settings: settings, maxSide: maxSide, faceFix: faceFix, faceModel: faceModel,
                    faceStrength: faceStrength, faceFidelity: faceFidelity, label: label)
    }

    /// Short name for the current settings: the lit preset, else a summary of the knobs.
    var settingsLabel: String {
        if let u = userPresets.first(where: { userPresetMatches($0) }) { return u.name }
        if let p = activePreset {
            return p + (settings.likeness > 0 ? " +likeness" : "") + (settings.faceDetailActive ? " +detail" : "")
                + (faceFix ? " +face" : "") + (settings.keepTones ? "" : " -tones")
        }
        let s = settings
        if s.mode == "instant" { return "INSTANT" }
        var parts = [String(format: "C%.2f R%.2f", s.creativity, s.resemblance)]
        if s.turbo { parts.append("turbo") }
        if s.likeness > 0 { parts.append(String(format: "L%.0f%%", s.likeness * 100)) }
        if s.faceDetailActive { parts.append(String(format: "FD%.0f%%", s.faceDetailStrength * 100)) }
        if faceFix { parts.append(faceModel == "codeformer" ? "CF" : "RF") }
        if !s.keepTones { parts.append("raw-tones") }
        return parts.joined(separator: " ")
    }

    /// + QUEUE: adds the selected photo again with the current settings, so you can queue several versions
    /// (change knobs, + QUEUE, change again, + QUEUE …) and run them all with ENHANCE.
    func queueVariation() {
        guard let src = selected ?? items.last else { errorMessage = "Open a photo first."; return }
        if !batchMode { batchMode = true }
        // The photo you opened is waiting to run with the live knobs: the first + QUEUE gives it these settings
        // instead of adding a second, unlabelled run.
        if src.state == .queued, src.job == nil, let i = index(of: src.id) {
            items[i].job = snapshot(label: settingsLabel)
            updateDockBadge()
            return
        }
        var row = QueueItem(url: src.url)
        row.thumbnail = src.thumbnail
        row.inputSize = src.inputSize
        row.job = snapshot(label: settingsLabel)
        items.append(row)
        if running { runTotal += 1 }
        if src.inputSize == .zero { loadMetadata(row.id, url: row.url) }
        updateDockBadge()
        pump()
    }

    /// Puts a queued row's settings back on the knobs.
    func useSettings(of id: QueueItem.ID) {
        guard let job = items.first(where: { $0.id == id })?.job else { return }
        let profile = settings.profile
        settings = job.settings
        settings.profile = profile
        maxSide = job.maxSide
        faceFix = job.faceFix; faceModel = job.faceModel
        faceStrength = job.faceStrength; faceFidelity = job.faceFidelity
        syncPreset()
    }

    func userPresetMatches(_ u: UserPreset) -> Bool {
        var a = u.settings, b = settings
        a.profile = ""; b.profile = ""
        guard a == b, u.maxSide == maxSide, u.faceFix == faceFix else { return false }
        if faceFix, u.faceModel != faceModel || abs(u.faceStrength - faceStrength) > 0.001
            || abs(u.faceFidelity - faceFidelity) > 0.001 { return false }
        if let f = u.filters, f != filters { return false }
        return true
    }

    func applyUserPreset(_ u: UserPreset) {
        let profile = settings.profile
        applyingPreset = true
        settings = u.settings
        settings.profile = profile  // the Memory switch belongs to this Mac, not to the preset
        maxSide = u.maxSide
        faceFix = u.faceFix; faceModel = u.faceModel
        faceStrength = u.faceStrength; faceFidelity = u.faceFidelity
        if let f = u.filters { filters = f }
        if let st = u.filterStage { filterStage = st }
        applyingPreset = false
        syncPreset()
    }

    private func makeUserPreset(name: String, includeFilters: Bool, id: UUID = UUID()) -> UserPreset {
        UserPreset(id: id, name: name, settings: settings, maxSide: maxSide, faceFix: faceFix, faceModel: faceModel,
                   faceStrength: faceStrength, faceFidelity: faceFidelity,
                   filters: includeFilters ? filters : nil, filterStage: includeFilters ? filterStage : nil)
    }

    /// Asks for a name (and whether to include Adjust & filters), then saves the current settings.
    func saveUserPreset() {
        let alert = NSAlert()
        alert.messageText = "Save these settings as a preset"
        alert.informativeText = "Keeps the AI knobs, Upscale, Max Size and Face Fix. The Memory switch isn't saved."
        let field = NSTextField(frame: NSRect(x: 0, y: 26, width: 260, height: 24))
        field.placeholderString = "Preset name"
        field.stringValue = activePreset.map { "My " + $0.capitalized } ?? "My preset \(userPresets.count + 1)"
        let check = NSButton(checkboxWithTitle: "Include Adjust & filters", target: nil, action: nil)
        check.frame = NSRect(x: 0, y: 0, width: 260, height: 20)
        check.state = filters.anyOn ? .on : .off
        let box = NSView(frame: NSRect(x: 0, y: 0, width: 260, height: 52))
        box.addSubview(field); box.addSubview(check)
        alert.accessoryView = box
        alert.addButton(withTitle: "Save")
        alert.addButton(withTitle: "Cancel")
        alert.window.initialFirstResponder = field
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        let name = field.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return }
        let include = check.state == .on
        if let i = userPresets.firstIndex(where: { $0.name.caseInsensitiveCompare(name) == .orderedSame }) {
            userPresets[i] = makeUserPreset(name: userPresets[i].name, includeFilters: include, id: userPresets[i].id)
        } else {
            userPresets.append(makeUserPreset(name: name, includeFilters: include))
        }
    }

    func updateUserPreset(_ u: UserPreset) {
        guard let i = userPresets.firstIndex(where: { $0.id == u.id }) else { return }
        userPresets[i] = makeUserPreset(name: u.name, includeFilters: u.filters != nil, id: u.id)
    }

    func renameUserPreset(_ u: UserPreset) {
        guard let i = userPresets.firstIndex(where: { $0.id == u.id }) else { return }
        let alert = NSAlert()
        alert.messageText = "Rename preset"
        let field = NSTextField(frame: NSRect(x: 0, y: 0, width: 260, height: 24))
        field.stringValue = u.name
        alert.accessoryView = field
        alert.addButton(withTitle: "Rename")
        alert.addButton(withTitle: "Cancel")
        alert.window.initialFirstResponder = field
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        let name = field.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        if !name.isEmpty { userPresets[i].name = name }
    }

    func deleteUserPreset(_ u: UserPreset) {
        let alert = NSAlert()
        alert.messageText = "Delete the preset “\(u.name)”?"
        alert.addButton(withTitle: "Delete")
        alert.addButton(withTitle: "Cancel")
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        userPresets.removeAll { $0.id == u.id }
    }

    // MARK: Helpers

    private func modeChanged() {
        if !batchMode {
            // Leaving batch mode: keep only the selected (or running) image.
            let keep = jobItemID ?? selectedID ?? items.last?.id
            if !running { items.removeAll { $0.id != keep } }
            selectedID = keep
        }
        syncPreset()
        updateDockBadge()
    }

    private func index(of id: QueueItem.ID?) -> Int? {
        guard let id else { return nil }
        return items.firstIndex { $0.id == id }
    }

    private func updateDockBadge() {
        let left = items.filter { $0.state == .queued || $0.state == .running }.count
        NSApp?.dockTile.badgeLabel = running && left > 0 ? "\(left)" : nil
    }
}

// MARK: - Views

/// Reports scroll-wheel and trackpad scrolls over its area (without blocking clicks or drags underneath).
struct ScrollWheelCatcher: NSViewRepresentable {
    /// delta > 0 = zoom in; precise = trackpad / Magic Mouse; point = pointer in this view, origin top-left.
    var onScroll: (CGFloat, Bool, CGPoint) -> Void

    func makeNSView(context: Context) -> CatcherView { CatcherView() }
    func updateNSView(_ v: CatcherView, context: Context) { v.onScroll = onScroll }

    final class CatcherView: NSView {
        var onScroll: ((CGFloat, Bool, CGPoint) -> Void)?
        private var monitor: Any?
        override var isFlipped: Bool { true }
        override func hitTest(_ point: NSPoint) -> NSView? { nil }   // clicks and drags pass through

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            if let m = monitor { NSEvent.removeMonitor(m); monitor = nil }
            guard window != nil else { return }
            monitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { [weak self] event in
                guard let self, let win = self.window, event.window === win, !self.isHiddenOrHasHiddenAncestor
                else { return event }
                let p = self.convert(event.locationInWindow, from: nil)
                guard self.bounds.contains(p) else { return event }
                var dy = event.scrollingDeltaY != 0 ? event.scrollingDeltaY : event.scrollingDeltaX
                if event.isDirectionInvertedFromDevice { dy = -dy }  // same feel with or without natural scrolling
                guard dy != 0 else { return nil }
                self.onScroll?(dy, event.hasPreciseScrollingDeltas, p)
                return nil
            }
        }

        deinit { if let m = monitor { NSEvent.removeMonitor(m) } }
    }
}

/// Before/after viewer with zoom. All three views share one zoom level and position, so switching views
/// keeps you on the same detail. Zoom: the scroll wheel (toward the pointer), the − / FIT / 100% / 200% / +
/// keys, or pinch on a trackpad; drag to
/// look around when zoomed in. "100%" shows one image pixel per screen pixel.
struct CompareView: View {
    let before: NSImage
    let after: NSImage
    var mode = "split"   // "single" (click to flip Before/After), "split" (drag the divider), "side" (side by side)
    @Binding var split: CGFloat
    @Binding var zoom: CGFloat?   // screen points per image pixel; nil = fit to the screen
    @Binding var pan: CGSize      // offset of the picture's centre while zoomed
    @State private var holding = false
    @State private var dragMode = 0          // 0 undecided, 1 divider, 2 pan
    @State private var panStart: CGSize = .zero
    @State private var pinchStart: CGFloat?

    private var backing: CGFloat { NSScreen.main?.backingScaleFactor ?? 2 }
    private var oneToOne: CGFloat { 1 / backing }   // points per image pixel at 100%

    var body: some View {
        GeometryReader { geo in
            let box = mode == "side" ? CGSize(width: (geo.size.width - 10) / 2, height: geo.size.height) : geo.size
            let fs = fitScale(in: box)
            let scale = max(fs, zoom ?? fs)
            ZStack(alignment: .bottomTrailing) {
                Group {
                    switch mode {
                    case "single": single(geo.size, scale: scale)
                    case "side": sideBySide(geo.size, box: box, scale: scale)
                    default: splitView(geo.size, scale: scale)
                    }
                }
                .overlay(ScrollWheelCatcher { delta, precise, point in
                    // Smooth zoom toward the pointer: ~10% per wheel notch, finer for trackpads.
                    let factor = exp(delta * (precise ? 0.01 : 0.1))
                    zoomAt(scale * factor, point: point, size: geo.size, box: box, scale: scale, fit: fs)
                })
                .simultaneousGesture(MagnificationGesture()
                    .onChanged { m in
                        let base = pinchStart ?? scale
                        if pinchStart == nil { pinchStart = scale }
                        setZoom(base * m, fit: fs)
                    }
                    .onEnded { _ in pinchStart = nil })
                zoomBar(scale: scale, fit: fs, box: box)
                    .padding(10)
            }
            .onChange(of: geo.size) { _ in pan = clamped(pan, scale: scale, box: box) }
        }
    }

    // MARK: Geometry

    private func fitScale(in box: CGSize) -> CGFloat {
        guard after.size.width > 0, after.size.height > 0 else { return 1 }
        return min(box.width / after.size.width, box.height / after.size.height)
    }

    /// Where the picture sits inside a box of `box` size at `scale` (points per image pixel).
    private func imageRect(_ box: CGSize, scale: CGFloat) -> CGRect {
        let w = after.size.width * scale, h = after.size.height * scale
        let p = clamped(pan, scale: scale, box: box)
        return CGRect(x: (box.width - w) / 2 + p.width, y: (box.height - h) / 2 + p.height, width: w, height: h)
    }

    private func clamped(_ p: CGSize, scale: CGFloat, box: CGSize) -> CGSize {
        let w = after.size.width * scale, h = after.size.height * scale
        let mx = max(0, (w - box.width) / 2), my = max(0, (h - box.height) / 2)
        return CGSize(width: min(mx, max(-mx, p.width)), height: min(my, max(-my, p.height)))
    }

    /// Zoom so the picture point under `point` (in this view's coordinates) stays under the pointer.
    private func zoomAt(_ z: CGFloat, point: CGPoint, size: CGSize, box: CGSize, scale: CGFloat, fit: CGFloat) {
        // In SIDE view both panels show the same picture: work in the panel the pointer is over.
        var p = point
        if mode == "side", p.x > box.width { p.x -= box.width + 10 }
        let maxZ = max(oneToOne * 4, fit * 8)
        let s2 = min(maxZ, max(fit, z))
        guard abs(s2 - scale) > 0.0001 else { return }
        if s2 <= fit * 1.001 { zoom = nil; pan = .zero; return }
        let r = imageRect(box, scale: scale)
        let ix = (p.x - r.minX) / scale, iy = (p.y - r.minY) / scale    // picture pixel under the pointer
        let w2 = after.size.width * s2, h2 = after.size.height * s2
        let newPan = CGSize(width: p.x - ix * s2 - (box.width - w2) / 2,
                            height: p.y - iy * s2 - (box.height - h2) / 2)
        zoom = s2
        pan = clamped(newPan, scale: s2, box: box)
    }

    private func setZoom(_ z: CGFloat, fit: CGFloat) {
        let maxZ = max(oneToOne * 4, fit * 8)
        let v = min(maxZ, max(fit, z))
        zoom = v <= fit * 1.001 ? nil : v
        if zoom == nil { pan = .zero }
    }

    private func picture(_ image: NSImage, _ r: CGRect) -> some View {
        Image(nsImage: image).resizable().interpolation(zoom == nil ? .high : .medium)
            .frame(width: r.width, height: r.height)
            .position(x: r.midX, y: r.midY)
    }

    private func panGesture(scale: CGFloat, box: CGSize) -> some Gesture {
        DragGesture(minimumDistance: 3)
            .onChanged { v in
                if dragMode == 0 { dragMode = 2; panStart = pan }
                pan = clamped(CGSize(width: panStart.width + v.translation.width,
                                     height: panStart.height + v.translation.height), scale: scale, box: box)
            }
            .onEnded { _ in dragMode = 0 }
    }

    // MARK: Views

    private func splitView(_ size: CGSize, scale: CGFloat) -> some View {
        let r = imageRect(size, scale: scale)
        let vis = r.intersection(CGRect(origin: .zero, size: size))
        let x = vis.minX + vis.width * split
        return ZStack(alignment: .topLeading) {
            picture(after, r)
            picture(before, r)   // Before on the left of the divider, After on the right
                .mask(alignment: .topLeading) { Rectangle().frame(width: max(0, x), height: size.height) }
            Rectangle().fill(.white).frame(width: 2, height: vis.height)
                .shadow(radius: 2)
                .position(x: x, y: vis.midY)
            Circle().fill(.white).frame(width: 28, height: 28).shadow(radius: 3)
                .overlay(Image(systemName: "arrow.left.and.right").font(.system(size: 12, weight: .bold))
                    .foregroundStyle(.black))
                .position(x: x, y: vis.midY)
            Text("Before").pill().position(x: vis.minX + 38, y: vis.minY + 22)
            Text("After").pill().position(x: vis.maxX - 34, y: vis.minY + 22)
        }
        .frame(width: size.width, height: size.height, alignment: .topLeading)
        .clipped()
        .contentShape(Rectangle())
        // Not zoomed: drag anywhere moves the divider. Zoomed: drag near the divider moves it, elsewhere pans.
        .gesture(DragGesture(minimumDistance: 0)
            .onChanged { v in
                if dragMode == 0 {
                    dragMode = (zoom == nil || abs(v.startLocation.x - x) < 24) ? 1 : 2
                    panStart = pan
                }
                if dragMode == 1 {
                    split = min(1, max(0, (v.location.x - vis.minX) / max(vis.width, 1)))
                } else {
                    pan = clamped(CGSize(width: panStart.width + v.translation.width,
                                         height: panStart.height + v.translation.height), scale: scale, box: size)
                }
            }
            .onEnded { _ in dragMode = 0 })
        .onContinuousHover { phase in
            if case .active(let p) = phase {
                (zoom != nil && abs(p.x - x) >= 24 ? NSCursor.openHand : NSCursor.resizeLeftRight).set()
            } else { NSCursor.arrow.set() }
        }
    }

    private func single(_ size: CGSize, scale: CGFloat) -> some View {
        let r = imageRect(size, scale: scale)
        return ZStack(alignment: .topLeading) {
            picture(holding ? before : after, r)
            Text(holding ? "Before · click for After" : "After · click to see Before").pill().padding(10)
        }
        .frame(width: size.width, height: size.height, alignment: .topLeading)
        .clipped()
        .contentShape(Rectangle())
        .onTapGesture { holding.toggle() }
        .gesture(panGesture(scale: scale, box: size))
        .onDisappear { holding = false }
    }

    private func sideBySide(_ size: CGSize, box: CGSize, scale: CGFloat) -> some View {
        let r = imageRect(box, scale: scale)
        return HStack(spacing: 10) {
            panel(before, "Before", box: box, r: r)
            panel(after, "After", box: box, r: r)
        }
        .frame(width: size.width, height: size.height)
        .contentShape(Rectangle())
        .gesture(panGesture(scale: scale, box: box))
    }

    private func panel(_ image: NSImage, _ title: String, box: CGSize, r: CGRect) -> some View {
        ZStack(alignment: .topLeading) {
            picture(image, r)
            Text(title).pill().padding(8)
        }
        .frame(width: box.width, height: box.height, alignment: .topLeading)
        .clipped()
    }

    // MARK: Zoom keys

    private func zoomBar(scale: CGFloat, fit: CGFloat, box: CGSize) -> some View {
        let pct = Int((scale / oneToOne * 100).rounded())
        return HStack(spacing: 4) {
            zoomKey("minus", help: "Zoom out") { setZoom(scale / 1.5, fit: fit) }
            zoomKey(nil, "FIT", lit: zoom == nil, help: "Fit the whole picture") { zoom = nil; pan = .zero }
            zoomKey(nil, "100%", lit: zoom != nil && abs(scale - oneToOne) < 0.001,
                    help: "Actual pixels") { setZoom(oneToOne, fit: fit) }
            zoomKey(nil, "200%", lit: zoom != nil && abs(scale - 2 * oneToOne) < 0.001,
                    help: "Twice actual size") { setZoom(2 * oneToOne, fit: fit) }
            zoomKey("plus", help: "Zoom in") { setZoom(scale * 1.5, fit: fit) }
            Text("\(pct)%")
                .font(Console.mono(10)).foregroundStyle(Console.amber)
                .frame(width: 42)
                .padding(.vertical, 4)
                .background(RoundedRectangle(cornerRadius: 3).fill(Console.ledGlass))
        }
        .padding(5)
        .background(RoundedRectangle(cornerRadius: 6).fill(Color.black.opacity(0.6)))
    }

    private func zoomKey(_ icon: String?, _ title: String = "", lit: Bool = false, help: String,
                         action: @escaping () -> Void) -> some View {
        Button {
            withAnimation(.easeOut(duration: 0.15)) { action() }
        } label: {
            Group {
                if let icon { Image(systemName: icon).font(.system(size: 10, weight: .bold)) }
                else { Text(title).font(Console.label(11)).tracking(0.5) }
            }
            .frame(minWidth: 26)
        }
        .buttonStyle(PresetKeyStyle(lit: lit))
        .help(help)
    }
}

extension Text {
    func pill() -> some View {
        self.font(.caption.weight(.semibold))
            .padding(.horizontal, 8).padding(.vertical, 4)
            .background(.black.opacity(0.55), in: Capsule())
            .foregroundStyle(.white)
    }
}

struct DropHighlight: ViewModifier {
    let onDrop: ([URL]) -> Void
    @State private var targeted = false

    func body(content: Content) -> some View {
        content
            .overlay {
                if targeted {
                    RoundedRectangle(cornerRadius: 12).strokeBorder(Console.amber, lineWidth: 3).padding(6)
                        .shadow(color: Console.amber, radius: 6)
                }
            }
            .dropDestination(for: URL.self) { urls, _ in
                onDrop(urls)
                return !urls.isEmpty
            } isTargeted: { targeted = $0 }
    }
}

extension View {
    func acceptsImageDrops(_ onDrop: @escaping ([URL]) -> Void) -> some View {
        modifier(DropHighlight(onDrop: onDrop))
    }
}

// MARK: - Console materials (procedural brushed metal with wear)

struct SeededRNG: RandomNumberGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

enum Materials {
    /// Brushed aluminium faceplate.
    static let faceplate = brushedMetal(width: 2200, height: 1500, base: 0.2, seed: 11, anodized: true)
    /// Charcoal anodized panel (scratches show bright bare metal).
    static let darkPanel = brushedMetal(width: 1400, height: 1400, base: 0.085, seed: 29, anodized: true)

    static func brushedMetal(width: Int, height: Int, base: CGFloat, seed: UInt64, anodized: Bool) -> NSImage {
        var rng = SeededRNG(state: seed)
        func r(_ a: CGFloat, _ b: CGFloat) -> CGFloat { CGFloat.random(in: a...b, using: &rng) }
        func coin() -> Bool { Bool.random(using: &rng) }
        func col(_ v: CGFloat, _ a: CGFloat = 1) -> CGColor {
            // a hint of warmth so it reads as aluminium, not flat grey
            CGColor(srgbRed: min(1, v * 1.01), green: v, blue: v * 0.975, alpha: a)
        }

        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return NSImage(size: NSSize(width: 10, height: 10)) }
        let W = CGFloat(width), H = CGFloat(height)

        // Base sheen
        let sheen = CGGradient(colorsSpace: cs, colors: [col(base + 0.07), col(base - 0.02), col(base + 0.03)] as CFArray,
                               locations: [0, 0.55, 1])!
        ctx.drawLinearGradient(sheen, start: CGPoint(x: 0, y: H), end: CGPoint(x: W * 0.3, y: 0),
                               options: [.drawsBeforeStartLocation, .drawsAfterEndLocation])

        // Brushing: thousands of fine horizontal strokes
        ctx.setLineCap(.butt)
        for _ in 0..<Int(H * 1.8) {
            let y = r(0, H), x0 = r(-300, W), len = r(80, W * 0.9)
            ctx.setStrokeColor(coin() ? col(1, r(0.015, 0.06)) : col(0, r(0.015, 0.075)))
            ctx.setLineWidth(r(0.4, 1.5))
            ctx.move(to: CGPoint(x: x0, y: y))
            ctx.addLine(to: CGPoint(x: x0 + len, y: y + r(-0.7, 0.7)))
            ctx.strokePath()
        }

        // Grain
        for _ in 0..<(width * height / 45) {
            ctx.setFillColor(coin() ? col(1, r(0.02, 0.07)) : col(0, r(0.02, 0.08)))
            ctx.fill(CGRect(x: r(0, W), y: r(0, H), width: 1, height: 1))
        }

        // Oxidation / fingerprints / handling smudges
        for _ in 0..<10 {
            let c = CGPoint(x: r(0, W), y: r(0, H)), rad = r(60, 300)
            let g = CGGradient(colorsSpace: cs, colors: [col(anodized ? 0.45 : 0.25, r(0.025, 0.07)), col(0.5, 0)] as CFArray,
                               locations: [0, 1])!
            ctx.drawRadialGradient(g, startCenter: c, startRadius: 0, endCenter: c, endRadius: rad, options: [])
        }

        // Scratches: a dark groove with a bright lip beside it
        func scratch(in region: CGRect?, long: Bool) {
            let box = region ?? CGRect(x: 0, y: 0, width: W, height: H)
            let p0 = CGPoint(x: r(box.minX, box.maxX), y: r(box.minY, box.maxY))
            let ang = r(0, .pi * 2), len = long ? r(240, 760) : r(10, 170)
            let p1 = CGPoint(x: p0.x + cos(ang) * len, y: p0.y + sin(ang) * len)
            let bend = len * 0.18
            let ctrl = CGPoint(x: (p0.x + p1.x) / 2 + r(-bend, bend), y: (p0.y + p1.y) / 2 + r(-bend, bend))
            let w = long ? r(0.6, 1.4) : r(0.35, 1.0)
            let path = CGMutablePath()
            path.move(to: p0)
            path.addQuadCurve(to: p1, control: ctrl)

            ctx.saveGState()
            ctx.setLineCap(.round)
            ctx.setLineWidth(w)
            ctx.translateBy(x: 0.7, y: -0.7)
            ctx.addPath(path)
            ctx.setStrokeColor(col(0, anodized ? r(0.4, 0.7) : r(0.12, 0.3)))
            ctx.strokePath()
            ctx.restoreGState()

            ctx.saveGState()
            ctx.setLineCap(.round)
            ctx.setLineWidth(w * 0.8)
            ctx.addPath(path)
            ctx.setStrokeColor(anodized ? col(0.82, r(0.18, 0.5)) : col(1, r(0.25, 0.6)))
            ctx.strokePath()
            ctx.restoreGState()
        }
        for _ in 0..<90 { scratch(in: nil, long: false) }
        for _ in 0..<7 { scratch(in: nil, long: true) }
        // Heavier wear where hands and rack rails rub: the side ears and lower lip
        let ear = W * 0.06
        for _ in 0..<60 { scratch(in: CGRect(x: 0, y: 0, width: ear, height: H), long: false) }
        for _ in 0..<60 { scratch(in: CGRect(x: W - ear, y: 0, width: ear, height: H), long: false) }
        for _ in 0..<40 { scratch(in: CGRect(x: 0, y: 0, width: W, height: H * 0.08), long: false) }

        // Dings and tiny dents
        for _ in 0..<14 {
            let c = CGPoint(x: r(0, W), y: r(0, H)), rad = r(1.5, 5)
            let g = CGGradient(colorsSpace: cs, colors: [col(0, 0.35), col(0, 0.05), col(1, 0.35), col(1, 0)] as CFArray,
                               locations: [0, 0.6, 0.8, 1])!
            ctx.drawRadialGradient(g, startCenter: c, startRadius: 0, endCenter: c, endRadius: rad, options: [])
        }

        // Soft vignette: edges catch less light
        let mid = CGPoint(x: W / 2, y: H / 2)
        let vig = CGGradient(colorsSpace: cs, colors: [col(0, 0), col(0, anodized ? 0.25 : 0.16)] as CFArray,
                             locations: [0.55, 1])!
        ctx.drawRadialGradient(vig, startCenter: mid, startRadius: 0, endCenter: mid,
                               endRadius: hypot(W, H) / 2, options: [.drawsAfterEndLocation])

        guard let cg = ctx.makeImage() else { return NSImage(size: NSSize(width: 10, height: 10)) }
        return NSImage(cgImage: cg, size: NSSize(width: W / 2, height: H / 2)) // @2x
    }
}

enum Console {
    static let amber = Color(red: 1.0, green: 0.68, blue: 0.22)
    static let green = Color(red: 0.42, green: 1.0, blue: 0.45)
    static let red = Color(red: 1.0, green: 0.22, blue: 0.16)
    static let ink = Color(white: 0.12)
    static let ledGlass = Color(red: 0.07, green: 0.045, blue: 0.03)
    static func label(_ size: CGFloat) -> Font { .custom("DIN Condensed", size: size) }
    static func mono(_ size: CGFloat) -> Font { .system(size: size, weight: .semibold, design: .monospaced) }
    static let marker = Font.custom("Marker Felt", size: 14)

    static func clock(_ s: Double?) -> String {
        guard let s, s.isFinite else { return "--:--" }
        let t = max(0, Int(s.rounded()))
        if t >= 3600 { return String(format: "%d:%02d:%02d", t / 3600, (t % 3600) / 60, t % 60) }
        return String(format: "%02d:%02d", t / 60, t % 60)
    }
}

extension View {
    /// Text that looks stamped into the metal.
    func engraved(onDark: Bool = true) -> some View {
        self.foregroundStyle(onDark ? Color(white: 0.87) : Console.ink)
            .shadow(color: onDark ? .black.opacity(0.9) : .white.opacity(0.75),
                    radius: 0, x: 0, y: onDark ? -0.7 : 0.9)
    }
}

struct EngravedLabel: View {
    let text: String
    var size: CGFloat = 12
    var onDark = true

    init(_ text: String, size: CGFloat = 12, onDark: Bool = true) {
        self.text = text
        self.size = size
        self.onDark = onDark
    }

    var body: some View {
        Text(text.uppercased()).font(Console.label(size)).tracking(1.4).lineLimit(1).engraved(onDark: onDark)
    }
}

struct Brushed: View {
    var dark = false
    var body: some View {
        // Color.clear takes exactly the space offered; the texture fills it and is cut to it.
        // (scaledToFill on its own reports a bigger size and spills over neighbouring views.)
        Color.clear
            .overlay {
                Image(nsImage: dark ? Materials.darkPanel : Materials.faceplate)
                    .resizable()
                    .interpolation(.high)
                    .scaledToFill()
            }
            .clipped()
            .allowsHitTesting(false)
    }
}

/// A charcoal panel sunk into the faceplate.
struct RecessedPanel: ViewModifier {
    var corner: CGFloat = 8
    func body(content: Content) -> some View {
        let shape = RoundedRectangle(cornerRadius: corner)
        return content
            .background { Brushed(dark: true).clipShape(shape) }
            .overlay {
                shape.stroke(Color.black.opacity(0.65), lineWidth: 6).blur(radius: 5).offset(y: 3).clipShape(shape)
            }
            .overlay {
                shape.strokeBorder(LinearGradient(colors: [.black.opacity(0.85), .white.opacity(0.16)],
                                                  startPoint: .top, endPoint: .bottom), lineWidth: 1.5)
            }
    }
}

extension View {
    func recessed(corner: CGFloat = 8) -> some View { modifier(RecessedPanel(corner: corner)) }
}

struct Screw: View {
    var angle: Double = 20
    var size: CGFloat = 13

    var body: some View {
        ZStack {
            Circle().fill(RadialGradient(colors: [Color(white: 0.96), Color(white: 0.64), Color(white: 0.38)],
                                         center: UnitPoint(x: 0.35, y: 0.3), startRadius: 0, endRadius: size * 0.75))
            Circle().strokeBorder(Color.black.opacity(0.55), lineWidth: 0.7)
            ZStack {
                Capsule().frame(width: size * 0.62, height: size * 0.15)
                Capsule().frame(width: size * 0.15, height: size * 0.62)
            }
            .foregroundStyle(Color.black.opacity(0.62))
            .rotationEffect(.degrees(angle))
        }
        .frame(width: size, height: size)
        .shadow(color: .black.opacity(0.45), radius: 1, y: 1)
    }
}

struct LED: View {
    let color: Color
    let on: Bool
    var blink = false
    let label: String

    var body: some View {
        VStack(spacing: 4) {
            TimelineView(.periodic(from: .now, by: 0.45)) { ctx in
                let lit = on && (!blink || Int(ctx.date.timeIntervalSinceReferenceDate / 0.45) % 2 == 0)
                Circle()
                    .fill(lit ? color : color.opacity(0.16))
                    .overlay(Circle().fill(RadialGradient(colors: [.white.opacity(lit ? 0.85 : 0.25), .clear],
                                                          center: UnitPoint(x: 0.35, y: 0.3), startRadius: 0, endRadius: 5)))
                    .overlay(Circle().strokeBorder(Color.black.opacity(0.65), lineWidth: 1))
                    .frame(width: 11, height: 11)
                    .shadow(color: lit ? color : .clear, radius: lit ? 6 : 0)
            }
            EngravedLabel(label, size: 10)
        }
    }
}

struct LEDReadout: View {
    let text: String
    var width: CGFloat? = nil
    var size: CGFloat = 12
    var color: Color = Console.amber

    var body: some View {
        Text(text)
            .font(Console.mono(size))
            .foregroundStyle(color)
            .shadow(color: color.opacity(0.75), radius: 3)
            .lineLimit(1)
            .padding(.horizontal, 7).padding(.vertical, 3)
            .frame(minWidth: width)
            .background(RoundedRectangle(cornerRadius: 3).fill(Console.ledGlass))
            .overlay(RoundedRectangle(cornerRadius: 3).strokeBorder(
                LinearGradient(colors: [.black.opacity(0.9), .white.opacity(0.16)], startPoint: .top, endPoint: .bottom),
                lineWidth: 1))
    }
}

// MARK: - Knobs & switches

struct KnobScale: View {
    var positions: [Double] = (0...10).map { Double($0) / 10 }
    var labels: [String]? = nil
    var radius: CGFloat = 36

    var body: some View {
        ZStack {
            ForEach(Array(positions.enumerated()), id: \.offset) { i, p in
                let deg = -135 + 270 * p
                Capsule().fill(Color(white: 0.8))
                    .frame(width: 1.6, height: (labels != nil || i % 5 == 0) ? 7 : 4)
                    .offset(y: -radius)
                    .rotationEffect(.degrees(deg))
                if let labels, i < labels.count {
                    let rad = deg * .pi / 180
                    Text(labels[i]).font(Console.label(10)).engraved()
                        .offset(x: CGFloat(sin(rad)) * (radius + 12), y: -CGFloat(cos(rad)) * (radius + 12))
                }
            }
        }
    }
}

struct KnobCap: View {
    let angle: Double

    var body: some View {
        ZStack {
            Circle().fill(Color.black.opacity(0.4)).frame(width: 56, height: 56).blur(radius: 3).offset(y: 3)
            // knurled aluminium skirt (lighting stays put while the knob turns)
            Circle()
                .fill(AngularGradient(colors: [Color(white: 0.9), Color(white: 0.45), Color(white: 0.92),
                                               Color(white: 0.4), Color(white: 0.9)], center: .center))
                .frame(width: 56, height: 56)
            ForEach(0..<40, id: \.self) { i in
                Rectangle().fill(Color.black.opacity(0.2)).frame(width: 1, height: 5)
                    .offset(y: -25.5).rotationEffect(.degrees(Double(i) * 9 + angle))
            }
            Circle().strokeBorder(Color.black.opacity(0.6), lineWidth: 1).frame(width: 56, height: 56)
            Rectangle().fill(Color.white).frame(width: 2, height: 6).offset(y: -25)
                .rotationEffect(.degrees(angle))
            // bakelite cap
            Circle()
                .fill(RadialGradient(colors: [Color(white: 0.34), Color(white: 0.07)],
                                     center: UnitPoint(x: 0.35, y: 0.28), startRadius: 1, endRadius: 30))
                .frame(width: 42, height: 42)
            Circle()
                .strokeBorder(LinearGradient(colors: [.white.opacity(0.35), .black.opacity(0.7)],
                                             startPoint: .top, endPoint: .bottom), lineWidth: 1)
                .frame(width: 42, height: 42)
            Capsule().fill(Color(white: 0.96)).frame(width: 2.6, height: 13).offset(y: -12.5)
                .rotationEffect(.degrees(angle))
        }
    }
}

/// Continuous knob: drag up/right to turn clockwise.
struct Knob: View {
    let label: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    var step: Double = 0.01
    var display: (Double) -> String
    var help: String = ""
    var size: CGFloat = 1
    @State private var start: Double?
    @Environment(\.isEnabled) private var isEnabled

    private var norm: Double {
        let span = range.upperBound - range.lowerBound
        return span > 0 ? min(1, max(0, (value - range.lowerBound) / span)) : 0
    }

    var body: some View {
        VStack(spacing: 6 * size) {
            EngravedLabel(label, size: size < 1 ? 10 : 12)
            ZStack {
                KnobScale(radius: 36 * size)
                KnobCap(angle: -135 + 270 * norm).scaleEffect(size)
            }
            .frame(width: 84 * size, height: 80 * size)
            .contentShape(Circle())
            .gesture(DragGesture(minimumDistance: 0)
                .onChanged { g in
                    if start == nil { start = value }
                    let span = range.upperBound - range.lowerBound
                    var v = (start ?? value) + Double(g.translation.width - g.translation.height) / 180 * span
                    v = (v / step).rounded() * step
                    value = min(range.upperBound, max(range.lowerBound, v))
                }
                .onEnded { _ in start = nil })
            LEDReadout(text: display(value), width: 58 * max(size, 0.85), size: size < 1 ? 10 : 11)
        }
        .opacity(isEnabled ? 1 : 0.4)
        .allowsHitTesting(isEnabled)
        .help(help)
    }
}

/// Detented rotary selector.
struct SteppedKnob: View {
    let label: String
    let options: [(label: String, readout: String, value: Int)]
    @Binding var selection: Int
    var help: String = ""
    var size: CGFloat = 1
    @State private var startIndex: Int?

    private var index: Int { options.firstIndex { $0.value == selection } ?? 0 }
    private var positions: [Double] {
        options.count < 2 ? [0] : (0..<options.count).map { Double($0) / Double(options.count - 1) }
    }

    var body: some View {
        VStack(spacing: 6 * size) {
            EngravedLabel(label, size: size < 1 ? 10 : 12)
            ZStack {
                KnobScale(positions: positions, labels: options.map { $0.label }, radius: 33 * size)
                KnobCap(angle: -135 + 270 * positions[index]).scaleEffect(size)
                    .animation(.spring(response: 0.25, dampingFraction: 0.55), value: index)
            }
            .frame(width: 108 * size, height: 96 * size)
            .contentShape(Circle())
            .onTapGesture { selection = options[(index + 1) % options.count].value }
            .gesture(DragGesture(minimumDistance: 3)
                .onChanged { g in
                    if startIndex == nil { startIndex = index }
                    let delta = Int(((g.translation.width - g.translation.height) / 30).rounded())
                    let i = min(options.count - 1, max(0, (startIndex ?? index) + delta))
                    if options[i].value != selection { selection = options[i].value }
                }
                .onEnded { _ in startIndex = nil })
            LEDReadout(text: options[index].readout, width: 70 * max(size, 0.85), size: size < 1 ? 10 : 11)
        }
        .help(help)
    }
}

/// Chrome bat-handle toggle switch.
struct ToggleSwitch: View {
    let label: String
    let off: String
    let on: String
    @Binding var isOn: Bool
    var help: String = ""
    @Environment(\.isEnabled) private var isEnabled

    var body: some View {
        VStack(spacing: 5) {
            EngravedLabel(label, size: 12)
            HStack(spacing: 6) {
                EngravedLabel(off, size: 11).opacity(isOn ? 0.5 : 1)
                ZStack {
                    // hex nut + bushing
                    Circle()
                        .fill(RadialGradient(colors: [Color(white: 0.95), Color(white: 0.55)],
                                             center: UnitPoint(x: 0.35, y: 0.3), startRadius: 0, endRadius: 16))
                        .overlay(Circle().strokeBorder(Color.black.opacity(0.5), lineWidth: 0.8))
                        .frame(width: 24, height: 24)
                        .shadow(color: .black.opacity(0.35), radius: 1.5, y: 1)
                    Circle().fill(Color(white: 0.18)).frame(width: 9, height: 9)
                    // bat handle
                    Capsule()
                        .fill(LinearGradient(colors: [Color(white: 0.98), Color(white: 0.55), Color(white: 0.85)],
                                             startPoint: .top, endPoint: .bottom))
                        .overlay(Capsule().strokeBorder(Color.black.opacity(0.35), lineWidth: 0.5))
                        .frame(width: 22, height: 6)
                        .offset(x: isOn ? 11 : -11)
                    Circle()
                        .fill(RadialGradient(colors: [.white, Color(white: 0.55)],
                                             center: UnitPoint(x: 0.35, y: 0.3), startRadius: 0, endRadius: 6))
                        .overlay(Circle().strokeBorder(Color.black.opacity(0.35), lineWidth: 0.5))
                        .frame(width: 9, height: 9)
                        .offset(x: isOn ? 22 : -22)
                        .shadow(color: .black.opacity(0.4), radius: 1.5, x: isOn ? 1 : -1, y: 1.5)
                }
                .frame(width: 56, height: 28)
                .contentShape(Rectangle())
                .onTapGesture { withAnimation(.spring(response: 0.18, dampingFraction: 0.6)) { isOn.toggle() } }
                EngravedLabel(on, size: 11).opacity(isOn ? 1 : 0.5)
            }
        }
        .opacity(isEnabled ? 1 : 0.6)
        .help(help)
    }
}

// MARK: - Buttons

/// Small aluminium push button with an engraved legend.
struct MetalButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View { MetalButtonBody(configuration: configuration) }
}

struct MetalButtonBody: View {
    let configuration: ButtonStyleConfiguration
    @Environment(\.isEnabled) private var isEnabled

    var body: some View {
        let pressed = configuration.isPressed
        configuration.label
            .font(Console.label(13)).tracking(1.2)
            .engraved(onDark: false)
            .padding(.horizontal, 11).padding(.top, 6).padding(.bottom, 4)
            .background(RoundedRectangle(cornerRadius: 4).fill(LinearGradient(
                colors: pressed ? [Color(white: 0.6), Color(white: 0.78)] : [Color(white: 0.93), Color(white: 0.66)],
                startPoint: .top, endPoint: .bottom)))
            .overlay(RoundedRectangle(cornerRadius: 4).strokeBorder(Color.black.opacity(0.55), lineWidth: 1))
            .shadow(color: .black.opacity(pressed ? 0.1 : 0.4), radius: pressed ? 0.5 : 2, y: pressed ? 0 : 1.5)
            .offset(y: pressed ? 1 : 0)
            .opacity(isEnabled ? 1 : 0.45)
            .contentShape(Rectangle())
    }
}

/// Illuminated push button: coloured lens with a crisp outline ring.
struct OutlinedButtonStyle: ButtonStyle {
    let fill: Color
    let outline: Color
    let text: Color
    var compact = false  // header-sized

    /// Light blue with a near-white outline.
    static let process = OutlinedButtonStyle(
        fill: Color(red: 0.56, green: 0.80, blue: 1.00),
        outline: Color(red: 0.96, green: 0.98, blue: 1.00),
        text: Color(red: 0.04, green: 0.14, blue: 0.30))

    /// Warm brown with a tan outline (HOW TO).
    static let help = OutlinedButtonStyle(
        fill: Color(red: 0.55, green: 0.33, blue: 0.16),
        outline: Color(red: 0.93, green: 0.72, blue: 0.45),
        text: Color(red: 1.0, green: 0.95, blue: 0.86),
        compact: true)

    /// Red with a bright orange outline.
    static let stop = OutlinedButtonStyle(
        fill: Color(red: 0.84, green: 0.12, blue: 0.12),
        outline: Color(red: 1.00, green: 0.58, blue: 0.00),
        text: .white)

    func makeBody(configuration: Configuration) -> some View {
        OutlinedButtonBody(configuration: configuration, style: self)
    }
}

struct OutlinedButtonBody: View {
    let configuration: ButtonStyleConfiguration
    let style: OutlinedButtonStyle
    @Environment(\.isEnabled) private var isEnabled

    var body: some View {
        let pressed = configuration.isPressed
        let shape = RoundedRectangle(cornerRadius: 7)
        configuration.label
            .font(Console.label(style.compact ? 15 : 18)).tracking(style.compact ? 1.5 : 2)
            .foregroundStyle(style.text)
            .padding(.horizontal, style.compact ? 16 : 22)
            .padding(.top, style.compact ? 8 : 12).padding(.bottom, style.compact ? 6 : 9)
            .background(shape.fill(style.fill))
            // lens: lit from behind, glossy on top
            .overlay(shape.fill(LinearGradient(colors: [.white.opacity(0.45), .clear, .black.opacity(0.18)],
                                               startPoint: .top, endPoint: .bottom)))
            .overlay(shape.strokeBorder(style.outline, lineWidth: 2.5))
            // dark bezel ring the button sits in
            .padding(3)
            .background(RoundedRectangle(cornerRadius: 9).fill(Color(white: 0.1)))
            .shadow(color: style.fill.opacity(isEnabled ? 0.65 : 0), radius: pressed ? 6 : 10)
            .shadow(color: .black.opacity(0.4), radius: 2, y: 2)
            .brightness(pressed ? -0.1 : 0)
            .scaleEffect(pressed ? 0.97 : 1)
            .saturation(isEnabled ? 1 : 0.25)
            .opacity(isEnabled ? 1 : 0.55)
            .animation(.easeOut(duration: 0.12), value: pressed)
            .contentShape(Rectangle())
    }
}

// MARK: - Panels

struct BrandPlate: View {
    var body: some View {
        HStack(spacing: 10) {
            Screw(angle: 35, size: 9)
            VStack(alignment: .leading, spacing: 0) {
                Text("PROTOART")
                    .font(.custom("Futura-CondensedExtraBold", size: 22)).tracking(2)
                    .engraved(onDark: true)
                Text("BEAUTIFY · IMAGE ENHANCER · MODEL PA-768")
                    .font(Console.label(10)).tracking(1.2)
                    .foregroundStyle(Console.amber.opacity(0.85))
            }
            Screw(angle: 110, size: 9)
        }
        .padding(.horizontal, 10).padding(.vertical, 6)
        .recessed(corner: 6)
    }
}

struct HeaderStrip: View {
    @EnvironmentObject var model: AppModel

    private var fault: Bool {
        model.errorMessage != nil || model.items.contains { if case .failed = $0.state { return true }; return false }
    }

    var body: some View {
        HStack(spacing: 18) {
            Color.clear.frame(width: 62) // window buttons live here
            BrandPlate()
            Spacer(minLength: 10)
            HStack(spacing: 8) {
                Button(model.batchMode ? "ADD" : "OPEN") { model.openPanel() }
                Button("SAVE") { model.quickSave() }.disabled(!model.canSave)
                    .help(model.saveMode == "overwrite" ? "Replace the original photo (⌘S)"
                          : "Save a new file in \(model.outputFolder.lastPathComponent) (⌘S)")
                Button("SAVE AS") { model.saveSelectedAs() }.disabled(!model.canSave)
                    .help("Choose name, folder and format (⇧⌘S)")
                Button("FINDER") { model.reveal() }
                    .disabled(model.selected?.savedURL == nil && model.selected?.outputURL == nil)
            }
            .buttonStyle(MetalButtonStyle())
            Spacer(minLength: 10)
            Button { model.showHelp = true } label: {
                HStack(spacing: 6) {
                    Image(systemName: "questionmark.circle.fill").font(.system(size: 15, weight: .bold))
                    Text("HOW TO")
                }
            }
            .buttonStyle(OutlinedButtonStyle.help)
            .help("How the app works and fixes for common problems (⌘?)")
            HStack(spacing: 16) {
                LED(color: Console.green, on: model.engineReady, label: "Ready")
                LED(color: Console.amber, on: model.running || !model.engineReady, blink: true,
                    label: model.engineReady ? "Process" : "Loading")
                LED(color: Console.red, on: fault, label: "Fault")
            }
            Rectangle().fill(Color.black.opacity(0.7)).frame(width: 1, height: 38)
                .overlay(Rectangle().fill(Color.white.opacity(0.1)).frame(width: 1).offset(x: 1))
            ToggleSwitch(label: "Batch", off: "Off", on: "On", isOn: $model.batchMode,
                         help: "Batch mode: queue many images, auto-save results, 768 px default for speed (⌘B)")
                .disabled(model.running)
                .guide(.batch)
        }
        .padding(.leading, 18).padding(.trailing, 30)
    }
}

/// A control module: engraved title with a groove line, contents below.
struct Module<Content: View>: View {
    let title: String
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                EngravedLabel(title, size: 13)
                VStack(spacing: 0) {
                    Rectangle().fill(Color.black.opacity(0.35)).frame(height: 1)
                    Rectangle().fill(Color.white.opacity(0.1)).frame(height: 1)
                }
            }
            content
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.black.opacity(0.22)))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Color.black.opacity(0.7), lineWidth: 1))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Color.white.opacity(0.07), lineWidth: 1).offset(y: 1))
    }
}

/// Masking tape label you can write on.
struct Tape: View {
    var body: some View {
        Rectangle()
            .fill(LinearGradient(colors: [Color(red: 0.96, green: 0.92, blue: 0.80), Color(red: 0.90, green: 0.85, blue: 0.70)],
                                 startPoint: .top, endPoint: .bottom))
            .overlay(alignment: .top) { Rectangle().fill(Color.white.opacity(0.25)).frame(height: 1.5) }
            .overlay(alignment: .leading) { Rectangle().fill(Color.black.opacity(0.06)).frame(width: 3) }
            .shadow(color: .black.opacity(0.28), radius: 1.2, y: 1)
    }
}

struct TapeField: View {
    let label: String
    @Binding var text: String
    var tilt: Double = -0.6

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            EngravedLabel(label, size: 11)
            TextField("", text: $text, axis: .vertical)
                .textFieldStyle(.plain)
                .font(Console.marker)
                .foregroundStyle(Color(red: 0.08, green: 0.1, blue: 0.22))
                .lineLimit(1...3)
                .padding(.horizontal, 10).padding(.vertical, 6)
                .background(Tape())
                .rotationEffect(.degrees(tilt))
        }
    }
}


// MARK: - Guide (plain-language help for every control) and presets

enum Guide: Equatable {
    case instant, turbo, fast, balanced, quality, subtle, creative
    case upscale, maxSize, memory, batch, format, saveTo, saveMode, save
    case filterStage, bleach, dark, extract, viewMode, filtersKey, light, color, grain, bloom, facePreset
    case creativity, resemblance, hdr, steps, decay, solver, seed, prompt, negative
    case faceFix, faceModel, faceStrength, faceFidelity, zoom, likeness, myPresets, addQueue, autoSave
    case keepTones, faceDetail, faceDetailStrength, faceDetailHair
    case presence, mixer, grading, glow, blurFilter, skin

    var title: String {
        switch self {
        case .instant: return "Preset · Instant"
        case .turbo: return "Preset · Turbo"
        case .fast: return "Preset · Fast"
        case .balanced: return "Preset · Balanced"
        case .quality: return "Preset · Quality"
        case .subtle: return "Preset · Subtle"
        case .creative: return "Preset · Creative"
        case .upscale: return "Upscale"
        case .maxSize: return "Max Size"
        case .memory: return "Memory"
        case .batch: return "Batch"
        case .format: return "Format"
        case .saveTo: return "Save To"
        case .saveMode: return "Save mode"
        case .save: return "Save"
        case .filterStage: return "Apply filters"
        case .bleach: return "Bleach Bypass"
        case .dark: return "Dark Contrasts"
        case .extract: return "Detail Extractor"
        case .viewMode: return "View"
        case .filtersKey: return "Adjust panel"
        case .light: return "Light"
        case .grain: return "Film Grain"
        case .bloom: return "Bloom"
        case .facePreset: return "Face presets"
        case .color: return "Color"
        case .creativity: return "Creativity"
        case .resemblance: return "Resemblance"
        case .hdr: return "HDR"
        case .steps: return "Steps"
        case .decay: return "Decay"
        case .solver: return "Solver"
        case .seed: return "Seed"
        case .prompt: return "Prompt"
        case .negative: return "Negative prompt"
        case .faceFix: return "Face Fix"
        case .faceModel: return "Face model"
        case .faceStrength: return "Face strength"
        case .faceFidelity: return "Fidelity (CodeFormer)"
        case .zoom: return "Zoom"
        case .likeness: return "Likeness"
        case .myPresets: return "My presets"
        case .addQueue: return "+ Queue"
        case .autoSave: return "Auto-save"
        case .keepTones: return "Keep Tones"
        case .faceDetail: return "Face Detail"
        case .faceDetailStrength: return "Face Detail strength"
        case .faceDetailHair: return "Face Detail · Hair"
        case .presence: return "Presence"
        case .mixer: return "Color Mixer"
        case .grading: return "Color Grading"
        case .glow: return "Glamour Glow"
        case .blurFilter: return "Blur"
        case .skin: return "Skin Texture"
        }
    }

    var text: String {
        switch self {
        case .instant:
            return "Seconds, not minutes. Only the 4× sharpening upscaler runs, with no AI re-detailing, so nothing is invented. Great for previews, huge batches and text or logos."
        case .turbo:
            return "About twice as fast as Fast with similar results: a distilled add-on finishes the AI detail in 2 quick passes and skips the negative-prompt pass. HDR has no effect in Turbo."
        case .fast:
            return "Quickest results. Caps the output at 768 px and uses fewer, smarter refinement steps. Great for previews and big batches."
        case .balanced:
            return "The original web app's settings, capped at 1024 px. A good default for most photos and artwork."
        case .quality:
            return "Slowest, cleanest detail. More refinement steps and a 1536 px output. Use it for final images you'll print or post."
        case .subtle:
            return "Gentle cleanup that stays very close to the original: sharper and less noisy, with almost no invented detail. Safest for faces, logos and text. Keeps your Upscale and Max Size."
        case .creative:
            return "Lets the AI add lots of new texture and fine detail. Can look stunning on landscapes and illustrations, but may alter faces or lettering. Keeps your Upscale and Max Size: set Upscale to 1.0× to add detail without enlarging."
        case .upscale:
            return "How much bigger the result is: 2× doubles width and height; 1.0× keeps the size and only re-details it (refine only). Presets never change this. Max Size can cap the final size."
        case .maxSize:
            return "Caps the long side of the result. Smaller is much faster and uses less memory: 768 px is quickest, ∞ means no cap (slow on 8 GB). Click to step through."
        case .memory:
            return "8 GB keeps everything safe for your Mac by working in smaller pieces. 16 GB+ uses the original web app's larger pieces: faster on big Macs, but it may crash on 8 GB."
        case .batch:
            return "Off: one image at a time. On: a queue appears on the left. Drop many images or folders and they're enhanced one after another, saved automatically."
        case .format:
            return "File type for new files. PNG is lossless but large; JPEG is much smaller at high quality (95%); TIFF is lossless for print and editing."
        case .saveTo:
            return "Your default save folder: Save puts new files here, Save As starts here, and batch results land here. Click the tape to choose a folder."
        case .saveMode:
            return "COPY saves a brand-new file (photo-beautify.png) and never touches your original. OVERWRITE replaces the original photo, keeping its name and format; it asks first."
        case .save:
            return "Saves the result with the settings above (⌘S). Save As (⇧⌘S) lets you pick any name, folder and format. Detail filters are included."
        case .filterStage:
            return "AFTER AI: adjustments and filters go on top of the enhanced result (live, change them any time, nothing re-runs). BEFORE AI: the photo is filtered first and the AI enhances that. Either way you can save a brand-new file."
        case .bleach:
            return "Bleach Bypass: silvery, high-contrast, half-desaturated film look. Click the name to apply it and open its knobs; the checkbox turns it on or off."
        case .dark:
            return "Dark Contrasts: deeper, moodier tones with crisp mid-tone detail. Detail Strength sets how dark and punchy; raise Brightness to keep it from getting muddy."
        case .extract:
            return "Detail Extractor: pulls out texture everywhere for an HDR-like, illustrated look. Radius: FINE for small texture, COARSE for big shapes."
        case .filtersKey:
            return "Shows or hides the Adjust panel beside the picture: Camera Raw-style Light, Color, Presence, Color Mixer and Color Grading, plus the Detail Extractor, Dark Contrasts, Bleach Bypass, Glamour Glow, Bloom, Blur, Skin Texture and Film Grain filters."
        case .bloom:
            return "Bloom, like Procreate's: the bright parts of the picture spill soft light around themselves. Amount = how strong, Transition = how much counts as bright (low = only the brightest highlights), Size = how far the light spreads, Burn = from a soft screen glow to light that can blow out to white."
        case .facePreset:
            return "One-click CodeFormer settings (the same as Protoart Studio's). Balanced suits most portraits; Keep likeness for real people; Heavy repair for distorted or tiny faces; Subtle keeps skin texture; Old photo for scans; Gentle for faces that only need a polish."
        case .grain:
            return "Film Grain: photographic grain over the picture. Intensity = how visible, Grain Size = fine to coarse, Softness = crisp to smooth, Mono or Color grain. Usually best last in the stack."
        case .light:
            return "Like Camera Raw's Light panel: Exposure (overall brightness), Contrast, Highlights and Shadows (recover or deepen), Whites and Blacks (end points). Double-click a slider to reset it."
        case .color:
            return "Temperature (cooler ↔ warmer), Tint (green ↔ magenta), Vibrance (boosts muted colors, protects skin) and Saturation (all colors). Double-click a slider to reset it."
        case .viewMode:
            return "SINGLE shows the result (click it to flip to the original and back). SPLIT has a divider you drag anywhere: Before on the left, After on the right. SIDE shows both side by side."
        case .creativity:
            return "How much new detail the AI may invent. Low (0.2) is gentle cleanup that stays true to the original. High (0.6+) adds lots of texture but can change faces or text. 0.35 is the sweet spot."
        case .resemblance:
            return "How closely the result follows the original's shapes and layout. Higher stays faithful; lower gives the AI more freedom, and it may drift from the original."
        case .hdr:
            return "Contrast and 'punch' of the new detail. 4–7 looks natural; higher gets crunchy and over-sharpened."
        case .steps:
            return "Refinement passes. More is cleaner but slower. Only Creativity × Steps actually run, e.g. 0.35 × 18 ≈ 6 passes."
        case .decay:
            return "Loosens how strictly the finest layers follow the original. 1.0 is even guidance; around 0.8 allows a little more texture. Most people leave it at 1.0."
        case .solver:
            return "The math used for each refinement step. DDIM is the steady default. DPM gets good results in fewer steps (the Fast preset uses it)."
        case .seed:
            return "The random starting point. The same seed and settings give the same result. Press RND for a different variation of the same image."
        case .prompt:
            return "Words describing what you want. Usually leave it as is, or add hints like 'detailed skin, sharp eyes' for portraits or 'crisp foliage' for landscapes."
        case .negative:
            return "Things to avoid. The default steers away from blurry, low-quality artifacts."
        case .faceFix:
            return "Finds every face and restores it after the enhance: cleaner eyes, teeth and skin, fixing the warped faces AI detail can cause. Blended in with a soft edge. The first use downloads the face model once."
        case .faceModel:
            return "CODE (CodeFormer, the default): strong repair for blurry or damaged faces, with a Fidelity knob; its licence allows non-commercial use only. RESTORE (RestoreFormer): natural and faithful, fine for commercial use."
        case .faceStrength:
            return "How much of the restored face is used. 100% = fully restored; lower blends back toward the enhanced face for a subtler, more natural result. 70–80% suits most photos."
        case .faceFidelity:
            return "CodeFormer only. Low (0.2–0.4): cleanest, prettiest face, but may look less like the person. High (0.7–1.0): keeps the person's own features. 0.5 is a good balance."
        case .likeness:
            return "Keeps people and things looking like your photo while the AI adds detail (IP-Adapter Plus). 30–60% is a good start; higher sticks closer to the photo. Each part of the picture is guided by its own part of the photo. The first use downloads it once (1.3 GB)."
        case .myPresets:
            return "Your own presets: set the knobs, then + SAVE and give it a name (tick 'Include Adjust & filters' to keep those too). Click one to use it; right-click to update, rename or delete it."
        case .addQueue:
            return "Try several versions: + QUEUE adds this photo again with the current settings. Change the knobs (or pick a preset) and + QUEUE again, then press ENHANCE to run them all. Each result is saved with its settings in the file name."
        case .autoSave:
            return "On: every finished result is saved straight to your save folder (the Desktop unless you changed it), so you never have to hunt for it. Off: press SAVE yourself. In Overwrite mode results are never saved over your photo without SAVE."
        case .keepTones:
            return "On (recommended): after the AI adds detail, your photo's own brightness and colour are put back, so the result never comes out lighter or with shifted colours; all the new fine detail stays. Off: keep the AI's own light and colour (it often brightens a little)."
        case .faceDetail:
            return "Like Protoart Studio's face detailer: after the enhance, each face is cut out with some room around it, redrawn close up at 512–768 px by the same AI, colour-matched and blended back with a soft edge. Small or distant faces get real eyes, lips and skin. Adds roughly 10–30 s per face. Runs before Face Fix, so you can use both. Not used by INSTANT. Turn it off with the switch."
        case .faceDetailStrength:
            return "How much the close-up may change each face. 15–20% (the default is 15%) sharpens and adds detail while keeping the person; 30% and up redraws more and can change the look. The face's own shapes always guide the redraw, and it's matched to its surroundings."
        case .faceDetailHair:
            return "Off: only the face (forehead to chin) is redrawn. On: the hair above and beside the face is redrawn too, for portraits where the hair looks soft."
        case .presence:
            return "Like Camera Raw's Presence and Effects: Texture (fine detail such as skin and foliage, not edges), Clarity (mid-tone punch over larger areas), Dehaze (+ removes haze and mist, − adds a soft veil), Sharpen (fine edges) and Vignette (+ darkens, − lightens the corners). Double-click a slider to reset it."
        case .mixer:
            return "Like Camera Raw's Color Mixer: pick a colour band, then shift its Hue, Saturation and Luminance. Neighbouring colours blend smoothly; greys are left alone. RESET clears every band."
        case .grading:
            return "Like Camera Raw's Color Grading: tint the Shadows, Midtones and Highlights with any hue (Hue picks the colour, Amount how strong). Balance moves the split between shadows and highlights. Brightness is kept; only the colour changes."
        case .glow:
            return "Glamour Glow (Nik Color Efex): a soft, flattering glow over the picture. Glow = how strong, Saturation = colour of the glow, Warmth = warmer or cooler skin."
        case .blurFilter:
            return "Blur: softens the picture. Size = how far, Mix = how much of the blur is used. SOFT FOCUS screens the blur over the picture so it glows instead of smearing (a dreamy portrait look)."
        case .skin:
            return "Skin Texture: adds natural pores, fine bumps and optional freckles to smooth, plastic-looking AI skin. It only shades (no new colours), most in the mid-tones. Pattern changes the random layout. Tip: use it low (20–40%)."
        case .zoom:
            return "Scroll the mouse wheel (or two fingers on a trackpad) over the picture to zoom smoothly toward the pointer; pinch works too. Drag to look around. FIT shows the whole picture."
        }
    }
}

final class GuideCenter: ObservableObject {
    @Published var current: Guide?
}

struct GuideHover: ViewModifier {
    @EnvironmentObject var guides: GuideCenter
    let guide: Guide

    func body(content: Content) -> some View {
        content.onHover { inside in
            if inside {
                guides.current = guide
            } else if guides.current == guide {
                guides.current = nil
            }
        }
    }
}

extension View {
    func guide(_ g: Guide) -> some View { modifier(GuideHover(guide: g)) }
}

struct Preset: Identifiable {
    let id: String
    let guide: Guide
    /// Max Size this preset sets (nil: leave it alone). Presets never change Upscale, so any look can be
    /// used at 1× (refine only), 1.5×, 2× …
    let maxSide: Int?
    let creativity: Double
    let resemblance: Double
    let hdr: Double
    let steps: Double
    let solver: String
    var mode: String = "detail"
    var turbo = false

    static let all: [Preset] = [
        Preset(id: "INSTANT", guide: .instant, maxSide: 0, creativity: 0.35, resemblance: 0.6,
               hdr: 6, steps: 18, solver: "DDIM", mode: "instant"),
        Preset(id: "TURBO", guide: .turbo, maxSide: 768, creativity: 0.35, resemblance: 0.6,
               hdr: 6, steps: 8, solver: "DDIM", turbo: true),
        Preset(id: "FAST", guide: .fast, maxSide: 768, creativity: 0.35, resemblance: 0.6,
               hdr: 6, steps: 10, solver: "DPMSolver"),
        Preset(id: "BALANCED", guide: .balanced, maxSide: 1024, creativity: 0.35, resemblance: 0.6,
               hdr: 6, steps: 18, solver: "DDIM"),
        Preset(id: "QUALITY", guide: .quality, maxSide: 1536, creativity: 0.35, resemblance: 0.6,
               hdr: 6, steps: 26, solver: "DDIM"),
        Preset(id: "SUBTLE", guide: .subtle, maxSide: nil, creativity: 0.2, resemblance: 0.9,
               hdr: 5, steps: 20, solver: "DDIM"),
        Preset(id: "CREATIVE", guide: .creative, maxSide: nil, creativity: 0.55, resemblance: 0.45,
               hdr: 7, steps: 20, solver: "DDIM"),
    ]

    @MainActor func apply(to model: AppModel) {
        model.settings.mode = mode
        model.settings.turbo = turbo
        if let maxSide { model.maxSide = maxSide }
        if mode == "instant" { return }  // keep the AI knobs where they were
        model.settings.creativity = creativity
        model.settings.resemblance = resemblance
        model.settings.resemblanceDecay = 1.0
        model.settings.hdr = hdr
        model.settings.steps = steps
        model.settings.solver = solver
    }

    @MainActor func matches(_ model: AppModel) -> Bool {
        let s = model.settings
        func eq(_ a: Double, _ b: Double) -> Bool { abs(a - b) < 0.001 }
        guard s.mode == mode, s.turbo == turbo || mode == "instant" else { return false }
        let sizeOK = maxSide.map { model.maxSide == $0 } ?? true
        if mode == "instant" { return sizeOK }
        return eq(s.creativity, creativity) && eq(s.resemblance, resemblance)
            && eq(s.resemblanceDecay, 1.0) && eq(s.hdr, hdr) && eq(s.steps, steps) && s.solver == solver
            && sizeOK
    }
}

/// Small console key with an indicator LED above it.
struct PresetKey: View {
    let preset: Preset
    let lit: Bool
    let action: () -> Void
    @State private var pressed = false

    var body: some View {
        VStack(spacing: 5) {
            Circle()
                .fill(lit ? Console.amber : Console.amber.opacity(0.14))
                .overlay(Circle().strokeBorder(Color.black.opacity(0.7), lineWidth: 0.8))
                .frame(width: 7, height: 7)
                .shadow(color: lit ? Console.amber : .clear, radius: lit ? 5 : 0)
            Button(action: action) {
                Text(preset.id)
                    .font(Console.label(11)).tracking(0.6)
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(PresetKeyStyle(lit: lit))
        }
        .frame(maxWidth: .infinity)
        .guide(preset.guide)
    }
}

struct PresetKeyStyle: ButtonStyle {
    let lit: Bool
    func makeBody(configuration: Configuration) -> some View {
        let pressed = configuration.isPressed
        return configuration.label
            .foregroundStyle(lit ? Color(white: 0.08) : Color(white: 0.82))
            .padding(.top, 6).padding(.bottom, 4)
            .background(RoundedRectangle(cornerRadius: 3).fill(LinearGradient(
                colors: lit ? [Color(white: 0.92), Color(white: 0.68)]
                            : (pressed ? [Color(white: 0.12), Color(white: 0.2)] : [Color(white: 0.26), Color(white: 0.13)]),
                startPoint: .top, endPoint: .bottom)))
            .overlay(RoundedRectangle(cornerRadius: 3).strokeBorder(Color.black.opacity(0.8), lineWidth: 1))
            .shadow(color: .black.opacity(pressed ? 0.1 : 0.5), radius: pressed ? 0.5 : 2, y: pressed ? 0 : 1.5)
            .offset(y: pressed ? 1 : 0)
            .contentShape(Rectangle())
    }
}

/// Amber LCD strip under the monitor that explains whatever you're pointing at.
struct GuideStrip: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var guides: GuideCenter

    private var shown: (String, String) {
        if let g = guides.current { return (g.title, g.text) }
        if let p = Preset.all.first(where: { $0.id == model.activePreset }) { return (p.guide.title, p.guide.text) }
        return (model.settings.turbo ? "Custom settings · Turbo on" : "Custom settings",
                "Hover over any knob, switch or preset to see what it does. Pick a preset above the knobs to start from a known-good setup.")
    }

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            EngravedLabel("Guide", size: 11, onDark: true)
                .padding(.top, 2)
            VStack(alignment: .leading, spacing: 3) {
                Text(shown.0.uppercased())
                    .font(Console.mono(11))
                    .foregroundStyle(Console.amber)
                    .shadow(color: Console.amber.opacity(0.7), radius: 3)
                Text(shown.1)
                    .font(.system(size: 12))
                    .foregroundStyle(Color(red: 1.0, green: 0.86, blue: 0.62).opacity(0.9))
                    .lineLimit(3)
                    .minimumScaleFactor(0.85)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        // Fixed height: hovering a control with a long explanation must never resize the picture above.
        .frame(maxWidth: .infinity, minHeight: 80, maxHeight: 80, alignment: .topLeading)
        .background(RoundedRectangle(cornerRadius: 4).fill(Console.ledGlass))
        .overlay(RoundedRectangle(cornerRadius: 4).strokeBorder(Color.black.opacity(0.85), lineWidth: 1))
        .padding(6)
        .recessed(corner: 8)
        .animation(.easeOut(duration: 0.12), value: shown.0)
    }
}

struct ControlsPanel: View {
    @EnvironmentObject var model: AppModel

    private var memory16: Binding<Bool> {
        Binding(get: { model.settings.profile == "16gb" }, set: { model.settings.profile = $0 ? "16gb" : "8gb" })
    }
    private var overwrite: Binding<Bool> {
        Binding(get: { model.saveMode == "overwrite" }, set: { model.saveMode = $0 ? "overwrite" : "copy" })
    }

    private var dpm: Binding<Bool> {
        Binding(get: { model.settings.solver == "DPMSolver" }, set: { model.settings.solver = $0 ? "DPMSolver" : "DDIM" })
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 14) {
                Module(title: "Presets") {
                    LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 5), count: 4), spacing: 8) {
                        ForEach(Preset.all) { p in
                            PresetKey(preset: p, lit: model.activePreset == p.id) {
                                withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) { model.applyPreset(p) }
                            }
                        }
                    }
                    HStack {
                        EngravedLabel("My presets", size: 11)
                        Spacer()
                        Button("+ SAVE") { model.saveUserPreset() }
                            .buttonStyle(MetalButtonStyle())
                            .help("Save the current settings as your own preset")
                    }
                    .guide(.myPresets)
                    if model.userPresets.isEmpty {
                        EngravedLabel("Set the knobs you like, then + SAVE", size: 10)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    } else {
                        LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 5), count: 3), spacing: 6) {
                            ForEach(model.userPresets) { u in
                                Button { withAnimation(.easeOut(duration: 0.15)) { model.applyUserPreset(u) } } label: {
                                    Text(u.name.uppercased()).font(Console.label(10)).tracking(0.4)
                                        .lineLimit(1).minimumScaleFactor(0.7).frame(maxWidth: .infinity)
                                }
                                .buttonStyle(PresetKeyStyle(lit: model.userPresetMatches(u)))
                                .help(u.name + (u.filters != nil ? " (with Adjust & filters)" : "") + " · right-click to update, rename or delete")
                                .contextMenu {
                                    Button("Update with Current Settings") { model.updateUserPreset(u) }
                                    Button("Rename…") { model.renameUserPreset(u) }
                                    Divider()
                                    Button("Delete…") { model.deleteUserPreset(u) }
                                }
                            }
                        }
                        .guide(.myPresets)
                    }
                }

                Module(title: "Output") {
                    HStack(alignment: .top, spacing: 4) {
                        Knob(label: "Upscale", value: $model.settings.upscaleFactor, range: 1...4, step: 0.5,
                             display: { String(format: "%.1f×", $0) },
                             help: "Final size multiplier (before the max-size cap).")
                            .guide(.upscale)
                        SteppedKnob(label: "Max Size", options: [
                            ("768", "768 PX", 768), ("1K", "1024 PX", 1024), ("1.5K", "1536 PX", 1536),
                            ("2K", "2048 PX", 2048), ("∞", "NO LIMIT", 0),
                        ], selection: $model.maxSide,
                        help: "Caps the long side of the result. Smaller is much faster and lighter on memory. Click or drag to change.")
                        .guide(.maxSize)
                    }
                    .frame(maxWidth: .infinity)
                    ToggleSwitch(label: "Memory", off: "8 GB", on: "16 GB+", isOn: memory16,
                                 help: "8 GB: 512 px input cap and 512 px tiles. 16 GB+: the Space's original settings.")
                        .guide(.memory)
                        .frame(maxWidth: .infinity)
                }

                Module(title: "Save") {
                    VStack(alignment: .leading, spacing: 4) {
                        EngravedLabel("Save To", size: 11)
                        Button { model.chooseOutputFolder() } label: {
                            Text(model.outputFolder.lastPathComponent)
                                .font(Console.marker)
                                .foregroundStyle(Color(red: 0.08, green: 0.1, blue: 0.22))
                                .lineLimit(1).truncationMode(.middle)
                                .padding(.horizontal, 10).padding(.vertical, 6)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(Tape())
                                .rotationEffect(.degrees(0.8))
                        }
                        .buttonStyle(.plain)
                        .help(model.outputFolder.path)
                    }
                    .guide(.saveTo)
                    HStack(alignment: .bottom, spacing: 12) {
                        VStack(alignment: .leading, spacing: 5) {
                            EngravedLabel("Format", size: 11)
                            HStack(spacing: 4) {
                                ForEach([("PNG", "png"), ("JPEG", "jpg"), ("TIFF", "tif")], id: \.1) { f in
                                    Button { model.outputFormat = f.1 } label: {
                                        Text(f.0).font(Console.label(11)).tracking(0.6).frame(width: 38)
                                    }
                                    .buttonStyle(PresetKeyStyle(lit: model.outputFormat == f.1))
                                }
                            }
                            .disabled(model.saveMode == "overwrite")
                            .opacity(model.saveMode == "overwrite" ? 0.5 : 1)
                        }
                        .guide(.format)
                        ToggleSwitch(label: "Mode", off: "Copy", on: "Overwrite", isOn: overwrite)
                            .guide(.saveMode)
                    }
                    HStack(spacing: 10) {
                        ToggleSwitch(label: "Auto-save", off: "Off", on: "On", isOn: $model.autoSaveResults)
                            .disabled(model.saveMode == "overwrite" && !model.batchMode)
                        EngravedLabel(model.saveMode == "overwrite" && !model.batchMode
                                      ? "Off in Overwrite mode: press SAVE" : "Saves each result as soon as it's done",
                                      size: 10)
                        Spacer(minLength: 0)
                    }
                    .guide(.autoSave)
                    HStack {
                        EngravedLabel(model.batchMode ? "Batch results save automatically" :
                                        (model.saveMode == "overwrite" ? "Replaces the original" :
                                            (model.autoSaveResults ? "Every result saves itself here" : "Saves a new file")),
                                      size: 10)
                        Spacer()
                        Button("SAVE") { model.quickSave() }
                            .buttonStyle(MetalButtonStyle())
                            .disabled(!model.canSave)
                            .guide(.save)
                    }
                }

                Module(title: "Tone") {
                    HStack(spacing: 2) {
                        Knob(label: "Creativity", value: $model.settings.creativity, range: 0.05...1, step: 0.05,
                             display: { String(format: "%.2f", $0) },
                             help: "Denoise strength. Higher invents more detail; lower stays closer to the original.")
                            .guide(.creativity)
                        Knob(label: "Resemblance", value: $model.settings.resemblance, range: 0...1.5, step: 0.05,
                             display: { String(format: "%.2f", $0) },
                             help: "Tile ControlNet scale. Higher sticks to the original structure.")
                            .guide(.resemblance)
                        Knob(label: "HDR", value: $model.settings.hdr, range: 2...20, step: 1,
                             display: { String(format: "%.0f", $0) },
                             help: "Classifier-free guidance. Higher = more contrast.")
                            .guide(.hdr)
                    }
                    .frame(maxWidth: .infinity)
                    HStack(alignment: .center, spacing: 10) {
                        ToggleSwitch(label: "Keep Tones", off: "Off", on: "On", isOn: $model.settings.keepTones,
                                     help: "On: the result keeps your photo's own light and colour; only the new detail is added")
                            .guide(.keepTones)
                        EngravedLabel(model.settings.keepTones ? "On · photo's own light & colour"
                                                               : "Off · the AI's own light", size: 10)
                            .lineLimit(1).minimumScaleFactor(0.8)
                        Spacer(minLength: 0)
                    }
                }

                Module(title: "Likeness") {
                    HStack(alignment: .center, spacing: 8) {
                        Knob(label: "Likeness", value: $model.settings.likeness, range: 0...1, step: 0.05,
                             display: { $0 <= 0 ? "OFF" : "\(Int(($0 * 100).rounded()))%" },
                             help: "Keeps people and things looking like your photo while the AI adds detail (IP-Adapter Plus).")
                            .guide(.likeness)
                            .disabled(model.settings.mode == "instant")
                        VStack(alignment: .leading, spacing: 4) {
                            EngravedLabel(model.settings.mode == "instant" ? "Not used by INSTANT" :
                                            (model.settings.likeness > 0 ? "On · keeps the photo's look" : "Off"), size: 10)
                            EngravedLabel("IP-Adapter Plus · 1.3 GB on first use", size: 10)
                            #if arch(x86_64)
                            EngravedLabel("Slower on Intel Macs", size: 10)
                            #endif
                        }
                        Spacer(minLength: 0)
                    }
                }

                Module(title: "Face Detail") {
                    let instant = model.settings.mode == "instant"
                    HStack(alignment: .center, spacing: 6) {
                        ToggleSwitch(label: "Face Detail", off: "Off", on: "On", isOn: $model.settings.faceDetail,
                                     help: "Redraw each face close up with the same AI, so small faces get real detail")
                            .guide(.faceDetail)
                        Knob(label: "Strength", value: $model.settings.faceDetailStrength, range: 0.1...0.6, step: 0.05,
                             display: { "\(Int(($0 * 100).rounded()))%" },
                             help: "How much the close-up may change the face. 15–20% sharpens and keeps the person; higher redraws more.")
                            .guide(.faceDetailStrength)
                        ToggleSwitch(label: "Hair", off: "Off", on: "On", isOn: $model.settings.faceDetailHair,
                                     help: "Also redraw the hair around the face")
                            .guide(.faceDetailHair)
                    }
                    .frame(maxWidth: .infinity)
                    .disabled(instant)
                    .opacity(model.settings.faceDetail && !instant ? 1 : 0.75)
                    EngravedLabel(instant ? "Not used by INSTANT · pick an AI preset"
                                  : (model.settings.faceDetail ? "On · each face redrawn close up, then blended in"
                                                               : "Off · no download needed"), size: 10)
                }

                Module(title: "Face Fix") {
                    HStack(alignment: .top, spacing: 2) {
                        VStack(spacing: 14) {
                            ToggleSwitch(label: "Face Fix", off: "Off", on: "On", isOn: $model.faceFix,
                                         help: "Restore faces after the enhance (first use downloads the face model)")
                                .guide(.faceFix)
                            VStack(spacing: 5) {
                                EngravedLabel("Model", size: 11)
                                HStack(spacing: 4) {
                                    ForEach([("RESTORE", "restoreformer"), ("CODE", "codeformer")], id: \.1) { m in
                                        Button { model.faceModel = m.1 } label: {
                                            Text(m.0).font(Console.label(11)).tracking(0.6).frame(width: 56)
                                        }
                                        .buttonStyle(PresetKeyStyle(lit: model.faceModel == m.1))
                                    }
                                }
                            }
                            .guide(.faceModel)
                        }
                        Knob(label: "Strength", value: $model.faceStrength, range: 0.1...1, step: 0.05,
                             display: { "\(Int(($0 * 100).rounded()))%" },
                             help: "How much of the restored face is blended in.")
                            .guide(.faceStrength)
                        Knob(label: "Fidelity", value: $model.faceFidelity, range: 0...1, step: 0.05,
                             display: { String(format: "%.2f", $0) },
                             help: "CodeFormer only: higher keeps the person's own features, lower gives a cleaner face.")
                            .disabled(model.faceModel != "codeformer")
                            .guide(.faceFidelity)
                    }
                    .frame(maxWidth: .infinity)
                    .opacity(model.faceFix ? 1 : 0.75)
                    VStack(alignment: .leading, spacing: 5) {
                        EngravedLabel("CodeFormer presets", size: 11)
                        LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 4), count: 3), spacing: 5) {
                            ForEach(FacePreset.all) { fp in
                                let lit = model.faceFix && model.faceModel == "codeformer"
                                    && abs(model.faceFidelity - fp.fidelity) < 0.001 && abs(model.faceStrength - fp.strength) < 0.001
                                Button {
                                    model.faceFix = true
                                    model.faceModel = "codeformer"
                                    model.faceFidelity = fp.fidelity
                                    model.faceStrength = fp.strength
                                } label: {
                                    Text(fp.label).font(Console.label(10)).tracking(0.4).lineLimit(1)
                                        .minimumScaleFactor(0.8).frame(maxWidth: .infinity)
                                }
                                .buttonStyle(PresetKeyStyle(lit: lit))
                                .help(fp.tip)
                            }
                        }
                    }
                    .guide(.facePreset)
                    EngravedLabel(model.faceModel == "codeformer"
                                  ? "CodeFormer · non-commercial licence · 360 MB on first use"
                                  : "RestoreFormer · 280 MB on first use", size: 10)
                }

                Module(title: "Advanced") {
                    HStack(alignment: .top, spacing: 2) {
                        Knob(label: "Steps", value: $model.settings.steps, range: 1...30, step: 1,
                             display: { String(format: "%.0f", $0) },
                             help: "Inference steps (only Creativity × Steps actually run).")
                            .guide(.steps)
                        Knob(label: "Decay", value: $model.settings.resemblanceDecay, range: 0.5...1, step: 0.025,
                             display: { String(format: "%.3f", $0) },
                             help: "Resemblance decay across the UNet blocks.")
                            .guide(.decay)
                        VStack(spacing: 14) {
                            ToggleSwitch(label: "Solver", off: "DDIM", on: "DPM", isOn: dpm)
                                .guide(.solver)
                            VStack(spacing: 4) {
                                EngravedLabel("Seed", size: 12)
                                HStack(spacing: 4) {
                                    TextField("", value: $model.settings.seed, format: .number.grouping(.never))
                                        .textFieldStyle(.plain)
                                        .font(Console.mono(11))
                                        .foregroundStyle(Console.amber)
                                        .multilineTextAlignment(.center)
                                        .frame(width: 46)
                                        .padding(.vertical, 3)
                                        .background(RoundedRectangle(cornerRadius: 3).fill(Console.ledGlass))
                                    Button("RND") { model.settings.seed = Double(Int.random(in: 0...10_000)) }
                                        .buttonStyle(MetalButtonStyle())
                                        .help("Random seed")
                                }
                            }
                            .guide(.seed)
                        }
                        .frame(maxWidth: .infinity)
                    }
                    TapeField(label: "Prompt", text: $model.settings.prompt, tilt: -0.5)
                        .guide(.prompt)
                    TapeField(label: "Negative", text: $model.settings.negativePrompt, tilt: 0.4)
                        .guide(.negative)
                    HStack {
                        Spacer()
                        Button("RESET") {
                            let profile = model.settings.profile
                            model.settings = EnhanceSettings()
                            model.settings.profile = profile
                        }
                        .buttonStyle(MetalButtonStyle())
                        .help("Reset knobs to the Space's defaults")
                    }
                }
            }
            .padding(.vertical, 4)
        }
        .scrollIndicators(.never)
    }
}

/// Lightroom/Camera Raw-style slider in console dress: drag (or click) the track, double-click to reset to 0.
struct Fader: View {
    let label: String
    @Binding var value: Double
    var range: ClosedRange<Double> = -1...1
    var format: (Double) -> String = { String(format: "%+.0f", $0 * 100) }
    var home: Double = 0          // reset value (double-click) and where the lit span starts
    var showHome = false          // show the value even when it's at home (hue sliders)
    @State private var dragStart: Double?

    private var norm: Double { (value - range.lowerBound) / (range.upperBound - range.lowerBound) }
    private var zeroNorm: Double { (home - range.lowerBound) / (range.upperBound - range.lowerBound) }
    private var atHome: Bool { value == home }

    var body: some View {
        VStack(spacing: 3) {
            HStack {
                Text(label).font(.system(size: 12)).foregroundStyle(Color(white: atHome ? 0.75 : 0.95))
                Spacer()
                Text(atHome && !showHome ? "0" : format(value))
                    .font(Console.mono(11))
                    .foregroundStyle(atHome ? Color(white: 0.55) : Console.amber)
                    .frame(width: 44, alignment: .trailing)
            }
            GeometryReader { g in
                let w = g.size.width
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.black.opacity(0.6)).frame(height: 4)
                        .overlay(Capsule().strokeBorder(Color.white.opacity(0.08), lineWidth: 0.5))
                    // lit span from the zero point to the value
                    Capsule().fill(Console.amber.opacity(0.75))
                        .frame(width: abs(norm - zeroNorm) * w, height: 4)
                        .offset(x: min(norm, zeroNorm) * w)
                    Rectangle().fill(Color.white.opacity(0.35)).frame(width: 1, height: 8)
                        .offset(x: zeroNorm * w)
                    Circle()
                        .fill(RadialGradient(colors: [Color(white: 0.97), Color(white: 0.6)],
                                             center: UnitPoint(x: 0.35, y: 0.3), startRadius: 0, endRadius: 8))
                        .overlay(Circle().strokeBorder(Color.black.opacity(0.5), lineWidth: 0.6))
                        .frame(width: 13, height: 13)
                        .shadow(color: .black.opacity(0.5), radius: 1.5, y: 1)
                        .offset(x: norm * w - 6.5)
                }
                .frame(height: 16)
                .contentShape(Rectangle())
                .gesture(DragGesture(minimumDistance: 0).onChanged { d in
                    let t = min(1, max(0, d.location.x / max(w, 1)))
                    var v = range.lowerBound + t * (range.upperBound - range.lowerBound)
                    if abs(v - home) < (range.upperBound - range.lowerBound) * 0.015 { v = home } // snap home
                    value = (v * 100).rounded() / 100
                })
                .simultaneousGesture(TapGesture(count: 2).onEnded { value = home })
            }
            .frame(height: 16)
        }
        .help("Drag to adjust · double-click to reset")
    }
}

/// Camera Raw-style section header: ▼ Title … RESET
struct AdjustSection<Content: View>: View {
    let title: String
    @Binding var expanded: Bool
    let active: Bool
    var onReset: () -> Void
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Button { withAnimation(.easeOut(duration: 0.18)) { expanded.toggle() } } label: {
                    HStack(spacing: 6) {
                        Image(systemName: expanded ? "arrowtriangle.down.fill" : "arrowtriangle.right.fill")
                            .font(.system(size: 8)).foregroundStyle(Color(white: 0.7))
                        Text(title.uppercased()).font(Console.label(14)).tracking(1)
                            .foregroundStyle(active ? Console.amber : Color(white: 0.85))
                        Spacer()
                    }
                    .frame(height: 20)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                // Always present (dimmed when there's nothing to reset) so the box never changes height.
                Button("RESET") { onReset() }
                    .buttonStyle(MetalButtonStyle())
                    .disabled(!active)
            }
            .frame(height: 26)
            if expanded { content }
        }
        .padding(.horizontal, 9).padding(.vertical, 8)
        .background(RoundedRectangle(cornerRadius: 6).fill(Color.black.opacity(0.2)))
        .overlay(RoundedRectangle(cornerRadius: 6)
            .strokeBorder(active ? Console.amber.opacity(0.45) : Color.white.opacity(0.05), lineWidth: 1))
    }
}

/// Color Efex-style filter list beside the picture. Click a filter's name to apply it and open its
/// controls; the checkbox turns it on or off. Knobs only adjust a filter, they never switch it on.
struct FilterPanel: View {
    @EnvironmentObject var model: AppModel
    @State private var open: String? = nil
    @State private var lightOpen = true
    @State private var colorOpen = false
    @State private var presenceOpen = false
    @State private var mixerOpen = false
    @State private var gradingOpen = false
    @State private var mixBand = 0
    private static let bands: [(String, Color)] = [
        ("Red", Color(red: 0.9, green: 0.2, blue: 0.2)), ("Orange", Color(red: 0.95, green: 0.55, blue: 0.15)),
        ("Yellow", Color(red: 0.95, green: 0.85, blue: 0.2)), ("Green", Color(red: 0.3, green: 0.75, blue: 0.3)),
        ("Aqua", Color(red: 0.2, green: 0.8, blue: 0.8)), ("Blue", Color(red: 0.25, green: 0.45, blue: 0.95)),
        ("Purple", Color(red: 0.6, green: 0.35, blue: 0.9)), ("Magenta", Color(red: 0.9, green: 0.3, blue: 0.75)),
    ]

    private func band(_ kp: WritableKeyPath<DetailFilters, [Double]>) -> Binding<Double> {
        let b = mixBand
        return Binding(get: { model.filters[keyPath: kp][b] }, set: { model.filters[keyPath: kp][b] = $0 })
    }
    private func deg(_ x: Double) -> String { String(format: "%.0f°", x) }

    private func v(_ kp: WritableKeyPath<DetailFilters, Double>) -> Binding<Double> {
        Binding(get: { model.filters[keyPath: kp] }, set: { model.filters[keyPath: kp] = $0 })
    }
    private func pct(_ x: Double) -> String { String(format: "%+.0f%%", x * 100) }
    private func pct0(_ x: Double) -> String { String(format: "%.0f%%", x * 100) }
    private var activeCount: Int {
        [model.filters.lightOn, model.filters.colorOn, model.filters.presenceOn, model.filters.mixerOn,
         model.filters.gradingOn, model.filters.bleachOn, model.filters.darkOn, model.filters.extractOn,
         model.filters.grainOn, model.filters.bloomOn, model.filters.glowOn, model.filters.blurOn,
         model.filters.skinOn].filter { $0 }.count
    }
    private let grid = [GridItem(.flexible(), spacing: 2), GridItem(.flexible(), spacing: 2)]

    private func move(_ id: String, _ delta: Int) {
        var order = model.filters.order
        guard let i = order.firstIndex(of: id) else { return }
        let j = i + delta
        guard j >= 0, j < order.count else { return }
        order.swapAt(i, j)
        model.filters.order = order
    }

    @ViewBuilder
    private func filterRow(_ id: String, pos: Int, total: Int) -> some View {
        switch id {
        case "bleach":
            FilterRow(id: "bleach", title: "Bleach Bypass", isOn: $model.filters.bleachOn, open: $open,
                      guide: .bleach, position: pos, count: total, onMove: { move("bleach", $0) }, onReset: {
                model.filters.bbBrightness = 0; model.filters.bbSaturation = 0
                model.filters.bbContrast = 0; model.filters.bbLocalContrast = 0
            }) {
                LazyVGrid(columns: grid, spacing: 6) {
                    Knob(label: "Brightness", value: v(\.bbBrightness), range: -1...1, step: 0.05, display: pct, size: 0.82)
                    Knob(label: "Saturation", value: v(\.bbSaturation), range: -1...1, step: 0.05, display: pct, size: 0.82)
                    Knob(label: "Contrast", value: v(\.bbContrast), range: -1...1, step: 0.05, display: pct, size: 0.82)
                    Knob(label: "Local Contrast", value: v(\.bbLocalContrast), range: 0...1, step: 0.05, display: pct0, size: 0.82)
                }
            }
        case "dark":
            FilterRow(id: "dark", title: "Dark Contrasts", isOn: $model.filters.darkOn, open: $open,
                      guide: .dark, position: pos, count: total, onMove: { move("dark", $0) }, onReset: {
                model.filters.dcStrength = 0.5; model.filters.dcBrightness = 0; model.filters.dcSaturation = 0
            }) {
                LazyVGrid(columns: grid, spacing: 6) {
                    Knob(label: "Detail Strength", value: v(\.dcStrength), range: 0...1, step: 0.05, display: pct0, size: 0.82)
                    Knob(label: "Brightness", value: v(\.dcBrightness), range: -1...1, step: 0.05, display: pct, size: 0.82)
                    Knob(label: "Saturation", value: v(\.dcSaturation), range: -1...1, step: 0.05, display: pct, size: 0.82)
                }
            }
        case "extract":
            FilterRow(id: "extract", title: "Detail Extractor", isOn: $model.filters.extractOn, open: $open,
                      guide: .extract, position: pos, count: total, onMove: { move("extract", $0) }, onReset: {
                model.filters.deAmount = 0.5; model.filters.deContrast = 0
                model.filters.deSaturation = 0; model.filters.deRadius = 1
            }) {
                LazyVGrid(columns: grid, spacing: 6) {
                    Knob(label: "Extraction", value: v(\.deAmount), range: 0...1, step: 0.05, display: pct0, size: 0.82)
                    Knob(label: "Contrast", value: v(\.deContrast), range: -1...1, step: 0.05, display: pct, size: 0.82)
                    Knob(label: "Saturation", value: v(\.deSaturation), range: -1...1, step: 0.05, display: pct, size: 0.82)
                    SteppedKnob(label: "Effect Radius",
                                options: [("F", "FINE", 0), ("N", "NORMAL", 1), ("C", "COARSE", 2)],
                                selection: $model.filters.deRadius, size: 0.74)
                }
            }
        case "bloom":
            FilterRow(id: "bloom", title: "Bloom", isOn: $model.filters.bloomOn, open: $open,
                      guide: .bloom, position: pos, count: total, onMove: { move("bloom", $0) }, onReset: {
                model.filters.blAmount = 0.6; model.filters.blTransition = 0.4
                model.filters.blSize = 0.4; model.filters.blBurn = 0
            }) {
                LazyVGrid(columns: grid, spacing: 6) {
                    Knob(label: "Amount", value: v(\.blAmount), range: 0...1, step: 0.05, display: pct0, size: 0.82)
                    Knob(label: "Transition", value: v(\.blTransition), range: 0...1, step: 0.05, display: pct0, size: 0.82)
                    Knob(label: "Size", value: v(\.blSize), range: 0...1, step: 0.05, display: pct0, size: 0.82)
                    Knob(label: "Burn", value: v(\.blBurn), range: 0...1, step: 0.05, display: pct0, size: 0.82)
                }
            }
        case "glow":
            FilterRow(id: "glow", title: "Glamour Glow", isOn: $model.filters.glowOn, open: $open,
                      guide: .glow, position: pos, count: total, onMove: { move("glow", $0) }, onReset: {
                model.filters.ggGlow = 0.5; model.filters.ggSaturation = 0; model.filters.ggWarmth = 0
            }) {
                LazyVGrid(columns: grid, spacing: 6) {
                    Knob(label: "Glow", value: v(\.ggGlow), range: 0...1, step: 0.05, display: pct0, size: 0.82)
                    Knob(label: "Saturation", value: v(\.ggSaturation), range: -1...1, step: 0.05, display: pct, size: 0.82)
                    Knob(label: "Warmth", value: v(\.ggWarmth), range: -1...1, step: 0.05, display: pct, size: 0.82)
                }
            }
        case "blur":
            FilterRow(id: "blur", title: "Blur", isOn: $model.filters.blurOn, open: $open,
                      guide: .blurFilter, position: pos, count: total, onMove: { move("blur", $0) }, onReset: {
                model.filters.bzSize = 0.3; model.filters.bzMix = 1; model.filters.bzKind = 0
            }) {
                LazyVGrid(columns: grid, spacing: 6) {
                    Knob(label: "Size", value: v(\.bzSize), range: 0...1, step: 0.05, display: pct0, size: 0.82)
                    Knob(label: "Mix", value: v(\.bzMix), range: 0...1, step: 0.05, display: pct0, size: 0.82)
                    ToggleSwitch(label: "Kind", off: "Blur", on: "Soft", isOn: Binding(
                        get: { model.filters.bzKind == 1 }, set: { model.filters.bzKind = $0 ? 1 : 0 }),
                                 help: "Soft focus: the blur glows over the picture instead of smearing it")
                        .frame(maxHeight: .infinity)
                }
            }
        case "skin":
            FilterRow(id: "skin", title: "Skin Texture", isOn: $model.filters.skinOn, open: $open,
                      guide: .skin, position: pos, count: total, onMove: { move("skin", $0) }, onReset: {
                model.filters.skAmount = 0.5; model.filters.skSize = 0.35; model.filters.skFine = 0.5
                model.filters.skFreckles = 0; model.filters.skSeed = 0
            }) {
                LazyVGrid(columns: grid, spacing: 6) {
                    Knob(label: "Amount", value: v(\.skAmount), range: 0...1.5, step: 0.05, display: pct0, size: 0.82)
                    Knob(label: "Pore Size", value: v(\.skSize), range: 0...1, step: 0.05, display: pct0, size: 0.82)
                    Knob(label: "Fine Texture", value: v(\.skFine), range: 0...1, step: 0.05, display: pct0, size: 0.82)
                    Knob(label: "Freckles", value: v(\.skFreckles), range: 0...1, step: 0.05, display: pct0, size: 0.82)
                }
                HStack {
                    EngravedLabel("Pattern \(Int(model.filters.skSeed))", size: 10)
                    Spacer()
                    Button("NEW PATTERN") { model.filters.skSeed = Double(Int.random(in: 1...999)) }
                        .buttonStyle(MetalButtonStyle())
                        .help("A different random layout of pores")
                }
            }
        default:
            FilterRow(id: "grain", title: "Film Grain", isOn: $model.filters.grainOn, open: $open,
                      guide: .grain, position: pos, count: total, onMove: { move("grain", $0) }, onReset: {
                model.filters.grainAmount = 0.5; model.filters.grainSize = 0.3
                model.filters.grainSoftness = 0.3; model.filters.grainColor = false
            }) {
                LazyVGrid(columns: grid, spacing: 6) {
                    Knob(label: "Intensity", value: v(\.grainAmount), range: 0...1, step: 0.05, display: pct0, size: 0.82)
                    Knob(label: "Grain Size", value: v(\.grainSize), range: 0...1, step: 0.05, display: pct0, size: 0.82)
                    Knob(label: "Softness", value: v(\.grainSoftness), range: 0...1, step: 0.05, display: pct0, size: 0.82)
                    ToggleSwitch(label: "Grain", off: "Mono", on: "Color", isOn: $model.filters.grainColor)
                        .frame(maxHeight: .infinity)
                }
            }
        }
    }

    var body: some View {
        if model.adjustCollapsed { collapsedTab } else { panel }
    }

    /// Slim tab left on screen when the panel is folded away.
    private var collapsedTab: some View {
        VStack(spacing: 12) {
            Button { withAnimation(.easeOut(duration: 0.2)) { model.adjustCollapsed = false } } label: {
                Image(systemName: "chevron.right").font(.system(size: 11, weight: .bold))
                    .frame(width: 24, height: 24)
                    .background(RoundedRectangle(cornerRadius: 4).fill(Color.black.opacity(0.35)))
                    .foregroundStyle(Color(white: 0.85))
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Open the Adjust panel")
            LEDReadout(text: "\(activeCount)", size: 11)
            Text("ADJUST").font(Console.label(13)).tracking(2).engraved()
                .fixedSize().rotationEffect(.degrees(-90)).frame(width: 20, height: 70)
            Spacer()
        }
        .padding(.top, 12)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .contentShape(Rectangle())
        .onTapGesture { withAnimation(.easeOut(duration: 0.2)) { model.adjustCollapsed = false } }
        .recessed(corner: 10)
    }

    private var panel: some View {
        VStack(spacing: 0) {
            // Pinned header: never scrolls.
            VStack(spacing: 9) {
                HStack(spacing: 8) {
                    Button { withAnimation(.easeOut(duration: 0.2)) { model.adjustCollapsed = true } } label: {
                        Image(systemName: "chevron.left").font(.system(size: 11, weight: .bold))
                            .frame(width: 22, height: 22)
                            .background(RoundedRectangle(cornerRadius: 4).fill(Color.black.opacity(0.35)))
                            .foregroundStyle(Color(white: 0.85))
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .help("Fold the Adjust panel away")
                    EngravedLabel("Adjust", size: 14, onDark: true)
                    LEDReadout(text: "\(activeCount)", size: 11)
                    Spacer()
                    Button("RESET") { model.filters = DetailFilters() }
                        .buttonStyle(MetalButtonStyle())
                        .help("Reset every adjustment, turn every filter off and restore the filter order")
                }
                HStack(spacing: 6) {
                    EngravedLabel("Apply", size: 11, onDark: true)
                    Spacer()
                    ForEach([("AFTER AI", "after"), ("BEFORE AI", "before")], id: \.1) { opt in
                        Button { model.filterStage = opt.1 } label: {
                            Text(opt.0).font(Console.label(11)).tracking(0.6).frame(width: 74)
                        }
                        .buttonStyle(PresetKeyStyle(lit: model.filterStage == opt.1))
                    }
                }
                .guide(.filterStage)
            }
            .padding(.horizontal, 12).padding(.top, 12).padding(.bottom, 10)
            VStack(spacing: 0) {
                Rectangle().fill(Color.black.opacity(0.5)).frame(height: 1)
                Rectangle().fill(Color.white.opacity(0.08)).frame(height: 1)
            }
            .padding(.bottom, 6)
            ScrollView {
                VStack(spacing: 6) {
                    AdjustSection(title: "Light", expanded: $lightOpen, active: model.filters.lightOn,
                                  onReset: { model.filters.resetLight() }) {
                        Fader(label: "Exposure", value: v(\.exposure), range: -2...2,
                              format: { String(format: "%+.2f", $0) })
                        Fader(label: "Contrast", value: v(\.contrast))
                        Fader(label: "Highlights", value: v(\.highlights))
                        Fader(label: "Shadows", value: v(\.shadows))
                        Fader(label: "Whites", value: v(\.whites))
                        Fader(label: "Blacks", value: v(\.blacks))
                    }
                    .guide(.light)
                    AdjustSection(title: "Color", expanded: $colorOpen, active: model.filters.colorOn,
                                  onReset: { model.filters.resetColor() }) {
                        Fader(label: "Temperature", value: v(\.temperature))
                        Fader(label: "Tint", value: v(\.tint))
                        Fader(label: "Vibrance", value: v(\.vibrance))
                        Fader(label: "Saturation", value: v(\.saturation))
                    }
                    .guide(.color)
                    AdjustSection(title: "Presence", expanded: $presenceOpen, active: model.filters.presenceOn,
                                  onReset: { model.filters.resetPresence() }) {
                        Fader(label: "Texture", value: v(\.texture))
                        Fader(label: "Clarity", value: v(\.clarity))
                        Fader(label: "Dehaze", value: v(\.dehaze))
                        Fader(label: "Sharpen", value: v(\.sharpen), range: 0...1,
                              format: { String(format: "%.0f", $0 * 100) })
                        Fader(label: "Vignette", value: v(\.vignette))
                    }
                    .guide(.presence)
                    AdjustSection(title: "Color Mixer", expanded: $mixerOpen, active: model.filters.mixerOn,
                                  onReset: { model.filters.resetMixer() }) {
                        HStack(spacing: 4) {
                            ForEach(0..<8, id: \.self) { i in
                                let b = Self.bands[i]
                                let used = model.filters.mixHue[i] != 0 || model.filters.mixSat[i] != 0 || model.filters.mixLum[i] != 0
                                Button { mixBand = i } label: {
                                    Circle().fill(b.1)
                                        .frame(width: 18, height: 18)
                                        .overlay(Circle().strokeBorder(mixBand == i ? Color.white : Color.black.opacity(0.5),
                                                                       lineWidth: mixBand == i ? 2 : 1))
                                        .overlay(alignment: .topTrailing) {
                                            if used { Circle().fill(Console.amber).frame(width: 6, height: 6) }
                                        }
                                        .contentShape(Circle())
                                }
                                .buttonStyle(.plain)
                                .help(b.0 + (used ? " (changed)" : ""))
                            }
                        }
                        .frame(maxWidth: .infinity)
                        EngravedLabel(Self.bands[mixBand].0, size: 11, onDark: true)
                        Fader(label: "Hue", value: band(\.mixHue), format: { String(format: "%+.0f°", $0 * 30) })
                        Fader(label: "Saturation", value: band(\.mixSat))
                        Fader(label: "Luminance", value: band(\.mixLum))
                    }
                    .guide(.mixer)
                    AdjustSection(title: "Color Grading", expanded: $gradingOpen, active: model.filters.gradingOn,
                                  onReset: { model.filters.resetGrading() }) {
                        ForEach([("Shadows", \DetailFilters.cgShHue, \DetailFilters.cgShSat, 220.0),
                                 ("Midtones", \DetailFilters.cgMidHue, \DetailFilters.cgMidSat, 30.0),
                                 ("Highlights", \DetailFilters.cgHiHue, \DetailFilters.cgHiSat, 40.0)], id: \.0) { row in
                            HStack(spacing: 6) {
                                Circle().fill(Color(hue: model.filters[keyPath: row.1] / 360, saturation: 0.8, brightness: 0.9))
                                    .frame(width: 10, height: 10)
                                    .opacity(model.filters[keyPath: row.2] > 0 ? 1 : 0.35)
                                EngravedLabel(row.0, size: 11, onDark: true)
                            }
                            Fader(label: "Hue", value: v(row.1), range: 0...360, format: deg, home: row.3, showHome: true)
                            Fader(label: "Amount", value: v(row.2), range: 0...1,
                                  format: { String(format: "%.0f", $0 * 100) })
                        }
                        Fader(label: "Balance", value: v(\.cgBalance))
                    }
                    .guide(.grading)
                    HStack(spacing: 8) {
                        EngravedLabel("Detail", size: 11, onDark: true)
                        VStack(spacing: 0) {
                            Rectangle().fill(Color.black.opacity(0.35)).frame(height: 1)
                            Rectangle().fill(Color.white.opacity(0.1)).frame(height: 1)
                        }
                    }
                    .padding(.horizontal, 12).padding(.bottom, 6)
                    .padding(.top, 6)
                    Text("Applied top to bottom · ▲▼ to restack")
                        .font(Console.mono(9)).foregroundStyle(Color(white: 0.5))
                        .frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 4)
                    ForEach(Array(model.filters.order.enumerated()), id: \.element) { idx, fid in
                        filterRow(fid, pos: idx + 1, total: model.filters.order.count)
                    }
                }
                .padding(.horizontal, 10).padding(.bottom, 10)
            }
            .scrollIndicators(.never)
        }
        .recessed(corner: 10)
    }
}

/// One filter in the list: checkbox (on/off), name (click to apply and open), and its controls when open.
struct FilterRow<Content: View>: View {
    let id: String
    let title: String
    @Binding var isOn: Bool
    @Binding var open: String?
    let guide: Guide
    var position = 1
    var count = 1
    var onMove: (Int) -> Void = { _ in }
    var onReset: () -> Void
    @ViewBuilder var content: Content

    private var expanded: Bool { open == id }

    private func moveButton(_ icon: String, _ delta: Int, enabled: Bool, help: String) -> some View {
        Button { withAnimation(.easeOut(duration: 0.2)) { onMove(delta) } } label: {
            Image(systemName: icon).font(.system(size: 9, weight: .bold))
                .frame(width: 18, height: 18)
                .background(RoundedRectangle(cornerRadius: 3).fill(Color.black.opacity(0.35)))
                .foregroundStyle(Color(white: enabled ? 0.85 : 0.3))
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
        .help(help)
    }

    var body: some View {
        VStack(spacing: 8) {
            HStack(spacing: 9) {
                Button { withAnimation(.easeOut(duration: 0.15)) { isOn.toggle() } } label: {
                    RoundedRectangle(cornerRadius: 3)
                        .fill(isOn ? Console.amber : Color.black.opacity(0.45))
                        .overlay(RoundedRectangle(cornerRadius: 3).strokeBorder(Color.white.opacity(isOn ? 0 : 0.3), lineWidth: 1))
                        .overlay { if isOn { Image(systemName: "checkmark").font(.system(size: 10, weight: .heavy)).foregroundStyle(.black) } }
                        .frame(width: 17, height: 17)
                        .shadow(color: isOn ? Console.amber.opacity(0.7) : .clear, radius: 4)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help(isOn ? "Turn \(title) off" : "Turn \(title) on")
                Button {
                    withAnimation(.easeOut(duration: 0.18)) {
                        if expanded {
                            open = nil
                        } else {
                            open = id
                            isOn = true // like Color Efex: picking a filter applies it
                        }
                    }
                } label: {
                    HStack {
                        Text("\(position)")
                            .font(Console.mono(10)).foregroundStyle(Color(white: 0.5))
                        Text(title.uppercased())
                            .font(Console.label(14)).tracking(1)
                            .lineLimit(1).minimumScaleFactor(0.8)
                            .foregroundStyle(isOn ? Console.amber : Color(white: 0.85))
                        Spacer()
                        Image(systemName: expanded ? "chevron.up" : "chevron.down")
                            .font(.system(size: 10, weight: .bold))
                            .foregroundStyle(Color(white: 0.6))
                    }
                    .frame(height: 22)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help(expanded ? "Hide controls" : "Apply \(title) and show its controls")
                moveButton("arrow.up", -1, enabled: position > 1, help: "Move \(title) earlier in the stack")
                moveButton("arrow.down", 1, enabled: position < count, help: "Move \(title) later in the stack")
            }
            if expanded {
                content
                    .opacity(isOn ? 1 : 0.5)
                HStack {
                    Text(isOn ? "ON" : "OFF · tick box to apply")
                        .font(Console.mono(10)).foregroundStyle(isOn ? Console.amber : Color(white: 0.55))
                        .lineLimit(1).minimumScaleFactor(0.8)
                    Spacer()
                    Button("RESET") { onReset() }
                        .buttonStyle(MetalButtonStyle())
                        .help("Reset \(title)'s knobs")
                }
            }
        }
        .padding(.horizontal, 9).padding(.vertical, 8)
        .background(RoundedRectangle(cornerRadius: 6).fill(expanded ? Color.white.opacity(0.05) : Color.black.opacity(0.2)))
        .overlay(RoundedRectangle(cornerRadius: 6)
            .strokeBorder(isOn ? Console.amber.opacity(0.45) : Color.white.opacity(0.05), lineWidth: 1))
        .guide(guide)
    }
}

/// Illuminated monitor-view key in the meter bridge.
struct ViewKey: View {
    let icon: String
    let title: String
    let lit: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 3) {
                Image(systemName: icon).font(.system(size: 15, weight: .semibold))
                Text(title).font(Console.label(10)).tracking(0.8)
            }
            .frame(width: 66, height: 40)
        }
        .buttonStyle(PresetKeyStyle(lit: lit))
    }
}

/// Analog meter; the needle shows progress with a little mechanical overshoot.
struct VUMeter: View {
    let value: Double
    var title = "PROGRESS %"

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width, h = geo.size.height
            let pivot = CGPoint(x: w / 2, y: h * 1.08)
            let R = h * 0.86
            let v = min(1, max(0, value))
            ZStack {
                RoundedRectangle(cornerRadius: 4)
                    .fill(LinearGradient(colors: [Color(red: 0.99, green: 0.94, blue: 0.78),
                                                  Color(red: 0.93, green: 0.84, blue: 0.60)],
                                         startPoint: .top, endPoint: .bottom))
                Canvas { ctx, _ in
                    func pt(_ frac: Double, _ r: CGFloat) -> CGPoint {
                        let a = (-48 + 96 * frac) * .pi / 180
                        return CGPoint(x: pivot.x + CGFloat(sin(a)) * r, y: pivot.y - CGFloat(cos(a)) * r)
                    }
                    var scale = Path()
                    scale.move(to: pt(0, R))
                    for i in 1...80 { scale.addLine(to: pt(Double(i) / 100, R)) }
                    ctx.stroke(scale, with: .color(.black.opacity(0.85)), lineWidth: 1.4)
                    var red = Path()
                    red.move(to: pt(0.8, R + 2))
                    for i in 81...100 { red.addLine(to: pt(Double(i) / 100, R + 2)) }
                    ctx.stroke(red, with: .color(Color(red: 0.8, green: 0.1, blue: 0.08)), lineWidth: 5)
                    for i in 0...20 {
                        let f = Double(i) / 20, major = i % 2 == 0
                        var t = Path()
                        t.move(to: pt(f, R))
                        t.addLine(to: pt(f, R + (major ? 9 : 5)))
                        ctx.stroke(t, with: .color(.black.opacity(0.85)), lineWidth: major ? 1.3 : 0.8)
                        if i % 4 == 0 {
                            ctx.draw(Text("\(i * 5)").font(.system(size: 9, weight: .semibold)).foregroundColor(.black),
                                     at: pt(f, R + 17))
                        }
                    }
                    ctx.draw(Text(title).font(Console.label(13)).foregroundColor(.black.opacity(0.8)),
                             at: CGPoint(x: w / 2, y: h * 0.62))
                    ctx.draw(Text("PROTOART").font(Console.label(8)).foregroundColor(.black.opacity(0.45)),
                             at: CGPoint(x: w / 2, y: h * 0.76))
                }
                Capsule()
                    .fill(Color.black.opacity(0.85))
                    .frame(width: 1.8, height: R + 6)
                    .position(x: pivot.x, y: pivot.y - (R + 6) / 2)
                    .rotationEffect(.degrees(-48 + 96 * v), anchor: UnitPoint(x: pivot.x / w, y: pivot.y / h))
                    .shadow(color: .black.opacity(0.25), radius: 1, x: 1.5, y: 1.5)
                    .animation(.interpolatingSpring(stiffness: 45, damping: 6), value: v)
                // pivot housing
                Rectangle().fill(LinearGradient(colors: [Color(white: 0.2), Color(white: 0.05)],
                                                 startPoint: .top, endPoint: .bottom))
                    .frame(height: h * 0.14)
                    .frame(maxHeight: .infinity, alignment: .bottom)
                // glass
                LinearGradient(colors: [.white.opacity(0.35), .clear, .clear], startPoint: .topLeading, endPoint: .center)
            }
            .clipShape(RoundedRectangle(cornerRadius: 4))
            .overlay(RoundedRectangle(cornerRadius: 4).strokeBorder(Color.black.opacity(0.85), lineWidth: 2))
        }
    }
}

struct Readout: View {
    let label: String
    let value: String
    var width: CGFloat = 78

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            EngravedLabel(label, size: 10, onDark: true)
            LEDReadout(text: value, width: width, size: 14)
        }
    }
}

struct MeterBridge: View {
    @EnvironmentObject var model: AppModel

    private var needle: Double {
        if model.running, let cur = model.current { return cur.fraction ?? 0 }
        if !model.engineReady { return model.engineFraction ?? 0 }
        if model.selected?.state == .done { return 1 }
        return 0
    }

    var body: some View {
        HStack(spacing: 18) {
            VUMeter(value: needle, title: model.engineReady || model.running ? "PROGRESS %" : "LOADING %")
                .frame(width: 210, height: 108)
                .padding(5)
                .background(RoundedRectangle(cornerRadius: 7).fill(Color(white: 0.08)))
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 12) {
                    Readout(label: "Progress", value: percentText)
                    Readout(label: "Time Left", value: Console.clock(model.running ? model.current?.eta : model.setupETA))
                    if model.batchMode {
                        Readout(label: "Image", value: imageText)
                        Readout(label: "Queue Left", value: Console.clock(model.queueETA))
                    }
                    Spacer(minLength: 8)
                    HStack(spacing: 8) {
                        EngravedLabel("View", size: 10, onDark: true)
                        ViewKey(icon: "rectangle", title: "SINGLE", lit: model.viewMode == "single") { model.viewMode = "single" }
                        ViewKey(icon: "rectangle.lefthalf.filled", title: "SPLIT", lit: model.viewMode == "split") { model.viewMode = "split" }
                        ViewKey(icon: "rectangle.split.2x1", title: "SIDE", lit: model.viewMode == "side") { model.viewMode = "side" }
                    }
                    .guide(.viewMode)
                    Rectangle().fill(Color.black.opacity(0.7)).frame(width: 1, height: 40)
                        .overlay(Rectangle().fill(Color.white.opacity(0.1)).frame(width: 1).offset(x: 1))
                    ViewKey(icon: "slider.horizontal.3", title: "ADJUST", lit: model.showFilters) {
                        withAnimation(.easeOut(duration: 0.2)) { model.showFilters.toggle() }
                    }
                    .guide(.filtersKey)
                }
                VStack(alignment: .leading, spacing: 4) {
                    EngravedLabel("Status", size: 10, onDark: true)
                    Text(statusText.uppercased())
                        .font(Console.mono(12))
                        .foregroundStyle(Console.amber)
                        .shadow(color: Console.amber.opacity(0.7), radius: 3)
                        .lineLimit(1).truncationMode(.tail)
                        .padding(.horizontal, 8).padding(.vertical, 5)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(RoundedRectangle(cornerRadius: 3).fill(Console.ledGlass))
                        .overlay(RoundedRectangle(cornerRadius: 3).strokeBorder(Color.black.opacity(0.8), lineWidth: 1))
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            if model.running {
                Button { model.stop() } label: { Text("STOP") }
                    .buttonStyle(OutlinedButtonStyle.stop)
                    .help("Stop the queue (⌘.)")
            }
            Button { model.queueVariation() } label: { Text("+ QUEUE") }
                .buttonStyle(MetalButtonStyle())
                .keyboardShortcut(.return, modifiers: [.command, .shift])
                .disabled(model.selected == nil && model.items.isEmpty)
                .help("Queue this photo again with the current settings (⇧⌘↩). Change the knobs and queue again to try several versions, then press ENHANCE.")
                .guide(.addQueue)
            Button { model.start() } label: { Text(startTitle) }
                .buttonStyle(OutlinedButtonStyle.process)
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(!model.canStart)
                .help("Enhance (⌘↩)")
        }
        .padding(.horizontal, 16).padding(.vertical, 12)
        .recessed(corner: 10)
    }

    private var startTitle: String {
        guard model.batchMode else { return "ENHANCE" }
        let n = model.queuedCount
        return n > 0 ? "ENHANCE \(n)" : "ENHANCE"
    }

    private var percentText: String {
        if model.running, let f = model.current?.fraction { return "\(Int((f * 100).rounded()))%" }
        if !model.engineReady, let f = model.engineFraction { return "\(Int((f * 100).rounded()))%" }
        return "---"
    }

    private var imageText: String {
        guard model.running, model.runTotal > 0 else { return "\(model.queuedCount) Q" }
        return "\(min(model.runCompleted + 1, model.runTotal))/\(model.runTotal)"
    }

    private var statusText: String {
        if model.running, let cur = model.current { return "\(cur.name) — \(cur.message)" }
        if !model.engineReady { return model.engineStatus }
        if let item = model.selected {
            switch item.state {
            case .done:
                let size = item.outputSize.map(Fmt.size) ?? ""
                let saved = item.savedURL.map { " · saved: \($0.lastPathComponent)" } ?? ""
                let faces = item.facesFixed.map { $0 == 0 ? " · no faces found" : " · \($0) face\($0 == 1 ? "" : "s") fixed" } ?? ""
                return "Done in \(Fmt.duration(item.seconds ?? 0)) · \(size)\(faces)\(saved)"
            case .failed(let m): return "Fault: \(m)"
            case .stopped: return "Stopped"
            case .cancelled: return "Cancelled"
            default: break
            }
            if let saved = item.savedURL { return "Saved · \(saved.lastPathComponent)" }
            if item.inputSize != .zero {
                let out = model.settings.predictedOutput(for: item.inputSize, maxSide: model.maxSide)
                let est = model.estimate(item).map { " · ~\(Fmt.duration($0))" } ?? ""
                return "\(Fmt.size(item.inputSize)) → \(Fmt.size(out))\(est)"
            }
        }
        #if arch(x86_64)
        let hint = " · Intel Mac: TURBO or INSTANT recommended"
        #else
        let hint = ""
        #endif
        return (model.batchMode ? "Ready — \(model.queuedCount) queued" : "Ready — drop an image to begin") + hint
    }
}

struct ImageScreen: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        ZStack {
            LinearGradient(colors: [Color(red: 0.05, green: 0.06, blue: 0.07), Color(red: 0.02, green: 0.02, blue: 0.025)],
                           startPoint: .top, endPoint: .bottom)
            if let before = model.previewInput, let after = model.displayAfter {
                CompareView(before: before, after: after, mode: model.viewMode, split: $model.splitPosition,
                            zoom: $model.zoomScale, pan: $model.zoomPan).padding(14)
            } else if let img = model.previewInput {
                Image(nsImage: img).resizable().interpolation(.high).scaledToFit().padding(14)
                if let item = model.selected, item.state == .running {
                    RunningOverlay(item: item)
                }
            } else {
                VStack(spacing: 14) {
                    Text(model.batchMode ? "NO SIGNAL — DROP IMAGES OR FOLDERS" : "NO SIGNAL — DROP AN IMAGE")
                        .font(Console.mono(14))
                        .foregroundStyle(Console.amber)
                        .shadow(color: Console.amber.opacity(0.7), radius: 4)
                    Button(model.batchMode ? "ADD IMAGES" : "CHOOSE IMAGE") { model.openPanel() }
                        .buttonStyle(MetalButtonStyle())
                }
            }
            // screen glass
            LinearGradient(colors: [.white.opacity(0.06), .clear], startPoint: .top, endPoint: .center)
                .allowsHitTesting(false)
        }
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.black, lineWidth: 6).blur(radius: 6)
            .clipShape(RoundedRectangle(cornerRadius: 6)).allowsHitTesting(false))
        .padding(10)
        .recessed(corner: 10)
        .overlay(alignment: .topLeading) { Screw(angle: 15, size: 10).padding(4) }
        .overlay(alignment: .topTrailing) { Screw(angle: 70, size: 10).padding(4) }
        .overlay(alignment: .bottomLeading) { Screw(angle: 120, size: 10).padding(4) }
        .overlay(alignment: .bottomTrailing) { Screw(angle: 40, size: 10).padding(4) }
        .acceptsImageDrops { model.add($0) }
    }
}

struct RunningOverlay: View {
    let item: QueueItem

    var body: some View {
        VStack(spacing: 6) {
            Text(item.fraction.map { "\(Int(($0 * 100).rounded()))%" } ?? "…")
                .font(.system(size: 38, weight: .semibold, design: .monospaced))
            Text(item.eta.map { "\(Console.clock($0)) LEFT" } ?? "ESTIMATING…")
                .font(Console.mono(13))
        }
        .foregroundStyle(Console.amber)
        .shadow(color: Console.amber.opacity(0.8), radius: 6)
        .padding(.horizontal, 22).padding(.vertical, 14)
        .background(Console.ledGlass.opacity(0.85), in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Color.black, lineWidth: 1))
    }
}

struct QueueRow: View {
    let item: QueueItem
    let estimate: Double?
    var selected = false
    var onCancel: () -> Void = {}
    @State private var hovering = false

    var body: some View {
        HStack(spacing: 10) {
            ZStack {
                RoundedRectangle(cornerRadius: 3).fill(Color.black.opacity(0.6))
                if let t = item.thumbnail {
                    Image(nsImage: t).resizable().scaledToFill()
                }
            }
            .frame(width: 42, height: 42)
            .clipShape(RoundedRectangle(cornerRadius: 3))
            .overlay(RoundedRectangle(cornerRadius: 3).strokeBorder(Color.white.opacity(0.25), lineWidth: 1))
            .overlay(alignment: .bottomTrailing) { badge.font(.system(size: 13)).offset(x: 5, y: 5) }

            VStack(alignment: .leading, spacing: 3) {
                Text(item.name)
                    .font(Console.label(14)).tracking(0.5)
                    .foregroundStyle(Color(white: 0.9))
                    .lineLimit(1).truncationMode(.middle)
                if let job = item.job {
                    Text(job.label.uppercased())
                        .font(Console.mono(9.5))
                        .foregroundStyle(Console.amber.opacity(0.9))
                        .lineLimit(1).truncationMode(.tail)
                }
                if item.state == .running {
                    GeometryReader { g in
                        ZStack(alignment: .leading) {
                            Capsule().fill(Color.black.opacity(0.6))
                            Capsule().fill(Console.amber)
                                .frame(width: g.size.width * (item.fraction ?? 0))
                                .shadow(color: Console.amber.opacity(0.8), radius: 3)
                        }
                    }
                    .frame(height: 4)
                }
                Text(subtitle.uppercased())
                    .font(Console.mono(9.5))
                    .foregroundStyle(subtitleColor)
                    .lineLimit(2)
            }
            Spacer(minLength: 0)
            if item.state == .queued || item.state == .running {
                Button(action: onCancel) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 16))
                        .foregroundStyle(hovering ? Console.red : Color(white: 0.55))
                }
                .buttonStyle(.plain)
                .onHover { hovering = $0 }
                .help(item.state == .running ? "Cancel this image and move on" : "Cancel this image")
            }
        }
        .padding(7)
        .background(RoundedRectangle(cornerRadius: 5).fill(selected ? Color.white.opacity(0.1) : Color.black.opacity(0.28)))
        .overlay(alignment: .leading) {
            if selected {
                RoundedRectangle(cornerRadius: 1.5).fill(Console.amber).frame(width: 3)
                    .shadow(color: Console.amber, radius: 4).padding(.vertical, 6)
            }
        }
        .overlay(RoundedRectangle(cornerRadius: 5).strokeBorder(Color.black.opacity(0.5), lineWidth: 1))
        .contentShape(Rectangle())
    }

    @ViewBuilder private var badge: some View {
        switch item.state {
        case .done: Image(systemName: "checkmark.circle.fill").foregroundStyle(Console.green)
        case .failed: Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(Console.red)
        case .stopped: Image(systemName: "stop.circle.fill").foregroundStyle(Color(white: 0.6))
        case .cancelled: Image(systemName: "xmark.circle.fill").foregroundStyle(Color(white: 0.6))
        default: EmptyView()
        }
    }

    private var subtitle: String {
        switch item.state {
        case .queued:
            let size = item.inputSize == .zero ? "" : Fmt.size(item.inputSize)
            let est = estimate.map { " · ~\(Fmt.duration($0))" } ?? ""
            return "Queued" + (size.isEmpty ? "" : " · \(size)") + est
        case .running:
            let pct = item.fraction.map { "\(Int(($0 * 100).rounded()))%" } ?? "Starting"
            let eta = item.eta.map { " · \(Console.clock($0)) left" } ?? ""
            return pct + eta
        case .done:
            let size = item.outputSize.map { Fmt.size($0) + " · " } ?? ""
            return size + (item.seconds.map { Fmt.duration($0) } ?? "")
        case .stopped: return "Stopped"
        case .cancelled: return "Cancelled"
        case .failed(let m): return m
        }
    }

    private var subtitleColor: Color {
        switch item.state {
        case .failed: return Console.red
        case .running: return Console.amber
        case .done: return Console.green.opacity(0.85)
        default: return Color(white: 0.6)
        }
    }
}

struct QueuePanel: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                EngravedLabel("Queue", size: 14, onDark: true)
                LEDReadout(text: "\(model.items.count)", size: 11)
                Spacer()
                Button("ADD") { model.openPanel() }
                Button("CLEAR") { model.clearFinished() }.disabled(!model.hasFinished)
                    .help("Remove finished, cancelled and failed rows")
            }
            .buttonStyle(MetalButtonStyle())
            .padding(.horizontal, 12).padding(.top, 12).padding(.bottom, 8)
            if model.items.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "tray.and.arrow.down").font(.system(size: 26, weight: .light))
                    Text("DROP IMAGES / FOLDERS").font(Console.mono(11))
                }
                .foregroundStyle(Console.amber.opacity(0.8))
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(spacing: 6) {
                        ForEach(model.items) { item in
                            QueueRow(item: item,
                                     estimate: item.state == .queued ? model.estimate(item) : nil,
                                     selected: item.id == model.selectedID,
                                     onCancel: { model.cancel(item.id) })
                                .onTapGesture { model.selectedID = item.id }
                                .contextMenu {
                                    if item.state == .queued || item.state == .running {
                                        Button("Cancel") { model.cancel(item.id) }
                                    }
                                    if item.state != .running && item.state != .queued {
                                        Button("Enhance Again") { model.requeue(item.id) }
                                    }
                                    if item.job != nil {
                                        Button("Use These Settings") { model.useSettings(of: item.id) }
                                    }
                                    if item.outputURL != nil {
                                        Button("Show in Finder") { model.reveal(item.id) }
                                    }
                                    if item.state != .running {
                                        Button("Remove") { model.remove([item.id]) }
                                    }
                                }
                        }
                    }
                    .padding(.horizontal, 10).padding(.bottom, 10)
                }
                .scrollIndicators(.never)
            }
        }
        .recessed(corner: 10)
        .acceptsImageDrops { model.add($0) }
    }
}

struct ContentView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        VStack(spacing: 12) {
            HeaderStrip().frame(height: 66).padding(.top, 6)
            HStack(alignment: .top, spacing: 14) {
                if model.batchMode {
                    QueuePanel().frame(width: 260)
                }
                if model.showFilters {
                    FilterPanel().frame(width: model.adjustCollapsed ? 40 : 262)
                        .transition(.move(edge: .leading).combined(with: .opacity))
                }
                VStack(spacing: 10) {
                    ImageScreen()
                    GuideStrip()
                }
                ControlsPanel().frame(width: 336)
            }
            .padding(.horizontal, 30)
            MeterBridge()
                .padding(.horizontal, 30)
                .padding(.bottom, 14)
        }
        .frame(minWidth: (model.batchMode ? 1200 : 1060) + (model.showFilters && !model.adjustCollapsed ? 200 : 0), minHeight: 600)
        .background { Brushed().ignoresSafeArea() }
        .overlay { RackEars().ignoresSafeArea().allowsHitTesting(false) }
        .preferredColorScheme(.dark)
        .sheet(isPresented: $model.showHelp) { HelpSheet() }
        .alert("Something went wrong", isPresented: Binding(
            get: { model.errorMessage != nil }, set: { if !$0 { model.errorMessage = nil } })
        ) {
            Button("Show Log") { NSWorkspace.shared.open(Engine.logURL) }
            Button("OK", role: .cancel) {}
        } message: {
            Text(model.errorMessage ?? "")
        }
    }
}

/// Rack-ear screws and slots down both sides of the faceplate.
struct RackEars: View {
    var body: some View {
        GeometryReader { g in
            let h = g.size.height
            ForEach([0, 1], id: \.self) { side in
                let x: CGFloat = side == 0 ? 12 : g.size.width - 12
                ForEach([0.22, 0.78], id: \.self) { f in
                    ZStack {
                        Capsule().fill(Color.black.opacity(0.55)).frame(width: 10, height: 26)
                            .overlay(Capsule().strokeBorder(Color.white.opacity(0.35), lineWidth: 0.8).offset(y: 1))
                        Screw(angle: f * 300 + Double(side) * 47, size: 12)
                    }
                    .position(x: x, y: h * f)
                }
            }
        }
    }
}

// MARK: - How-to guide

struct HelpSheet: View {
    @Environment(\.dismiss) private var dismiss
    @State private var tab = 0
    private let tabs = ["Start", "Controls", "Filters & Views", "Saving & Batch", "Troubleshooting"]

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                EngravedLabel("How To", size: 18)
                Spacer()
                ForEach(tabs.indices, id: \.self) { i in
                    Button { tab = i } label: {
                        Text(tabs[i].uppercased()).font(Console.label(11)).tracking(0.6).padding(.horizontal, 6)
                    }
                    .buttonStyle(PresetKeyStyle(lit: tab == i))
                }
                Spacer()
                Button("CLOSE") { dismiss() }.buttonStyle(MetalButtonStyle()).keyboardShortcut(.cancelAction)
            }
            .padding(14)
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    ForEach(Array(HelpContent.pages[tab].enumerated()), id: \.offset) { _, section in
                        VStack(alignment: .leading, spacing: 6) {
                            Text(section.0.uppercased())
                                .font(Console.mono(12)).foregroundStyle(Console.amber)
                                .shadow(color: Console.amber.opacity(0.6), radius: 3)
                            ForEach(Array(section.1.enumerated()), id: \.offset) { _, line in
                                Text(LocalizedStringKey(line))
                                    .font(.system(size: 13))
                                    .foregroundStyle(Color(white: 0.86))
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                        }
                    }
                }
                .padding(18)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(RoundedRectangle(cornerRadius: 6).fill(Console.ledGlass))
            .padding([.horizontal, .bottom], 14)
        }
        .frame(width: 760, height: 560)
        .background(Brushed())
        .preferredColorScheme(.dark)
    }
}

enum HelpContent {
    typealias Section = (String, [String])
    static let pages: [[Section]] = [start, controls, filters, saving, troubleshooting]

    static let start: [Section] = [
        ("What it does", [
            "ProtoArt Beautify re-draws fine detail into your image with AI, entirely on this Mac (no uploads). It first sharpens the picture with a 4× upscaler, then repaints it tile by tile with Stable Diffusion, guided so the shapes stay the same.",
        ]),
        ("Quick start", [
            "1. **OPEN** (⌘O) or drag an image onto the screen.",
            "2. Pick a **preset**: **TURBO** for speed, **BALANCED** for the classic look, **QUALITY** for final prints.",
            "3. Press the blue **ENHANCE** button (⌘↩). The meter, percentage and time-left show progress.",
            "4. Compare the result with the view keys at the bottom (**SINGLE / SPLIT / SIDE**).",
            "5. The result is saved to your **Desktop** automatically (turn **Auto-save** off in the Save module to save by hand with **SAVE** ⌘S or **SAVE AS** ⇧⌘S).",
        ]),
        ("The first launch", [
            "The app downloads its AI models once (about 6 GB) and converts them to a smaller format (about 3 GB kept). After that it works offline. Each launch takes about 30–40 seconds to load the models; the READY light turns green when it's done.",
        ]),
        ("Privacy", [
            "Everything runs on this Mac: images are never uploaded and the app sends no analytics. The only network use is the one-time model download from huggingface.co. Temporary copies of your images are deleted when you quit.",
        ]),
        ("Lights", [
            "**READY** (green): models loaded. **PROCESS / LOADING** (amber, blinking): working. **FAULT** (red): something went wrong; the status line says what.",
        ]),
    ]

    static let controls: [Section] = [
        ("Presets", [
            "**INSTANT**: only the 4× sharpening upscaler, about 10 seconds. Nothing is invented, so it's safe for text and logos.",
            "**TURBO**: a fast distilled AI pass, about twice as fast as Fast with a similar look.",
            "**FAST / BALANCED / QUALITY**: more refinement steps and bigger output, slower each step up.",
            "**SUBTLE**: gentle cleanup that stays faithful (faces, lettering). **CREATIVE**: lots of new texture.",
            "A preset's light stays on until you turn a knob it set; then you're on custom settings.",
        ]),
        ("Output knobs", [
            "**Upscale**: how much bigger the result is (2× doubles width and height; **1.0×** keeps the size and only re-details it). Presets never change it, so you can use CREATIVE, SUBTLE or any preset at 1×.",
            "**Max Size**: caps the long side. Smaller is much faster; 768 px is quickest.",
            "**Memory**: 8 GB keeps work in small pieces so an 8 GB Mac doesn't run out of memory. Use 16 GB+ only on bigger Macs.",
        ]),
        ("Likeness", [
            "**Likeness** keeps people and things looking like your photo while the AI adds detail. It uses IP-Adapter Plus: each part of the picture is guided by its own part of the photo, so big pictures don't grow extra faces.",
            "Off by default. Try 30–60%; higher sticks closer to the photo but adds less new detail. Not used by INSTANT. The first use downloads it once (about 1.3 GB), checked against a fingerprint like the other models. It runs on every Mac; on Intel Macs it's slower.",
        ]),
        ("Trying several settings", [
            "Press **+ QUEUE** (⇧⌘↩) to add the photo again with the current settings. Change the knobs or pick another preset and press **+ QUEUE** again, as many times as you like, then press **ENHANCE** to run them all.",
            "Each queued row shows its settings in amber, and each result is saved with them in its file name. Right-click a row and choose **Use These Settings** to put them back on the knobs.",
        ]),
        ("My presets", [
            "Set the knobs the way you like, then press **+ SAVE** under **My presets** and give it a name. Tick **Include Adjust & filters** to keep your Light, Color and filter settings too.",
            "Click a preset to use it (its key lights up while the knobs match). Right-click it to **update** it with the current settings, **rename** it or **delete** it.",
        ]),
        ("Face Detail", [
            "Turn **Face Detail** on to redraw every face close up after the enhance (like Protoart Studio's face detailer, or ADetailer): each face is cut out with some room around it, redrawn at 512–768 px by the same AI, matched to the light and colour around it, and blended back through a soft oval. Small or distant faces get real eyes, lips and skin instead of mush.",
            "**Strength** sets how much each face may change: 15–20% (default 15%) sharpens and keeps the person; 30% and up redraws more and can change the look. **Hair** on also redraws the hair around the face. It adds roughly 10–30 seconds per face and needs no download. It isn't used by INSTANT. Turn it off with its switch.",
            "It runs before Face Fix, so you can use both: Face Detail adds detail, then Face Fix tidies eyes and teeth.",
        ]),
        ("Face Fix", [
            "Turn **Face Fix** on to restore faces after the enhance: it finds every face (with macOS's built-in face detection), repairs eyes, teeth and skin, and blends the result back in with a soft edge. Great for fixing faces the AI detail warped, and for old or blurry portraits.",
            "**CODE** (CodeFormer, the default) repairs badly blurred or damaged faces more strongly. Its **Fidelity** knob trades a cleaner face (low) for keeping the person's own features (high, 0.7+). **RESTORE** (RestoreFormer) is the natural, faithful alternative with a commercial-friendly licence.",
            "The restored face keeps the photo's own light and colour, so it never shows as a lighter or pinker patch.",
            "**Strength** blends the restored face with the original: 70–80% looks the most natural.",
            "**CodeFormer presets** (the same as Protoart Studio's) set everything in one click: **BALANCED** for most portraits, **LIKENESS** for real people, **HEAVY** for distorted or tiny faces, **SUBTLE** keeps skin texture, **OLD PHOTO** for scans, **GENTLE** for faces that only need a polish.",
            "The first time you use each model it downloads once (RestoreFormer about 280 MB, CodeFormer about 360 MB), checked against a fingerprint like the other models. CodeFormer's licence allows **non-commercial use only**.",
            "It works with every preset, including INSTANT. Faces smaller than about 12 pixels between the eyes are left alone. The status line says how many faces were fixed.",
        ]),
        ("Tone and Advanced knobs", [
            "**Creativity**: how much new detail the AI may invent (0.35 is the sweet spot).",
            "**Resemblance**: how strictly it follows the original's shapes. Higher = more faithful.",
            "**HDR**: punch of the new detail (no effect in Turbo).",
            "**Keep Tones** (on by default): after the AI adds detail, your photo's own brightness and colour are put back, so results don't come out lighter or with shifted colours. All the new fine detail stays. Turn it off to keep the AI's own light and colour.",
            "**Steps / Solver / Decay / Seed / Prompt**: fine-tuning for advanced users. **RESET** restores defaults.",
            "Drag a knob up or right to turn it clockwise. Hover over anything to see an explanation in the amber Guide strip.",
        ]),
    ]

    static let filters: [Section] = [
        ("Light and Color (like Camera Raw)", [
            "The **Adjust** panel left of the picture starts with Lightroom/Camera Raw-style sliders. **Light**: Exposure, Contrast, Highlights, Shadows, Whites, Blacks. **Color**: Temperature, Tint, Vibrance, Saturation.",
            "Drag a slider (it snaps to 0 in the middle); **double-click** it to reset. Each section's **RESET** clears it. They apply before the detail filters and preview live.",
            "**Presence**: **Texture** (fine detail such as skin and foliage), **Clarity** (mid-tone punch over larger areas), **Dehaze** (+ removes haze, − adds a soft veil), **Sharpen** and **Vignette** (+ darker, − lighter corners).",
            "**Color Mixer**: click a colour dot, then shift that colour's **Hue**, **Saturation** and **Luminance** (an amber dot marks changed colours). **Color Grading**: tint the **Shadows**, **Midtones** and **Highlights** (Hue + Amount); **Balance** moves the split. Grading changes colour, not brightness.",
        ]),
        ("Detail filters", [
            "Looks inspired by Color Efex Pro's Detail filters, below Light and Color in the **Adjust** panel (show or hide it with the **ADJUST** key next to the view keys).",
            "Fold the whole panel away with **◀** in its header (and open it again with **▶**). The header with **RESET** and **Apply: AFTER AI / BEFORE AI** stays put while the panel scrolls.",
            "**Click a filter's name** to apply it and open its knobs; click again to fold it away (it stays applied). The **checkbox** turns a filter on or off. Knobs only adjust a filter; they never switch it on. The screen previews everything live.",
            "**Bleach Bypass**: silvery, contrasty, half-desaturated film look. **Dark Contrasts**: deeper, moodier tones with crisp mid-tones. **Detail Extractor**: pulls out texture everywhere (FINE / NORMAL / COARSE radius).",
            "**Bloom** (like Procreate's): the bright parts glow and spill soft light around themselves. **Amount** = strength, **Transition** = how much counts as bright, **Size** = how far the light spreads, **Burn** = from a soft glow to light that blows out to white.",
            "**Glamour Glow**: a soft, flattering glow (Glow, Saturation, Warmth). **Blur**: softens the picture; **Soft** focus makes the blur glow instead of smear. **Skin Texture**: adds natural pores, fine bumps and optional freckles to smooth AI skin (keep it low, 20–40%; NEW PATTERN changes the layout).",
            "**Film Grain** adds photographic grain (Intensity, Grain Size, Softness, Mono or Color).",
            "Filters are applied **top to bottom**: use the **▲▼** buttons on a filter to restack it (e.g. grain usually goes last). Light and Color always run first. **RESET** turns everything off and restores the default order.",
        ]),
        ("Before or after the AI", [
            "**AFTER AI** (default): filters go on top of the enhanced result. Change them any time; nothing re-runs.",
            "**BEFORE AI**: the photo is filtered first and the AI enhances the filtered version (re-press Enhance after changing filters).",
            "You can also use the filters without enhancing at all: open a photo, turn on a filter, and save.",
        ]),
        ("Views", [
            "**SINGLE**: the result; click the picture to flip to the original and back.",
            "**SPLIT**: drag anywhere on the picture to move the divider (Before on the left, After on the right).",
            "**SIDE**: original and result side by side.",
            "**Zoom** by scrolling the **mouse wheel** (or two fingers on a trackpad) over the picture: it zooms smoothly toward the pointer, up to 400% of actual pixels. Pinch works too, and the keys in the corner jump to **FIT**, **100%** (one image pixel per screen pixel) and **200%**. Drag to look around while zoomed; in SPLIT, drag near the divider to move it. All three views keep the same zoom.",
        ]),
    ]

    static let saving: [Section] = [
        ("Save module", [
            "**Save To**: your default folder, the **Desktop** unless you change it (click the tape). Save, Save As, Auto-save and batch results use it.",
            "**Auto-save** (on by default): every finished result is saved there straight away, so nothing waits in a hidden temporary folder. macOS asks once to let the app save to your Desktop.",
            "**Format**: PNG (lossless), JPEG (small, 95% quality) or TIFF (lossless, for print and editing).",
            "**Mode**: **COPY** saves a brand-new file named like *photo-beautify.png* and never touches your original. **OVERWRITE** replaces the original photo, keeping its name and format; it asks first (you can tick \"Don't ask again\").",
            "**SAVE AS** (⇧⌘S) lets you pick any name, folder and format each time.",
            "Detail filters are always included in what you save.",
        ]),
        ("Batch mode", [
            "Flip **BATCH** (⌘B) to get a queue. Drop many images or whole folders and press **ENHANCE N**.",
            "Each result saves automatically with the Save module's settings as soon as it finishes.",
            "Click ✕ on a row to cancel just that image; right-click a row to re-run, remove or show it in Finder. **STOP** (⌘.) halts the whole queue.",
            "Batch defaults to 768 px for speed. Settings apply to images that haven't started yet.",
        ]),
    ]

    static let troubleshooting: [Section] = [
        ("It's slow", [
            "On an **Intel Mac** the AI runs on the processor (there's no Apple GPU to use), so expect minutes per image: **INSTANT** takes a few minutes, **TURBO** roughly 15–30+ minutes, FAST and up much longer. Keep Max Size at 768 px, and use the Adjust filters (instant on any Mac) when you don't need AI detail.",
            "Use **TURBO** or **INSTANT**, and keep Max Size at 768 px or 1K.",
            "Quit memory-heavy apps (browsers with many tabs, other AI apps, Protoart Studio). On an 8 GB Mac, running two AI apps at once can make each run several times slower.",
            "The very first image after launch is a little slower while things warm up.",
        ]),
        ("The download stops or fails", [
            "A firewall such as Little Snitch must allow **ProtoArt Beautify** to reach **huggingface.co**. Allow it, then quit and reopen the app; downloads resume where they stopped.",
            "Check you have about 10 GB free disk space for the first setup.",
        ]),
        ("\"Ran out of memory\" or the engine stopped", [
            "Set Memory to **8 GB**, lower Max Size or Upscale, close other apps, and try again.",
        ]),
        ("The result changed faces or text", [
            "Lower **Creativity** and raise **Resemblance**, or use **SUBTLE**. For logos and lettering use **INSTANT**.",
        ]),
        ("Face Fix didn't change anything", [
            "The status says **no faces found** when macOS can't spot a face: very small, side-on, heavily covered or painted faces are often missed. Try a larger Max Size, or crop closer to the person.",
            "If faces look too smooth or not like the person, lower **Strength**, or switch to **CODE** and raise **Fidelity**.",
        ]),
        ("Save says it can't overwrite", [
            "Some formats (for example camera RAW) can't be written. Switch Mode to **COPY** to save a new file instead.",
        ]),
        ("Filters look different after Enhance", [
            "With **BEFORE AI** the filters are baked into the enhanced result, so changing them afterwards needs another Enhance. Use **AFTER AI** to adjust them freely.",
        ]),
        ("Still stuck?", [
            "Use **Show Log** on any error message, or open *~/Library/Logs/ProtoArt Beautify.log*. To re-download the models, quit the app and delete *~/Library/Application Support/ProtoArt Beautify*.",
        ]),
    ]
}

// MARK: - App

/// `ProtoArt Beautify --selftest-filters <report.txt>`: renders every Adjust step on a test picture and writes
/// whether its kernel compiled, how much it changed the picture and how long it took (preview and full size).
enum FilterSelfTest {
    static func run(report path: String) {
        var lines = ["ProtoArt Beautify filter self-test"]
        func kernel(_ name: String, _ k: CIColorKernel?) { lines.append("kernel \(name): \(k == nil ? "FAILED" : "ok")") }
        kernel("local", Kernels.local); kernel("dehaze", Kernels.dehaze); kernel("vignette", Kernels.vignette)
        kernel("mixer", Kernels.mixer); kernel("grade", Kernels.grade); kernel("glow", Kernels.glow)
        kernel("blur", Kernels.blurMix); kernel("skin", Kernels.skin)
        func testImage(_ w: Int, _ h: Int) -> CGImage {
            let rect = CGRect(x: 0, y: 0, width: w, height: h)
            let grad = CIFilter(name: "CILinearGradient", parameters: [
                "inputPoint0": CIVector(x: 0, y: 0), "inputPoint1": CIVector(x: CGFloat(w), y: CGFloat(h)),
                "inputColor0": CIColor(red: 0.1, green: 0.3, blue: 0.7), "inputColor1": CIColor(red: 0.95, green: 0.7, blue: 0.4)])!
                .outputImage!.cropped(to: rect)
            let noise = CIFilter(name: "CIRandomGenerator")!.outputImage!
                .transformed(by: CGAffineTransform(scaleX: 6, y: 6)).cropped(to: rect)
                .applyingFilter("CIColorMatrix", parameters: [
                    "inputRVector": CIVector(x: 0.25, y: 0, z: 0, w: 0), "inputGVector": CIVector(x: 0, y: 0.25, z: 0, w: 0),
                    "inputBVector": CIVector(x: 0, y: 0, z: 0.25, w: 0), "inputBiasVector": CIVector(x: -0.12, y: -0.12, z: -0.12, w: 0)])
            let img = noise.applyingFilter("CIAdditionCompositing", parameters: [kCIInputBackgroundImageKey: grad]).cropped(to: rect)
            return DetailFilterEngine.context.createCGImage(img, from: rect, format: .RGBA8, colorSpace: DetailFilterEngine.sRGB)!
        }
        func meanDiff(_ a: CGImage, _ b: CGImage) -> Double {
            func px(_ i: CGImage) -> [UInt8] {
                let w = 64, h = 64
                var buf = [UInt8](repeating: 0, count: w * h * 4)
                let ctx = CGContext(data: &buf, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                    space: DetailFilterEngine.sRGB, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
                ctx.draw(i, in: CGRect(x: 0, y: 0, width: w, height: h))
                return buf
            }
            let x = px(a), y = px(b)
            var t = 0.0
            for k in 0..<x.count where k % 4 != 3 { t += abs(Double(x[k]) - Double(y[k])) }
            return t / Double(x.count / 4 * 3)
        }
        var cases: [(String, (inout DetailFilters) -> Void)] = [
            ("texture", { $0.texture = 0.8 }), ("clarity", { $0.clarity = 0.8 }), ("dehaze+", { $0.dehaze = 0.8 }),
            ("dehaze-", { $0.dehaze = -0.8 }), ("sharpen", { $0.sharpen = 0.8 }), ("vignette", { $0.vignette = 0.8 }),
            ("mixer", { $0.mixHue[5] = 1; $0.mixSat[1] = 1; $0.mixLum[5] = -1 }),
            ("grading", { $0.cgShSat = 1; $0.cgHiSat = 1; $0.cgBalance = 0.3 }),
            ("glow", { $0.glowOn = true; $0.ggGlow = 0.8; $0.ggWarmth = 0.5 }),
            ("blur", { $0.blurOn = true; $0.bzSize = 0.5 }), ("softfocus", { $0.blurOn = true; $0.bzKind = 1 }),
            ("skin", { $0.skinOn = true; $0.skAmount = 1.2; $0.skFreckles = 0.6 }),
            ("bloom", { $0.bloomOn = true }), ("grain", { $0.grainOn = true }), ("light", { $0.exposure = 0.5; $0.contrast = 0.3 }),
        ]
        let parts = cases  // a copy, so "everything" doesn't call itself
        cases.append(("everything", { f in for c in parts { c.1(&f) } }))
        for (w, h) in [(1600, 1067), (4000, 2667)] {
            let base = testImage(w, h)
            lines.append("-- \(w)x\(h)")
            for (name, set) in cases {
                var f = DetailFilters()
                set(&f)
                let t0 = Date()
                guard let out = DetailFilterEngine.render(f, base) else { lines.append("\(name): RENDER FAILED"); continue }
                let ms = Date().timeIntervalSince(t0) * 1000
                let d = meanDiff(base, out)
                lines.append(String(format: "%@: changed %.2f levels, %.0f ms%@", name, d, ms, d < 0.3 ? "  <-- NO VISIBLE CHANGE" : ""))
            }
        }
        // Off really is off: a default (all-off) setting must leave the picture untouched.
        lines.append("all-off anyOn=\(DetailFilters().anyOn)")
        try? lines.joined(separator: "\n").write(toFile: path, atomically: true, encoding: .utf8)
    }
}

@main
struct ProtoArtBeautifyApp: App {
    @StateObject private var model = AppModel()
    @StateObject private var guides = GuideCenter()
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    init() {
        let args = CommandLine.arguments
        if let i = args.firstIndex(of: "--selftest-filters"), i + 1 < args.count {
            FilterSelfTest.run(report: args[i + 1])
            exit(0)
        }
    }

    var body: some Scene {
        Window("ProtoArt Beautify", id: "main") {
            ContentView().environmentObject(model).environmentObject(guides)
                .onOpenURL { model.add([$0]) }
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1240, height: 740)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button(model.batchMode ? "Add Images…" : "Open Image…") { model.openPanel() }
                    .keyboardShortcut("o")
            }
            CommandGroup(replacing: .saveItem) {
                Button(model.saveMode == "overwrite" ? "Save (Replace Original)" : "Save") { model.quickSave() }
                    .keyboardShortcut("s").disabled(!model.canSave)
                Button("Save As…") { model.saveSelectedAs() }
                    .keyboardShortcut("s", modifiers: [.command, .shift]).disabled(!model.canSave)
                Button("Choose Save Folder…") { model.chooseOutputFolder() }
            }
            CommandMenu("Queue") {
                Toggle("Batch Mode", isOn: $model.batchMode).keyboardShortcut("b").disabled(model.running)
                Button("Enhance") { model.start() }.disabled(!model.canStart)
                Button("Stop") { model.stop() }.keyboardShortcut(".").disabled(!model.running)
                Divider()
                Button("Clear Finished") { model.clearFinished() }.disabled(!model.hasFinished)
                Button("Open Output Folder") { model.openOutputFolder() }
            }
            CommandGroup(replacing: .help) {
                Button("ProtoArt Beautify How-To") { model.showHelp = true }.keyboardShortcut("?")
            }
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationWillFinishLaunching(_ notification: Notification) {
        signal(SIGPIPE, SIG_IGN)
    }
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
    func applicationWillTerminate(_ notification: Notification) {
        AppModel.cleanTemp() // don't leave copies of the user's images behind
    }
}
