// Renders the ProtoArt Beautify icon (1024×1024 PNG). Usage: swift make_icon.swift out.png
import AppKit

let out = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "icon_1024.png"
let size = 1024
let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: size, pixelsHigh: size,
                           bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                           colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)

// macOS icon grid: 824 pt body, ~185 pt corner radius, centered.
let body = NSRect(x: 100, y: 100, width: 824, height: 824)
let shape = NSBezierPath(roundedRect: body, xRadius: 185, yRadius: 185)

// Soft drop shadow
NSGraphicsContext.saveGraphicsState()
let shadow = NSShadow()
shadow.shadowColor = NSColor.black.withAlphaComponent(0.35)
shadow.shadowBlurRadius = 28
shadow.shadowOffset = NSSize(width: 0, height: -12)
shadow.set()
NSColor.black.setFill()
shape.fill()
NSGraphicsContext.restoreGraphicsState()

// Left half: chunky "low-res" pixels. Right half: smooth gradient. A diagonal seam between them.
NSGraphicsContext.saveGraphicsState()
shape.addClip()
let gradient = NSGradient(colors: [
    NSColor(srgbRed: 0.98, green: 0.45, blue: 0.36, alpha: 1),
    NSColor(srgbRed: 0.89, green: 0.22, blue: 0.62, alpha: 1),
    NSColor(srgbRed: 0.42, green: 0.25, blue: 0.93, alpha: 1),
])!
gradient.draw(in: body, angle: -55)

let seam = NSBezierPath()
seam.move(to: NSPoint(x: 100, y: 100))
seam.line(to: NSPoint(x: 560, y: 100))
seam.line(to: NSPoint(x: 330, y: 924))
seam.line(to: NSPoint(x: 100, y: 924))
seam.close()
NSGraphicsContext.saveGraphicsState()
seam.addClip()
let cell: CGFloat = 103
var y: CGFloat = 100
var row = 0
while y < 924 {
    var x: CGFloat = 100
    var col = 0
    while x < 924 {
        let shade = CGFloat(((row * 7 + col * 3) % 5)) * 0.05
        NSColor.black.withAlphaComponent(0.08 + shade).setFill()
        NSRect(x: x, y: y, width: cell, height: cell).fill()
        x += cell; col += 1
    }
    y += cell; row += 1
}
NSGraphicsContext.restoreGraphicsState()

// Seam highlight
let line = NSBezierPath()
line.move(to: NSPoint(x: 560, y: 100))
line.line(to: NSPoint(x: 330, y: 924))
line.lineWidth = 10
NSColor.white.withAlphaComponent(0.85).setStroke()
line.stroke()
NSGraphicsContext.restoreGraphicsState()

// Sparkles symbol
let config = NSImage.SymbolConfiguration(pointSize: 430, weight: .semibold)
    .applying(.init(paletteColors: [.white]))
if let sym = NSImage(systemSymbolName: "sparkles", accessibilityDescription: nil)?
    .withSymbolConfiguration(config) {
    let s = sym.size
    let r = NSRect(x: 512 - s.width / 2 + 40, y: 512 - s.height / 2, width: s.width, height: s.height)
    NSGraphicsContext.saveGraphicsState()
    let glow = NSShadow()
    glow.shadowColor = NSColor.black.withAlphaComponent(0.25)
    glow.shadowBlurRadius = 18
    glow.shadowOffset = NSSize(width: 0, height: -6)
    glow.set()
    sym.draw(in: r)
    NSGraphicsContext.restoreGraphicsState()
}

NSGraphicsContext.restoreGraphicsState()
try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: out))
print("wrote \(out)")
