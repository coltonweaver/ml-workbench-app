import AppKit
import SwiftUI
import WorkbenchCore

/// A grid of images: datasets, conv filters, feature visualizations.
struct ImagesInstrumentView: View {
    let payload: ImagesPayload
    @State private var zoom: CGFloat = 3
    @State private var selected: Int?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("\(payload.images.count) image\(payload.images.count == 1 ? "" : "s")")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                Slider(value: $zoom, in: 1...8, step: 1) { Text("Zoom") }
                    .frame(width: 160)
                    .controlSize(.small)
            }
            ScrollView {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: cellWidth), spacing: 8)], alignment: .leading, spacing: 8) {
                    ForEach(payload.images.indices, id: \.self) { i in
                        let image = payload.images[i]
                        VStack(spacing: 2) {
                            if let cg = makeImage(image) {
                                Image(decorative: cg, scale: 1)
                                    .resizable()
                                    .interpolation(.none)
                                    .frame(width: CGFloat(image.width) * zoom, height: CGFloat(image.height) * zoom)
                                    .border(selected == i ? Color.accentColor : Color.secondary.opacity(0.3),
                                            width: selected == i ? 2 : 0.5)
                            } else {
                                Text("bad image").font(.caption2).foregroundStyle(.red)
                            }
                            if let label = image.label {
                                Text(label).font(.caption2).lineLimit(1)
                            }
                        }
                        .onTapGesture { selected = selected == i ? nil : i }
                    }
                }
            }
            .frame(maxHeight: 420)
            if let selected, payload.images.indices.contains(selected), let range = payload.images[selected].range,
               range.count == 2 {
                let image = payload.images[selected]
                let name = image.label ?? "image " + String(selected)
                Text("\(name): \(image.width)×\(image.height), values \(format(range[0])) to \(format(range[1]))")
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var cellWidth: CGFloat {
        let widest = payload.images.map(\.width).max() ?? 16
        return max(CGFloat(widest) * zoom, 40)
    }

    private func makeImage(_ image: ImagesPayload.Image) -> CGImage? {
        guard let bytes = image.bytes, let provider = CGDataProvider(data: bytes as CFData) else { return nil }
        let gray = image.channels == 1
        return CGImage(
            width: image.width, height: image.height,
            bitsPerComponent: 8, bitsPerPixel: 8 * image.channels, bytesPerRow: image.width * image.channels,
            space: gray ? CGColorSpaceCreateDeviceGray() : CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
        )
    }
}
