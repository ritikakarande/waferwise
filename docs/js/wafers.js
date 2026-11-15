// Loads the sample wafers and draws wafer maps on a canvas.

// wafers.bin stores 4 pixels per byte, first pixel in the lowest two bits (see src/waferwise/web.py).
export async function loadWafers(url, count, size) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url}`);
  const packed = new Uint8Array(await response.arrayBuffer());
  const pixels = size * size;
  const wafers = [];
  for (let w = 0; w < count; w++) {
    const codes = new Uint8Array(pixels);
    for (let p = 0; p < pixels; p++) {
      const index = w * pixels + p;
      codes[p] = (packed[index >> 2] >> ((index & 3) * 2)) & 3;
    }
    wafers.push(codes);
  }
  return wafers;
}

function themeColours() {
  const style = getComputedStyle(document.documentElement);
  return ["--wafer-off", "--wafer-pass", "--wafer-fail"].map((name) => style.getPropertyValue(name).trim());
}

function hexToRgb(hex) {
  const value = parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

// The canvas holds one pixel per die; CSS scales it up with sharp edges (image-rendering: pixelated).
export function drawWafer(canvas, codes, size) {
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d");
  const image = context.createImageData(size, size);
  const colours = themeColours().map(hexToRgb);
  for (let p = 0; p < codes.length; p++) {
    const [r, g, b] = colours[codes[p]];
    image.data.set([r, g, b, 255], p * 4);
  }
  context.putImageData(image, 0, 0);
}
