// Runs the wafer classifier in the browser with ONNX Runtime Web (loaded by index.html as the global `ort`).

const RUNTIME_FILES = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";

export async function loadModel(url) {
  const ort = window.ort;
  ort.env.wasm.wasmPaths = RUNTIME_FILES;
  ort.env.wasm.numThreads = 1; // threads need cross-origin isolation, which GitHub Pages does not provide
  const session = await ort.InferenceSession.create(url, { executionProviders: ["wasm"] });

  // Classifies one wafer: codes holds size*size values, 0 off wafer, 1 pass, 2 fail (as in training).
  async function classify(codes, size) {
    const pixels = new Float32Array(codes.length);
    for (let i = 0; i < codes.length; i++) pixels[i] = codes[i] / 2;
    const input = new ort.Tensor("float32", pixels, [1, 1, size, size]);
    const start = performance.now();
    const output = await session.run({ wafer: input });
    const milliseconds = performance.now() - start;
    return { probabilities: Array.from(output.probabilities.data), milliseconds };
  }

  // The first runs set up memory and take ~10 ms; do them now so the timings shown are steady-state.
  const blank = new Uint8Array(64 * 64);
  for (let i = 0; i < 3; i++) await classify(blank, 64);
  return classify;
}

export function topClass(probabilities) {
  let best = 0;
  for (let i = 1; i < probabilities.length; i++) if (probabilities[i] > probabilities[best]) best = i;
  return best;
}
