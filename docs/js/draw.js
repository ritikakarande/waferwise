// Draw tab: paint failing dies on a wafer and watch the model classify it live.

import { topClass } from "./model.js";
import { drawWafer } from "./wafers.js";
import { gateChip, paragraph, percent } from "./verdict.js";

const OFF = 0;
const PASS = 1;
const FAIL = 2;

// A round wafer of passing dies that fills most of the map, like the training wafers after resizing.
function blankWafer(size) {
  const codes = new Uint8Array(size * size);
  const centre = (size - 1) / 2;
  const radius = size / 2 - 1.5;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      codes[y * size + x] = (x - centre) ** 2 + (y - centre) ** 2 <= radius ** 2 ? PASS : OFF;
    }
  }
  return codes;
}

export function setupDraw({ site, wafers, classify }) {
  const { classes, threshold, image_size: size } = site;
  const canvas = document.getElementById("draw-wafer");
  const result = document.getElementById("draw-result");
  const bars = document.getElementById("draw-bars");
  const failTool = document.getElementById("draw-fail");
  const passTool = document.getElementById("draw-pass");
  const brush = document.getElementById("draw-brush");

  let codes = blankWafer(size);
  let paintWith = FAIL;
  let drawing = false;
  let lastCell = null;
  let busy = false;
  let pending = false;

  const rows = classes.map((name) => {
    const label = document.createElement("span");
    label.className = "name";
    label.textContent = name;
    const track = document.createElement("div");
    track.className = "track";
    const fill = document.createElement("div");
    fill.className = "fill";
    track.append(fill);
    const value = document.createElement("span");
    value.className = "value";
    bars.append(label, track, value);
    return { fill, value };
  });

  function stamp(column, row) {
    const reach = Number(brush.value);
    for (let dy = -reach; dy <= reach; dy++) {
      for (let dx = -reach; dx <= reach; dx++) {
        const x = column + dx;
        const y = row + dy;
        if (x < 0 || y < 0 || x >= size || y >= size || dx * dx + dy * dy > reach * reach + reach) continue;
        if (codes[y * size + x] !== OFF) codes[y * size + x] = paintWith;
      }
    }
  }

  // The browser reports only some points of a fast stroke, so fill in the dies between them.
  function paintTo(event) {
    const box = canvas.getBoundingClientRect();
    const column = Math.floor(((event.clientX - box.left) / box.width) * size);
    const row = Math.floor(((event.clientY - box.top) / box.height) * size);
    const [fromColumn, fromRow] = lastCell ?? [column, row];
    const steps = Math.max(Math.abs(column - fromColumn), Math.abs(row - fromRow), 1);
    for (let step = 0; step <= steps; step++) {
      stamp(Math.round(fromColumn + ((column - fromColumn) * step) / steps),
            Math.round(fromRow + ((row - fromRow) * step) / steps));
    }
    lastCell = [column, row];
    drawWafer(canvas, codes, size);
    scheduleClassify();
  }

  // Never run two classifications at once: changes made during a run are picked up by one more run.
  async function scheduleClassify() {
    if (busy) { pending = true; return; }
    busy = true;
    do {
      pending = false;
      showAnswer(await classify(codes, size));
    } while (pending);
    busy = false;
  }

  function showAnswer({ probabilities, milliseconds }) {
    const predicted = topClass(probabilities);
    const confidence = probabilities[predicted];
    const decision = paragraph(`${percent(confidence)} confident `);
    decision.append(gateChip(confidence, threshold));
    result.replaceChildren(
      paragraph(classes[predicted], "prediction"),
      decision,
      paragraph(`Classified in ${milliseconds.toFixed(1)} ms in your browser.`, "muted"),
    );
    probabilities.forEach((probability, index) => {
      rows[index].fill.style.width = percent(probability);
      rows[index].value.textContent = percent(probability, 0);
    });
  }

  function setTool(value) {
    paintWith = value;
    failTool.setAttribute("aria-pressed", String(value === FAIL));
    passTool.setAttribute("aria-pressed", String(value === PASS));
  }

  function load(newCodes) {
    codes = newCodes;
    drawWafer(canvas, codes, size);
    scheduleClassify();
  }

  function stopDrawing() {
    drawing = false;
    lastCell = null;
  }

  canvas.addEventListener("pointerdown", (event) => {
    drawing = true;
    lastCell = null;
    canvas.setPointerCapture(event.pointerId);
    paintTo(event);
  });
  canvas.addEventListener("pointermove", (event) => { if (drawing) paintTo(event); });
  canvas.addEventListener("pointerup", stopDrawing);
  canvas.addEventListener("pointercancel", stopDrawing);
  failTool.addEventListener("click", () => setTool(FAIL));
  passTool.addEventListener("click", () => setTool(PASS));
  document.getElementById("draw-clear").addEventListener("click", () => load(blankWafer(size)));
  document.getElementById("draw-real").addEventListener("click", () => {
    load(wafers[Math.floor(Math.random() * wafers.length)].slice());
  });

  return { start: () => load(codes), redraw: () => drawWafer(canvas, codes, size) };
}
