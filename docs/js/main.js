// Loads the model and the sample wafers, then wires up the three tabs.

import { setupDraw } from "./draw.js";
import { setupGate } from "./gate.js";
import { loadModel } from "./model.js";
import { setupPlay } from "./play.js";
import { loadWafers } from "./wafers.js";

const status = document.getElementById("status");

function setupTabs(onShow) {
  const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
  function select(tab) {
    tabs.forEach((other) => {
      const selected = other === tab;
      other.setAttribute("aria-selected", String(selected));
      other.tabIndex = selected ? 0 : -1;
      document.getElementById(other.getAttribute("aria-controls")).hidden = !selected;
    });
    onShow(tab.getAttribute("aria-controls"));
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => select(tab));
    tab.addEventListener("keydown", (event) => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
      if (!step) return;
      const nextTab = tabs[(index + step + tabs.length) % tabs.length];
      nextTab.focus();
      select(nextTab);
    });
  });
}

async function main() {
  // The gate tab needs only site.json, so it works while the model is still loading.
  const site = await (await fetch("assets/site.json")).json();
  const gate = setupGate({ site });
  let play = null;
  let draw = null;
  let drawShown = false;
  setupTabs((panel) => {
    if (panel === "gate") gate.show();
    if (panel === "draw") {
      drawShown = true;
      draw?.start();
    }
  });

  const [classify, wafers] = await Promise.all([
    loadModel("assets/wafer_cnn.onnx"),
    loadWafers("assets/wafers.bin", site.wafer_labels.length, site.image_size),
  ]);
  play = setupPlay({ site, wafers, classify });
  draw = setupDraw({ site, wafers, classify });
  if (drawShown) draw.start();

  // Wafer colours come from the theme, so redraw when the system switches between light and dark.
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    play.redraw();
    draw.redraw();
  });

  status.textContent = "";
  await play.start();
}

main().catch((error) => {
  status.textContent = `Something went wrong while loading: ${error.message}`;
  status.classList.add("error");
  console.error(error);
});
