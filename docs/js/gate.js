// Gate tab: move the confidence threshold and see what it does on the validation wafers.

import { percent } from "./verdict.js";

const SVG = "http://www.w3.org/2000/svg";
const HEIGHT = 300;
const MARGIN = { top: 16, right: 24, bottom: 44, left: 56 };

function svgElement(name, attributes, text) {
  const element = document.createElementNS(SVG, name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  if (text !== undefined) element.textContent = text;
  return element;
}

function tile(number, label) {
  const element = document.createElement("div");
  element.className = "tile";
  const big = document.createElement("div");
  big.className = "number";
  big.textContent = number;
  const small = document.createElement("div");
  small.className = "label";
  small.textContent = label;
  element.append(big, small);
  return element;
}

// Trade-off chart: x = share sent to review, y = wrong automatic decisions, one point per threshold.
// It is drawn at the width it is shown at, so its text stays readable on a phone.
function drawChart(svg, curve) {
  const width = Math.max(300, Math.min(640, svg.parentElement.clientWidth));
  const maxReview = Math.max(...curve.map((point) => point.review_share));
  const maxWrong = Math.max(...curve.map((point) => point.wrong_automatic));
  const x = (share) => MARGIN.left + (share / maxReview) * (width - MARGIN.left - MARGIN.right);
  const y = (wrong) => HEIGHT - MARGIN.bottom - (wrong / maxWrong) * (HEIGHT - MARGIN.top - MARGIN.bottom);
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${width} ${HEIGHT}`);

  for (let wrong = 0; wrong <= maxWrong; wrong += 200) {
    svg.append(svgElement("line", { class: "grid", x1: MARGIN.left, x2: width - MARGIN.right, y1: y(wrong), y2: y(wrong) }));
    svg.append(svgElement("text", { class: "axis-label", x: MARGIN.left - 8, y: y(wrong) + 4, "text-anchor": "end" }, wrong));
  }
  const shareStep = width < 480 ? 0.2 : 0.1;
  for (let share = 0; share <= maxReview + 1e-9; share += shareStep) {
    svg.append(svgElement("text", { class: "axis-label", x: x(share), y: HEIGHT - MARGIN.bottom + 18, "text-anchor": "middle" }, percent(share, 0)));
  }
  const middle = (HEIGHT - MARGIN.bottom + MARGIN.top) / 2;
  svg.append(svgElement("text", { class: "axis-label", x: (width + MARGIN.left) / 2, y: HEIGHT - 6, "text-anchor": "middle" }, "Sent to review"));
  svg.append(svgElement("text", { class: "axis-label", x: 14, y: middle, "text-anchor": "middle", transform: `rotate(-90 14 ${middle})` },
    "Wrong automatic decisions"));
  const path = curve.map((point, i) => `${i ? "L" : "M"}${x(point.review_share).toFixed(1)},${y(point.wrong_automatic).toFixed(1)}`).join("");
  svg.append(svgElement("path", { class: "line", d: path }));

  const marker = svgElement("circle", { class: "marker", r: 6 });
  const label = svgElement("text", { class: "marker-label" });
  svg.append(marker, label);
  return function moveMarker(point) {
    const nearRight = x(point.review_share) > width - 180;
    marker.setAttribute("cx", x(point.review_share));
    marker.setAttribute("cy", y(point.wrong_automatic));
    label.setAttribute("x", x(point.review_share) + (nearRight ? -10 : 10));
    label.setAttribute("y", y(point.wrong_automatic) - 10);
    label.setAttribute("text-anchor", nearRight ? "end" : "start");
    label.textContent = `${percent(point.review_share)} reviewed, ${point.wrong_automatic} wrong`;
  };
}

export function setupGate({ site }) {
  const { curve, threshold: chosen, val_wafers: valWafers } = site;
  const slider = document.getElementById("gate-slider");
  const value = document.getElementById("gate-value");
  const tiles = document.getElementById("gate-tiles");
  const chart = document.getElementById("gate-chart");
  const chosenIndex = curve.findIndex((point) => point.threshold === chosen);

  document.getElementById("gate-count").textContent = valWafers.toLocaleString("en-US");
  slider.min = 0;
  slider.max = curve.length - 1;
  slider.step = 1;
  slider.value = chosenIndex;
  let moveMarker = () => {};

  function update() {
    const point = curve[Number(slider.value)];
    value.textContent = `${percent(point.threshold, 1)}${point.threshold === chosen ? " (chosen on validation)" : ""}`;
    slider.setAttribute("aria-valuetext", `${percent(point.threshold, 1)} confidence`);
    tiles.replaceChildren(
      tile(percent(point.review_share), "of wafers sent to a person"),
      tile(percent(point.auto_accuracy, 2), "accuracy of automatic decisions"),
      tile(point.wrong_automatic.toLocaleString("en-US"), "wrong automatic decisions"),
      tile(point.missed_defects.toLocaleString("en-US"), "defects passed as clean"),
    );
    moveMarker(point);
  }

  function redrawChart() {
    if (chart.parentElement.clientWidth === 0) return; // tab hidden: draw when it is shown
    moveMarker = drawChart(chart, curve);
    update();
  }

  slider.addEventListener("input", update);
  document.getElementById("gate-reset").addEventListener("click", () => { slider.value = chosenIndex; update(); });
  window.addEventListener("resize", () => requestAnimationFrame(redrawChart));
  update();
  return { show: redrawChart };
}
