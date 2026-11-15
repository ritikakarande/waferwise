// Play tab: guess the defect pattern of a real wafer, then see what the model said.

import { topClass } from "./model.js";
import { drawWafer } from "./wafers.js";
import { gateChip, paragraph, percent } from "./verdict.js";

const DESCRIPTIONS = {
  "Center": "Failures clustered in the middle of the wafer.",
  "Donut": "A ring of failures around the centre, with the middle itself clean.",
  "Edge-Loc": "A cluster of failures in one spot at the edge.",
  "Edge-Ring": "Failures all the way round the edge.",
  "Loc": "A cluster of failures somewhere inside the wafer, away from the centre.",
  "Near-full": "Almost every die fails.",
  "Random": "Failures scattered all over the wafer.",
  "Scratch": "A thin line of failures, often curved.",
  "none": "No pattern: only a few scattered failures.",
};

function shuffled(count) {
  const order = Array.from({ length: count }, (_, i) => i);
  for (let i = count - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  return order;
}

export function setupPlay({ site, wafers, classify }) {
  const { classes, threshold, image_size: size, wafer_labels: labels } = site;
  const canvas = document.getElementById("play-wafer");
  const choices = document.getElementById("play-choices");
  const reveal = document.getElementById("play-reveal");
  const next = document.getElementById("play-next");
  const scoreText = document.getElementById("play-score");

  const order = shuffled(wafers.length);
  const score = { you: 0, model: 0, rounds: 0 };
  let position = 0;
  let current = null;
  let modelAnswer = null;

  const buttons = classes.map((name, index) => {
    const button = document.createElement("button");
    button.textContent = name;
    button.addEventListener("click", () => answer(index));
    choices.append(button);
    return button;
  });

  buildGuide(document.getElementById("play-guide"), classes, site.guide_examples, wafers, size);

  async function showNext() {
    current = order[position % order.length];
    position += 1;
    drawWafer(canvas, wafers[current], size);
    reveal.hidden = true;
    next.hidden = true;
    buttons.forEach((button) => { button.disabled = true; button.classList.remove("picked"); });
    modelAnswer = await classify(wafers[current], size); // the model answers before you, but stays hidden
    buttons.forEach((button) => { button.disabled = false; });
    buttons[0].focus({ preventScroll: true });
  }

  function answer(choice) {
    buttons.forEach((button) => { button.disabled = true; });
    buttons[choice].classList.add("picked");
    const truth = labels[current];
    const predicted = topClass(modelAnswer.probabilities);
    const confidence = modelAnswer.probabilities[predicted];
    const youRight = choice === truth;
    const modelRight = predicted === truth;
    score.rounds += 1;
    score.you += youRight ? 1 : 0;
    score.model += modelRight ? 1 : 0;
    scoreText.textContent = `You ${score.you} · Model ${score.model} · ${score.rounds} wafer${score.rounds === 1 ? "" : "s"}`;

    const verdict = youRight && modelRight ? "You both got it."
      : youRight ? "You beat the model!"
      : modelRight ? "The model got this one."
      : "Neither of you matched the label.";
    const modelLine = paragraph(`Model: ${classes[predicted]}, ${percent(confidence)} confident `);
    modelLine.append(gateChip(confidence, threshold));
    reveal.replaceChildren(
      paragraph(verdict, "verdict"),
      paragraph(`Dataset label: ${classes[truth]}. You said ${classes[choice]}.`),
      modelLine,
      paragraph(`The model took ${modelAnswer.milliseconds.toFixed(1)} ms in your browser.`, "muted"),
    );
    reveal.hidden = false;
    next.hidden = false;
    next.focus({ preventScroll: true });
  }

  next.addEventListener("click", showNext);
  return { start: showNext, redraw: () => current !== null && drawWafer(canvas, wafers[current], size) };
}

// One clear example per class (chosen in scripts/build_web.py), with a short description.
function buildGuide(list, classes, examples, wafers, size) {
  classes.forEach((name, index) => {
    const example = examples[index];
    const term = document.createElement("dt");
    const thumbnail = document.createElement("canvas");
    thumbnail.setAttribute("aria-hidden", "true");
    drawWafer(thumbnail, wafers[example], size);
    term.append(thumbnail);
    const description = document.createElement("dd");
    const title = document.createElement("strong");
    title.textContent = `${name}: `;
    description.append(title, DESCRIPTIONS[name] ?? "");
    list.append(term, description);
  });
}
