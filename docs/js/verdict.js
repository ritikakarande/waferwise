// Small pieces of markup shared by the Play and Draw tabs.

export function percent(value, digits = 1) {
  return `${(100 * value).toFixed(digits)}%`;
}

// The gate decision as a chip with an icon and a word, so it never relies on colour alone.
export function gateChip(confidence, threshold) {
  const automatic = confidence >= threshold;
  const chip = document.createElement("span");
  chip.className = `chip ${automatic ? "auto" : "review"}`;
  chip.textContent = automatic ? "✓ Automatic" : "⚠ Sent to review";
  chip.title = `Confidence ${percent(confidence)}; the gate is at ${percent(threshold)}`;
  return chip;
}

export function paragraph(text, className = "") {
  const element = document.createElement("p");
  element.textContent = text;
  if (className) element.className = className;
  return element;
}
