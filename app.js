/**
 * =============================================================================
 * NextWord AI - Client-Side Bigram Language Model Playground
 * =============================================================================
 * Implements tokenization, transition counting (Markov chain bigrams),
 * greedy & probabilistic next-word prediction, and live interactive visualizers.
 * =============================================================================
 */

// Presets data
const PRESETS = {
  animals: `The quick brown fox jumps over the lazy dog.
The lazy dog sleeps in the warm sun.
The warm sun shines over the green meadow.
The brown fox runs across the green meadow.
A quick rabbit hops past the sleepy cat.
The sleepy cat watches the playful bird.
The playful bird sings a sweet song.
The brown fox listens to the sweet song.
The green meadow is full of life and color.
The quick brown fox loves the quiet forest.
The lazy dog loves the warm sun.`,

  space: `The starship journeys through the deep cosmos.
The deep cosmos glows with distant stars.
The brave astronaut explores the distant stars.
The distant stars guide the starship home.
A silent probe navigates through the asteroid field.
The asteroid field hides an ancient mystery.
The brave astronaut discovers an ancient mystery.
The ancient mystery unlocks the secrets of time.
The deep cosmos holds infinite wonder for humanity.
The starship returns with infinite wonder.`,

  coding: `The software engineer writes clean code for the web app.
The clean code runs fast on the modern browser.
The modern browser renders the beautiful user interface.
The developer designs the beautiful user interface with care.
A smart algorithm solves the complex problem quickly.
The complex problem requires clean code and tests.
The test suite validates the web app before release.
The developer ships the web app with great confidence.
The clean code is easy to read and maintain.`
};

// Application State
const state = {
  tokens: [],
  model: {}, // { "word": { "nextWord": count, ... } }
  activeWord: null,
  selectedPreset: "animals",
  sentenceLength: 5,
  strategy: "greedy", // "greedy" | "probabilistic"
  isAnimating: false
};

// DOM Elements
const elements = {
  corpusInput: document.getElementById("corpus-input"),
  btnRetrain: document.getElementById("btn-retrain"),
  statTokens: document.getElementById("stat-tokens"),
  statVocab: document.getElementById("stat-vocab"),
  statTransitions: document.getElementById("stat-transitions"),
  presetBtns: document.querySelectorAll(".preset-btn"),
  
  startWordInput: document.getElementById("start-word-input"),
  btnGenerate: document.getElementById("btn-generate"),
  btnStep: document.getElementById("btn-step"),
  suggestedChips: document.getElementById("suggested-chips"),
  sentenceLengthSlider: document.getElementById("sentence-length-slider"),
  sentenceLengthVal: document.getElementById("sentence-length-val"),
  strategyBtns: document.querySelectorAll(".toggle-btn"),
  
  sentenceDisplay: document.getElementById("sentence-display"),
  generationBadge: document.getElementById("generation-badge"),
  generationStepsFlow: document.getElementById("generation-steps-flow"),
  
  activeWordTag: document.getElementById("active-word-tag"),
  candidatesList: document.getElementById("candidates-list"),
  btnToggleGuide: document.getElementById("btn-toggle-guide"),
  btnCloseGuide: document.getElementById("btn-close-guide"),
  eli5GuidePanel: document.getElementById("eli5-guide-panel"),
  
  btnToggleRawModel: document.getElementById("btn-toggle-raw-model"),
  rawModelPanel: document.getElementById("raw-model-panel"),
  rawModelJson: document.getElementById("raw-model-json")
};

// =============================================================================
// Core NLP & Model Logic
// =============================================================================

/**
 * Tokenize raw text into an array of lowercase clean words.
 */
function tokenize(text) {
  if (!text) return [];
  // Match word characters, stripping punctuation
  const matches = text.toLowerCase().match(/\b[a-z0-9']+\b/g);
  return matches || [];
}

/**
 * Build bigram transition frequency map:
 * model[currentWord][nextWord] = count
 */
function buildModel(tokens) {
  const model = {};

  for (let i = 0; i < tokens.length - 1; i++) {
    const current = tokens[i];
    const next = tokens[i + 1];

    if (!model[current]) {
      model[current] = {};
    }

    model[current][next] = (model[current][next] || 0) + 1;
  }

  return model;
}

/**
 * Predict next word given a current word.
 * - strategy = "greedy": pick candidate with max frequency
 * - strategy = "probabilistic": weighted random pick (temperature sampling)
 */
function predictNext(currentWord, model, strategy = "greedy") {
  const word = currentWord.toLowerCase();
  const candidates = model[word];
  if (!candidates) return null;

  const candidateKeys = Object.keys(candidates);
  if (candidateKeys.length === 0) return null;

  if (strategy === "greedy") {
    let bestWord = candidateKeys[0];
    let maxCount = -1;
    for (const cand of candidateKeys) {
      if (candidates[cand] > maxCount) {
        maxCount = candidates[cand];
        bestWord = cand;
      }
    }
    return bestWord;
  }

  // Probabilistic / Weighted random choice
  const totalVotes = candidateKeys.reduce((sum, key) => sum + candidates[key], 0);
  let randomThreshold = Math.random() * totalVotes;
  
  for (const cand of candidateKeys) {
    randomThreshold -= candidates[cand];
    if (randomThreshold <= 0) {
      return cand;
    }
  }

  return candidateKeys[0];
}

// =============================================================================
// UI Update & Rendering Functions
// =============================================================================

/**
 * Retrain model from current textarea content
 */
function retrain() {
  const text = elements.corpusInput.value;
  state.tokens = tokenize(text);
  state.model = buildModel(state.tokens);

  // Update statistics
  elements.statTokens.textContent = state.tokens.length.toLocaleString();
  elements.statVocab.textContent = Object.keys(state.model).length.toLocaleString();
  
  let totalTransitions = 0;
  for (const key in state.model) {
    totalTransitions += Object.keys(state.model[key]).length;
  }
  elements.statTransitions.textContent = totalTransitions.toLocaleString();

  // Update raw JSON viewer
  elements.rawModelJson.textContent = JSON.stringify(state.model, null, 2);

  // Update suggested chips
  renderSuggestedChips();

  // If there's an active word, refresh inspector
  if (state.activeWord && state.model[state.activeWord]) {
    renderCandidateInspector(state.activeWord);
  } else {
    // Pick the most frequent word as initial active
    const topWords = getTopWords(1);
    if (topWords.length > 0) {
      setActiveWord(topWords[0]);
    } else {
      clearCandidateInspector();
    }
  }
}

/**
 * Get top N words sorted by total outgoing transitions
 */
function getTopWords(n = 6) {
  const words = Object.keys(state.model);
  words.sort((a, b) => {
    const sumA = Object.values(state.model[a]).reduce((acc, v) => acc + v, 0);
    const sumB = Object.values(state.model[b]).reduce((acc, v) => acc + v, 0);
    return sumB - sumA;
  });
  return words.slice(0, n);
}

/**
 * Render clickable suggestion chips
 */
function renderSuggestedChips() {
  const topWords = getTopWords(7);
  elements.suggestedChips.innerHTML = "";

  topWords.forEach(word => {
    const chip = document.createElement("button");
    chip.className = "word-chip";
    chip.textContent = word;
    chip.onclick = () => {
      elements.startWordInput.value = word;
      setActiveWord(word);
      generateSentence(false);
    };
    elements.suggestedChips.appendChild(chip);
  });
}

/**
 * Set active word and inspect its candidate transitions
 */
function setActiveWord(word, winnerCandidate = null) {
  state.activeWord = word.toLowerCase();
  elements.activeWordTag.textContent = `"${state.activeWord}"`;
  renderCandidateInspector(state.activeWord, winnerCandidate);
}

/**
 * Render the candidate next words bar chart and vote counters
 */
function renderCandidateInspector(word, winnerCandidate = null) {
  const candidates = state.model[word];

  if (!candidates || Object.keys(candidates).length === 0) {
    elements.candidatesList.innerHTML = `
      <div class="empty-state">
        <p>No recorded words follow <strong>"${word}"</strong> in this text.</p>
        <p style="margin-top: 0.4rem; font-size: 0.75rem; color: var(--text-dim);">
          (It might have only appeared at the very end of sentences).
        </p>
      </div>`;
    return;
  }

  const keys = Object.keys(candidates);
  const totalVotes = keys.reduce((sum, k) => sum + candidates[k], 0);

  // Sort candidates by frequency descending
  keys.sort((a, b) => candidates[b] - candidates[a]);

  // If no winner explicitly passed, the top candidate is the greedy winner
  const topWinner = winnerCandidate || keys[0];

  let html = "";
  keys.forEach(cand => {
    const count = candidates[cand];
    const pct = Math.round((count / totalVotes) * 100);
    const isWinner = cand === topWinner;

    html += `
      <div class="candidate-row ${isWinner ? 'is-winner' : ''}" onclick="window.pickCandidate('${cand}')">
        <div class="candidate-row-top">
          <span class="candidate-name">
            ${isWinner ? '<span class="winner-crown">👑</span>' : ''}
            ${cand}
          </span>
          <div class="candidate-stats">
            <span class="candidate-count">${count} vote${count > 1 ? 's' : ''}</span>
            <span class="candidate-pct">${pct}%</span>
          </div>
        </div>
        <div class="candidate-bar-track">
          <div class="candidate-bar-fill" style="width: ${pct}%;"></div>
        </div>
      </div>
    `;
  });

  elements.candidatesList.innerHTML = html;
}

function clearCandidateInspector() {
  elements.activeWordTag.textContent = "-";
  elements.candidatesList.innerHTML = `
    <div class="empty-state">
      <p>Type a starting word or click a chip to inspect its transition notebook.</p>
    </div>`;
}

// Expose pickCandidate to window for inline onclick in candidate cards
window.pickCandidate = function(word) {
  elements.startWordInput.value = word;
  setActiveWord(word);
  generateSentence(false);
};

// =============================================================================
// Sentence Generation (Instant & Step-by-Step)
// =============================================================================

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Generate a sentence starting with user word
 */
async function generateSentence(animated = false) {
  if (state.isAnimating) return;

  const rawInput = elements.startWordInput.value.trim();
  if (!rawInput) {
    elements.startWordInput.focus();
    return;
  }

  const startWord = rawInput.toLowerCase();

  // Validate start word exists in model
  if (!state.model[startWord]) {
    elements.sentenceDisplay.innerHTML = `
      <div style="color: var(--accent-amber);">
        <p><strong>Word not found:</strong> "${startWord}" does not exist in the current training text.</p>
        <p style="font-size: 0.85rem; margin-top: 0.35rem; color: var(--text-muted);">
          Try starting with: <code>${getTopWords(5).join(", ")}</code>
        </p>
      </div>
    `;
    elements.generationStepsFlow.innerHTML = "";
    elements.generationBadge.style.display = "none";
    return;
  }

  const length = state.sentenceLength;
  const sentenceTokens = [startWord];
  const transitionsTaken = [];

  let current = startWord;
  for (let i = 0; i < length - 1; i++) {
    const nextWord = predictNext(current, state.model, state.strategy);
    if (!nextWord) break;
    transitionsTaken.push({ from: current, to: nextWord });
    sentenceTokens.push(nextWord);
    current = nextWord;
  }

  // Render Display
  elements.generationBadge.style.display = "inline-block";

  if (!animated) {
    // Instant render
    renderCompleteSentence(sentenceTokens);
    renderStepFlow(sentenceTokens);
    setActiveWord(sentenceTokens[0], sentenceTokens[1]);
  } else {
    // Animated token-by-token generation
    state.isAnimating = true;
    elements.btnGenerate.disabled = true;
    elements.btnStep.disabled = true;

    elements.sentenceDisplay.innerHTML = `<span class="sentence-rendered"></span>`;
    elements.generationStepsFlow.innerHTML = "";

    const container = elements.sentenceDisplay.querySelector(".sentence-rendered");

    for (let i = 0; i < sentenceTokens.length; i++) {
      const word = sentenceTokens[i];
      const isFirst = i === 0;
      const isLast = i === sentenceTokens.length - 1;

      // Update Inspector to show what the model is looking at
      const nextWord = sentenceTokens[i + 1] || null;
      setActiveWord(word, nextWord);

      // Create token element
      const span = document.createElement("span");
      span.className = "token-word active-step";
      span.textContent = isFirst 
        ? word.charAt(0).toUpperCase() + word.slice(1) 
        : word;
      
      if (isLast) {
        span.textContent += ".";
      }

      container.appendChild(span);

      // Step indicator
      appendStepFlowItem(i + 1, word);

      // Flash highlight
      await sleep(400);
      span.classList.remove("active-step");
    }

    state.isAnimating = false;
    elements.btnGenerate.disabled = false;
    elements.btnStep.disabled = false;
  }
}

/**
 * Format and render completed sentence
 */
function renderCompleteSentence(tokens) {
  let html = '<span class="sentence-rendered">';
  tokens.forEach((token, idx) => {
    const isFirst = idx === 0;
    const isLast = idx === tokens.length - 1;
    let wordText = isFirst ? token.charAt(0).toUpperCase() + token.slice(1) : token;
    if (isLast) wordText += ".";

    html += `<span class="token-word" onclick="window.inspectToken('${token}', '${tokens[idx + 1] || ''}')">${wordText}</span>`;
  });
  html += '</span>';

  elements.sentenceDisplay.innerHTML = html;
}

window.inspectToken = function(current, next) {
  setActiveWord(current, next);
};

function renderStepFlow(tokens) {
  elements.generationStepsFlow.innerHTML = "";
  tokens.forEach((token, idx) => {
    appendStepFlowItem(idx + 1, token);
  });
}

function appendStepFlowItem(stepNumber, word) {
  const item = document.createElement("div");
  item.className = "flow-step-item";
  item.innerHTML = `
    <span class="flow-step-num">${stepNumber}</span>
    <span>${word}</span>
  `;
  elements.generationStepsFlow.appendChild(item);
}

// =============================================================================
// Event Handlers & Initialization
// =============================================================================

function setupEventListeners() {
  // Retrain button
  elements.btnRetrain.addEventListener("click", () => {
    retrain();
  });

  // Preset switching
  elements.presetBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      elements.presetBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const presetName = btn.dataset.preset;
      elements.corpusInput.value = PRESETS[presetName] || PRESETS.animals;
      retrain();
    });
  });

  // Generate buttons
  elements.btnGenerate.addEventListener("click", () => generateSentence(false));
  elements.btnStep.addEventListener("click", () => generateSentence(true));

  // Enter key in input
  elements.startWordInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      generateSentence(false);
    }
  });

  // Live input update to inspect candidate words as you type
  elements.startWordInput.addEventListener("input", (e) => {
    const word = e.target.value.trim().toLowerCase();
    if (state.model[word]) {
      setActiveWord(word);
    }
  });

  // Sentence length slider
  elements.sentenceLengthSlider.addEventListener("input", (e) => {
    state.sentenceLength = parseInt(e.target.value, 10);
    elements.sentenceLengthVal.textContent = `${state.sentenceLength} words`;
  });

  // Strategy toggle
  elements.strategyBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      elements.strategyBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.strategy = btn.dataset.strategy;
    });
  });

  // ELI5 Guide toggle
  elements.btnToggleGuide.addEventListener("click", () => {
    const isHidden = elements.eli5GuidePanel.style.display === "none";
    elements.eli5GuidePanel.style.display = isHidden ? "block" : "none";
  });

  elements.btnCloseGuide.addEventListener("click", () => {
    elements.eli5GuidePanel.style.display = "none";
  });

  // Raw model JSON inspector toggle
  elements.btnToggleRawModel.addEventListener("click", () => {
    const isHidden = elements.rawModelPanel.style.display === "none";
    elements.rawModelPanel.style.display = isHidden ? "block" : "none";
  });
}

// Initial Boot
async function init() {
  setupEventListeners();

  // Load sample.txt if accessible via HTTP or fall back to preset
  try {
    const response = await fetch("sample.txt");
    if (response.ok) {
      const text = await response.text();
      elements.corpusInput.value = text.trim();
    } else {
      elements.corpusInput.value = PRESETS.animals;
    }
  } catch (err) {
    elements.corpusInput.value = PRESETS.animals;
  }

  // Train model on initial text
  retrain();

  // Set default starting word
  elements.startWordInput.value = "quick";
  setActiveWord("quick");
  generateSentence(false);
}

// Run on page ready
document.addEventListener("DOMContentLoaded", init);
