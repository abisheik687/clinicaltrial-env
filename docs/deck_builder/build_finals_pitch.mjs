import path from "node:path";
import { fileURLToPath } from "node:url";

import { Presentation, PresentationFile } from "@oai/artifact-tool";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUTPUT_PATH = path.resolve(__dirname, "..", "ClinicalTrialEnv_Finals_Pitch.pptx");

const COLORS = {
  bg: "#F4F8FD",
  panel: "#FFFFFF",
  ink: "#17324A",
  muted: "#5C7288",
  line: "#D6E2EF",
  accent: "#2B6CB0",
  accentSoft: "#E8F2FF",
  success: "#DDF4E7",
  warn: "#FFF5C4",
  danger: "#FDE7E7",
};

const FONT = {
  title: "Poppins",
  body: "Lato",
};

const presentation = Presentation.create({
  slideSize: { width: 1280, height: 720 },
});

function addPanel(slide, left, top, width, height, fill = COLORS.panel) {
  return slide.shapes.add({
    geometry: "roundRect",
    position: { left, top, width, height },
    fill,
    line: { width: 1, fill: COLORS.line },
  });
}

function addText(slide, text, left, top, width, height, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height },
    fill: { color: "#FFFFFF", transparency: 100000 },
    line: { width: 0, fill: "#FFFFFF" },
  });
  shape.text = text;
  shape.text.typeface = opts.typeface || FONT.body;
  shape.text.fontSize = opts.fontSize || 22;
  shape.text.bold = opts.bold || false;
  shape.text.color = opts.color || COLORS.ink;
  shape.text.insets = opts.insets || { left: 4, right: 4, top: 4, bottom: 4 };
  if (opts.alignment) {
    shape.text.alignment = opts.alignment;
  }
  return shape;
}

function addTitle(slide, kicker, title, subtitle) {
  addText(slide, kicker, 72, 46, 360, 34, {
    fontSize: 18,
    color: COLORS.accent,
    bold: true,
    typeface: FONT.body,
  });
  addText(slide, title, 72, 78, 980, 92, {
    fontSize: 34,
    bold: true,
    typeface: FONT.title,
  });
  addText(slide, subtitle, 72, 170, 1030, 64, {
    fontSize: 18,
    color: COLORS.muted,
    typeface: FONT.body,
  });
}

function addBulletCard(slide, left, top, width, title, bullets, fill = COLORS.panel) {
  addPanel(slide, left, top, width, 190, fill);
  addText(slide, title, left + 18, top + 14, width - 36, 30, {
    fontSize: 21,
    bold: true,
    typeface: FONT.title,
  });
  addText(slide, bullets.map((item) => `• ${item}`).join("\n"), left + 18, top + 52, width - 36, 120, {
    fontSize: 18,
    color: COLORS.ink,
    typeface: FONT.body,
  });
}

function createCover() {
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.bg;
  addPanel(slide, 56, 42, 1168, 636, "#F9FBFF");
  addTitle(
    slide,
    "OPENENV HACKATHON FINALS",
    "Clinical Trial Operations Arena",
    "A minimal clinical trial operations environment where an agent must screen a patient, react to a protocol amendment, schedule a safe follow-up, and escalate a seizure-symptom event.",
  );
  addBulletCard(
    slide,
    72,
    278,
    352,
    "Why this is compelling",
    [
      "Professional workflow instead of a toy puzzle",
      "Verifier-based reward with safety-critical outcome checks",
      "Judge-friendly demo on a fixed seeded episode",
    ],
    COLORS.accentSoft,
  );
  addBulletCard(
    slide,
    448,
    278,
    352,
    "What the agent must do",
    [
      "Screen against inclusion and exclusion criteria",
      "Re-check a criterion after an amendment appears",
      "Schedule a valid follow-up day and handle safety escalation",
    ],
  );
  addBulletCard(
    slide,
    824,
    278,
    352,
    "Submission package",
    [
      "OpenEnv-style API hosted on Hugging Face Spaces",
      "Training, eval, and plotting scripts in the repo",
      "Interactive demo, walkthrough, and linked artifacts",
    ],
    COLORS.success,
  );
}

function createWorkflow() {
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.bg;
  addTitle(
    slide,
    "WORKFLOW",
    "One bounded episode, four judge-visible decisions",
    "The finals scope adds just enough long-horizon workflow to make the environment feel real while keeping verification simple.",
  );
  const cards = [
    ["1. Screening", "Review protocol criteria, inspect patient evidence, and use clarifications sparingly."],
    ["2. Amendment", "Notice the protocol change and re-check INC-003 under the updated truth."],
    ["3. Follow-up", "After safe enrollment, schedule the follow-up visit inside day 7 to day 10."],
    ["4. Safety", "When seizure symptoms appear before follow-up, escalate to the investigator."],
  ];
  cards.forEach(([title, body], index) => {
    const left = 72 + (index % 2) * 560;
    const top = 270 + Math.floor(index / 2) * 180;
    addPanel(slide, left, top, 520, 150, index === 3 ? COLORS.warn : COLORS.panel);
    addText(slide, title, left + 20, top + 18, 240, 28, {
      fontSize: 22,
      bold: true,
      typeface: FONT.title,
    });
    addText(slide, body, left + 20, top + 54, 470, 78, {
      fontSize: 18,
      color: COLORS.muted,
      typeface: FONT.body,
    });
  });
}

function createRewardSlide() {
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.bg;
  addTitle(
    slide,
    "REWARD DESIGN",
    "Verifier-first scoring keeps the task legible and hard to game",
    "The environment scores the final workflow outcome rather than dense intermediate behavior.",
  );
  addBulletCard(
    slide,
    72,
    280,
    260,
    "Eligibility",
    ["Correct safe enroll or exclude decision under the latest protocol state."],
    COLORS.accentSoft,
  );
  addBulletCard(
    slide,
    356,
    280,
    260,
    "Amendment",
    ["Correct re-check when the amendment changes the active truth."],
    COLORS.panel,
  );
  addBulletCard(
    slide,
    640,
    280,
    260,
    "Scheduling",
    ["Follow-up day stays inside the allowed visit window."],
    COLORS.panel,
  );
  addBulletCard(
    slide,
    924,
    280,
    260,
    "Safety",
    ["Seizure-symptom event gets investigator escalation instead of silence."],
    COLORS.warn,
  );
  addText(
    slide,
    "Unsafe enrollment returns -1. Invalid but schema-valid actions get a small penalty. Diagnostic metrics are tracked separately so judges can see how the policy behaves, not just whether it finished.",
    72,
    504,
    1100,
    88,
    {
      fontSize: 18,
      color: COLORS.ink,
      typeface: FONT.body,
    },
  );
}

function createWinningSlide() {
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.bg;
  addTitle(
    slide,
    "WHY THIS SHOULD WIN",
    "The project is ambitious enough to matter and simple enough to verify",
    "It gives judges a crisp story, a real RL environment, and a safety-critical workflow they can understand in under two minutes.",
  );
  addBulletCard(
    slide,
    72,
    280,
    352,
    "Innovation",
    [
      "Moves beyond screening into a minimal operations workflow",
      "Targets OpenEnv Theme 3.1 professional tasks directly",
    ],
    COLORS.accentSoft,
  );
  addBulletCard(
    slide,
    448,
    280,
    352,
    "Storytelling",
    [
      "Seeded walkthrough is easy for mentors and judges to follow",
      "Amendment, scheduling, and safety escalation are visually obvious",
    ],
    COLORS.panel,
  );
  addBulletCard(
    slide,
    824,
    280,
    352,
    "Execution",
    [
      "Repo includes training, evaluation, plots, demo, and deployment package",
      "Artifacts and links are organized so judges can verify quickly",
    ],
    COLORS.success,
  );
}

function createNextSteps() {
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.bg;
  addTitle(
    slide,
    "FINAL CHECKLIST",
    "What judges will see",
    "Use this deck together with the README, plots, and seeded demo to tell a fast, credible finalist story.",
  );
  addText(
    slide,
    [
      "• Open the Hugging Face Space and play the seeded Task 3 walkthrough.",
      "• Show the patient screening decision, amendment re-check, follow-up day, and safety escalation.",
      "• Reference the reward curve and held-out comparison chart from the README.",
      "• Close with why verifier-based professional workflows are valuable for LLM RL training.",
    ].join("\n"),
    72,
    286,
    980,
    180,
    {
      fontSize: 22,
      typeface: FONT.body,
    },
  );
  addPanel(slide, 874, 286, 310, 184, COLORS.danger);
  addText(slide, "Submission focus", 894, 308, 220, 28, {
    fontSize: 22,
    bold: true,
    typeface: FONT.title,
    color: COLORS.danger,
  });
  addText(
    slide,
    "Do not expand scope before the training evidence is strong. The fastest path to a top submission is a clean story plus a clear before/after result.",
    894,
    346,
    250,
    96,
    {
      fontSize: 18,
      typeface: FONT.body,
      color: COLORS.ink,
    },
  );
}

createCover();
createWorkflow();
createRewardSlide();
createWinningSlide();
createNextSteps();

const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(OUTPUT_PATH);

console.log(`Created ${OUTPUT_PATH}`);
