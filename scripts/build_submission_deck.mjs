import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

let artifactTool;
try {
  artifactTool = await import("@oai/artifact-tool");
} catch {
  artifactTool = await import(
    "file:///C:/Users/vdeva/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs"
  );
}

const {
  Presentation,
  PresentationFile,
  auto,
  column,
  fill,
  fixed,
  fr,
  grid,
  grow,
  hug,
  image,
  panel,
  row,
  rule,
  shape,
  text,
  wrap,
} = artifactTool;

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUT_DIR = path.join(ROOT, "deliverables");
const PREVIEW_DIR = path.join(OUT_DIR, "judge_deck_previews");
const PPTX_PATH = path.join(OUT_DIR, "HeatGuard_AI_Judge_Ready_Deck.pptx");
const INSPECT_PATH = path.join(OUT_DIR, "HeatGuard_AI_Judge_Ready_Deck.inspect.ndjson");

const C = {
  bg: "#020604",
  bg2: "#06130D",
  panel: "#07170F",
  panel2: "#0A2117",
  panel3: "#0E2B1F",
  mint: "#67F3C2",
  mint2: "#20D998",
  soft: "#B8FFE4",
  white: "#F6FBF8",
  muted: "#9FB3AA",
  amber: "#F4B547",
  red: "#F36B6B",
  line: "#244438",
};

const SLIDE_W = 1920;
const SLIDE_H = 1080;

const presentation = Presentation.create({
  slideSize: { width: SLIDE_W, height: SLIDE_H },
});

function T(value, opts = {}) {
  return text(value, {
    name: opts.name,
    width: opts.width ?? fill,
    height: opts.height ?? hug,
    style: {
      fontSize: opts.size ?? 28,
      color: opts.color ?? C.white,
      bold: opts.bold ?? false,
    },
  });
}

function Root(children, opts = {}) {
  return panel(
    {
      name: "slide-root",
      width: fill,
      height: fill,
      fill: C.bg,
      padding: opts.padding ?? { x: 92, y: 64 },
    },
    column({ width: fill, height: fill, gap: opts.gap ?? 34 }, children),
  );
}

function Title(kicker, title, subtitle) {
  const kids = [
    T(kicker, { name: "kicker", size: 18, bold: true, color: C.mint, width: fill }),
    T(title, { name: "title", size: 58, bold: true, color: C.white, width: fill }),
  ];
  if (subtitle) {
    kids.push(T(subtitle, { name: "subtitle", size: 25, color: C.muted, width: wrap(1320), height: fixed(74) }));
  }
  kids.push(rule({ name: "title-rule", width: fixed(150), stroke: C.mint, weight: 4 }));
  return column({ name: "title-stack", width: fill, height: hug, gap: 12 }, kids);
}

function Pill(label, opts = {}) {
  return panel(
    {
      name: opts.name,
      width: opts.width ?? hug,
      height: fixed(opts.height ?? 48),
      fill: opts.fill ?? C.panel2,
      padding: { x: 18, y: 10 },
      borderRadius: "rounded-full",
    },
    T(label, { size: opts.size ?? 20, bold: true, color: opts.color ?? C.soft, width: opts.textWidth ?? hug }),
  );
}

function Symbol(label, color = C.mint) {
  return panel(
    {
      width: fixed(76),
      height: fixed(76),
      fill: C.panel2,
      padding: { x: 8, y: 14 },
      borderRadius: "rounded-full",
    },
    T(label, { size: 28, bold: true, color, width: fill }),
  );
}

function Metric(label, value, note, color = C.mint, opts = {}) {
  return panel(
    {
      width: fill,
      height: fixed(opts.height ?? 176),
      fill: C.panel,
      padding: opts.padding ?? { x: 24, y: 20 },
    },
    column({ width: fill, height: fill, gap: opts.gap ?? 12 }, [
      T(label, { size: opts.labelSize ?? 17, bold: true, color: C.muted, width: fill }),
      T(value, { size: opts.valueSize ?? 43, bold: true, color, width: fill }),
      T(note, { size: opts.noteSize ?? 19, color: C.muted, width: fill }),
    ]),
  );
}

function Bullet(title, body, symbol = ">", color = C.mint) {
  return row({ width: fill, height: hug, gap: 18 }, [
    Symbol(symbol, color),
    column({ width: fill, height: hug, gap: 8 }, [
      T(title, { size: 29, bold: true, color: C.white, width: fill }),
      T(body, { size: 22, color: C.muted, width: fill }),
    ]),
  ]);
}

function FlowStep(label, caption, mark) {
  return panel(
    { width: fill, height: fixed(152), fill: C.panel, padding: { x: 18, y: 18 } },
    column({ width: fill, height: fill, gap: 10 }, [
      T(mark, { size: 22, bold: true, color: C.mint, width: fill }),
      T(label, { size: 23, bold: true, color: C.white, width: fill }),
      T(caption, { size: 16, color: C.muted, width: fill }),
    ]),
  );
}

function TableRow(cells, widths, opts = {}) {
  return row(
    { width: fill, height: fixed(opts.height ?? 58), gap: 2 },
    cells.map((cell, i) =>
      panel(
        {
          width: fixed(widths[i]),
          height: fill,
          fill: opts.fill ?? C.panel,
          padding: { x: 14, y: 11 },
        },
        T(cell, {
          size: opts.size ?? 18,
          bold: opts.bold ?? false,
          color: opts.color ?? C.white,
          width: fill,
        }),
      ),
    ),
  );
}

function Bar(label, value, width, color = C.mint, note = "") {
  const trackWidth = 520;
  const items = [
    T(label, { size: 19, bold: true, color: C.white, width: fixed(110) }),
    panel(
      { width: fixed(trackWidth), height: fixed(22), fill: C.panel, padding: 0 },
      shape({ shapeType: "rect", width: fixed(Math.min(width, trackWidth)), height: fixed(22), fill: color }),
    ),
    T(value, { size: 19, bold: true, color, width: fixed(85) }),
  ];
  if (note) {
    items.push(T(note, { size: 16, color: C.muted, width: fill }));
  }
  return row({ width: fill, height: fixed(48), gap: 18 }, items);
}

function CompactMetric(label, value, note, color = C.mint) {
  return panel(
    { width: fill, height: fixed(132), fill: C.panel, padding: { x: 22, y: 18 } },
    column({ width: fill, height: fill, gap: 8 }, [
      T(label, { size: 16, bold: true, color: C.muted, width: fill }),
      T(value, { size: 34, bold: true, color, width: fill }),
      T(note, { size: 17, color: C.muted, width: fill }),
    ]),
  );
}

function RequirementRow(requirement, implementation, evidence, status = "MET") {
  return TableRow([status, requirement, implementation, evidence], [130, 410, 710, 300], {
    height: 54,
    size: 15,
    fill: C.panel,
    color: C.white,
  });
}

function ImportanceBar(label, value, maxValue = 0.15) {
  const trackWidth = 380;
  const widthPx = Math.max(34, Math.round((value / maxValue) * trackWidth));
  return row({ width: fill, height: fixed(47), gap: 14 }, [
    T(label, { size: 17, bold: true, color: C.white, width: fixed(330) }),
    panel(
      { width: fixed(trackWidth), height: fixed(20), fill: C.panel, padding: 0 },
      shape({ shapeType: "rect", width: fixed(widthPx), height: fixed(20), fill: C.mint }),
    ),
    T(value.toFixed(3), { size: 16, bold: true, color: C.soft, width: fixed(68) }),
  ]);
}

function addSlide(body) {
  const slide = presentation.slides.add();
  slide.compose(body, {
    frame: { left: 0, top: 0, width: SLIDE_W, height: SLIDE_H },
    baseUnit: 8,
  });
  return slide;
}

function addCover() {
  addSlide(
    Root(
      [
        row({ width: fill, height: hug, gap: 30 }, [
          panel(
            { width: fixed(118), height: fixed(118), fill: C.panel2, padding: { x: 20, y: 18 } },
            column({ width: fill, height: fill, gap: 2 }, [
              T("HG", { size: 39, bold: true, color: C.mint, width: fill }),
              T("AI", { size: 24, bold: true, color: C.soft, width: fill }),
            ]),
          ),
          column({ width: fill, height: hug, gap: 14 }, [
            T("IIIT LUCKNOW CLIMATE INTELLIGENCE CHALLENGE 2026", {
              size: 20,
              bold: true,
              color: C.mint,
              width: fill,
            }),
            T("HeatGuard AI", { size: 86, bold: true, color: C.white, width: fill }),
            T("10-Day Wet-Bulb Heat Risk Forecasting & Action Intelligence", {
              size: 34,
              bold: true,
              color: C.mint,
              width: fill,
            }),
          ]),
        ]),
        rule({ width: fixed(260), stroke: C.mint, weight: 5 }),
        T(
          "A deployable climate-intelligence platform that predicts dangerous wet-bulb heat, explains risk drivers, and generates audience-specific action advisories.",
          { size: 30, color: C.muted, width: wrap(1330) },
        ),
        row({ width: fill, height: hug, gap: 18 }, [
          Pill("Track 1: WBT Forecasting", { width: fixed(330) }),
          Pill("ML + Explainability", { width: fixed(270) }),
          Pill("Gemini Advisory", { width: fixed(250) }),
          Pill("Streamlit Dashboard", { width: fixed(300) }),
        ]),
        row({ width: fill, height: fixed(210), gap: 24 }, [
          panel({ width: grow(1), height: fill, fill: C.panel, padding: { x: 26, y: 24 } },
            column({ width: fill, height: fill, gap: 12 }, [
              T("Forecast", { size: 35, bold: true, color: C.white, width: fill }),
              T("Predict 10 future wet-bulb values for each Kaggle row.", { size: 23, color: C.muted, width: fill }),
            ])),
          panel({ width: grow(1), height: fill, fill: C.panel, padding: { x: 26, y: 24 } },
            column({ width: fill, height: fill, gap: 12 }, [
              T("Explain", { size: 35, bold: true, color: C.white, width: fill }),
              T("Show risk level, reliability, and driver explanation.", { size: 23, color: C.muted, width: fill }),
            ])),
          panel({ width: grow(1), height: fill, fill: C.panel2, padding: { x: 26, y: 24 } },
            column({ width: fill, height: fill, gap: 12 }, [
              T("Act", { size: 35, bold: true, color: C.soft, width: fill }),
              T("Generate heat advisories for citizens and institutions.", { size: 23, color: C.muted, width: fill }),
            ])),
        ]),
      ],
      { padding: { x: 108, y: 74 }, gap: 34 },
    ),
  );
}

function addJudgeRequirements() {
  addSlide(
    Root([
      Title(
        "JUDGE REQUIREMENT COVERAGE",
        "Everything the challenge asks for is represented",
        "HeatGuard AI is positioned as a complete deployable climate-intelligence system, not only a Kaggle notebook.",
      ),
      panel(
        { width: fill, height: fixed(535), fill: C.panel2, padding: { x: 28, y: 22 } },
        column({ width: fill, height: fill, gap: 8 }, [
          TableRow(["Status", "Requirement", "HeatGuard AI implementation", "Evidence"], [130, 410, 710, 300], {
            height: 46,
            size: 15,
            fill: C.panel3,
            bold: true,
            color: C.mint,
          }),
          RequirementRow("Track 1 forecast", "Predicts target_day_1 through target_day_10 for each Kaggle test row.", "submission.csv"),
          RequirementRow("Strong ML pipeline", "Reusable training, feature engineering, model selection, and saved artifacts.", "src/ + models/"),
          RequirementRow("Time-series validation", "Chronological split with shifted future labels and no target leakage.", "metrics.json"),
          RequirementRow("Interpretable risk", "Feature importances, plain-English explanation, and WBT risk thresholds.", "explainability.py"),
          RequirementRow("GenAI advisory", "Gemini action plans with environment-secret key and rule-based fallback.", "advisory.py"),
          RequirementRow("Modern dashboard", "Streamlit + Plotly interface focused on forecast, risk, explanation, and action.", "app.py"),
          RequirementRow("Deployment-ready", "GitHub/Hugging Face structure, requirements.txt, README, .env.example.", "repo root"),
        ]),
      ),
      row({ width: fill, height: fixed(132), gap: 22 }, [
        CompactMetric("No unrelated scope", "Clean", "no login, payment, or admin panel", C.soft),
        CompactMetric("Data policy", "Safe", "no external data enabled by default", C.mint),
        CompactMetric("Secret handling", "Secure", "GEMINI_API_KEY loaded from env", C.mint2),
        CompactMetric("Demo fallback", "Reliable", "synthetic data only for UI testing", C.soft),
      ]),
    ]),
  );
}

function addProblem() {
  addSlide(
    Root([
      Title("WHY IT MATTERS", "Heat risk is not just temperature", "Wet-bulb temperature captures combined heat and humidity stress, which is closer to human cooling limits."),
      grid({ width: fill, height: fill, columns: [fr(0.95), fr(1.05)], rows: [fr(1)], columnGap: 44 }, [
        column({ width: fill, height: fill, gap: 28 }, [
          Bullet("Physiological stress", "When humidity is high, sweating becomes less effective and outdoor work becomes unsafe faster.", "WBT"),
          Bullet("Operational pressure", "Hospitals, schools, and city agencies need advance warning, not just retrospective heat reports.", "10d"),
          Bullet("Actionable intelligence", "The platform translates forecasts into risk levels and practical heat-response actions.", "AI"),
        ]),
        panel(
          { width: fill, height: fill, fill: C.panel, padding: { x: 34, y: 30 } },
          column({ width: fill, height: fill, gap: 22 }, [
            T("Wet-bulb risk thresholds", { size: 34, bold: true, color: C.white, width: fill }),
            column({ width: fill, height: hug, gap: 6 }, [
              TableRow(["Low", "< 24 C", "normal monitoring"], [180, 170, 420], { fill: C.mint2, bold: true, color: C.bg, height: 52 }),
              TableRow(["Moderate", "24-27 C", "increase caution"], [180, 170, 420], { fill: C.mint, bold: true, color: C.bg, height: 52 }),
              TableRow(["High", "27-30 C", "heat action prep"], [180, 170, 420], { fill: C.amber, bold: true, color: C.bg, height: 52 }),
              TableRow(["Extreme", ">= 30 C", "urgent protective response"], [180, 170, 420], { fill: C.red, bold: true, color: C.bg, height: 52 }),
            ]),
            rule({ width: fill, stroke: C.line, weight: 2 }),
            T("Thresholds are centralized in code and can be adjusted if organizers provide local cutoffs.", {
              size: 25,
              color: C.muted,
              width: fill,
            }),
            row({ width: fill, height: hug, gap: 18 }, [
              Pill("Forecast", { width: fixed(170), textWidth: fill }),
              Pill("Classify", { width: fixed(170), textWidth: fill }),
              Pill("Explain", { width: fixed(150), textWidth: fill }),
              Pill("Advise", { width: fixed(150), textWidth: fill }),
            ]),
          ]),
        ),
      ]),
    ]),
  );
}

function addChallengeTarget() {
  addSlide(
    Root([
      Title("CHALLENGE TARGET", "Predict 10 future WBT values per test row", "HeatGuard AI produces Kaggle-ready target_day_1 through target_day_10 predictions."),
      grid({ width: fill, height: fill, columns: [fr(1.05), fr(0.95)], rows: [fr(1)], columnGap: 42 }, [
        panel(
          { width: fill, height: fill, fill: C.panel, padding: { x: 30, y: 28 } },
          column({ width: fill, height: fill, gap: 22 }, [
            T("Input files used", { size: 32, bold: true, color: C.white, width: fill }),
            TableRow(["File", "Role"], [240, 570], { fill: C.panel2, bold: true, color: C.mint, height: 54 }),
            TableRow(["train.csv", "Historical WBT labels for supervised training"], [240, 570]),
            TableRow(["context.csv", "Recent context for lag and rolling features"], [240, 570]),
            TableRow(["test.csv", "Future rows to score by location/day"], [240, 570]),
            TableRow(["sample_submission.csv", "Required output schema"], [240, 570]),
          ]),
        ),
        column({ width: fill, height: fill, gap: 22 }, [
          FlowStep("Feature frame", "weather, location, time, lag, rolling", "01"),
          FlowStep("Direct horizon models", "one forecast path for days 1-10", "02"),
          FlowStep("submission.csv", "row_id + target_day_1...target_day_10", "03"),
          panel(
            { width: fill, height: fill, fill: C.panel2, padding: { x: 26, y: 24 } },
            T("The generated CSV has 500,875 rows and no missing predictions.", {
              size: 32,
              bold: true,
              color: C.soft,
              width: fill,
            }),
          ),
        ]),
      ]),
    ]),
  );
}

function addPipeline() {
  const steps = [
    ["Dataset", "train/test/context CSV"],
    ["Preprocess", "missing values + dates"],
    ["Features", "lags, rolling, cyclical"],
    ["ML Forecast", "Fast ExtraTrees models"],
    ["Risk Layer", "WBT thresholds"],
    ["Advisory", "Gemini or rule fallback"],
  ];
  addSlide(
    Root([
      Title("SYSTEM DESIGN", "End-to-end climate intelligence pipeline", "The submission is more than a model: it is a reusable forecasting system connected to a demo-ready dashboard."),
      column({ width: fill, height: fill, gap: 30 }, [
        row(
          { width: fill, height: fixed(190), gap: 14 },
          steps.map((s, i) =>
            row({ width: grow(1), height: fill, gap: 10 }, [
              FlowStep(s[0], s[1], String(i + 1).padStart(2, "0")),
              i < steps.length - 1 ? T(">", { size: 34, bold: true, color: C.mint, width: fixed(22), height: fill }) : T("", { width: fixed(1), height: fill }),
            ]),
          ),
        ),
        panel(
          { width: fill, height: fill, fill: C.panel, padding: { x: 34, y: 32 } },
          grid({ width: fill, height: fill, columns: [fr(1), fr(1), fr(1)], rows: [fr(1)], columnGap: 28 }, [
            Bullet("No leakage", "Future target labels are shifted; features use current, past, and context values.", "!"),
            Bullet("Reusable", "Training, prediction, submission, explainability, and advisory are separated into src modules.", "ML"),
            Bullet("Deployable", "Streamlit app runs locally and is compatible with Hugging Face Spaces secrets.", "HF"),
          ]),
        ),
      ]),
    ]),
  );
}

function addFeatures() {
  addSlide(
    Root([
      Title("FEATURE ENGINEERING", "Strong temporal features drive the forecast", "The model sees weather predictors plus time-aware patterns that support 10-day lead predictions."),
      grid({ width: fill, height: fill, columns: [fr(1.15), fr(0.85)], rows: [fr(1)], columnGap: 42 }, [
        panel(
          { width: fill, height: fill, fill: C.panel, padding: { x: 28, y: 28 } },
          column({ width: fill, height: fill, gap: 10 }, [
            TableRow(["Feature group", "Examples"], [280, 600], { fill: C.panel2, bold: true, color: C.mint, height: 54 }),
            TableRow(["Weather state", "temperature, humidity, wind, pressure"], [280, 600]),
            TableRow(["Location", "relative lat/lon, location code"], [280, 600]),
            TableRow(["Calendar", "month, day of year, sine/cosine cycles"], [280, 600]),
            TableRow(["Lag history", "lag_1, lag_2, lag_3, lag_7, lag_10"], [280, 600]),
            TableRow(["Rolling trend", "mean_3, mean_7, std_7, max_7"], [280, 600]),
          ]),
        ),
        column({ width: fill, height: fill, gap: 24 }, [
          Metric("Engineered features", "237", "saved in models/features.json", C.mint),
          Metric("Forecast horizon", "10 days", "direct target_day outputs", C.soft),
          panel(
            { width: fill, height: fill, fill: C.panel2, padding: { x: 26, y: 24 } },
            T("Feature design emphasizes recent heat/humidity behavior and seasonal cycles while preserving chronological validation.", {
              size: 30,
              bold: true,
              color: C.white,
              width: fill,
            }),
          ),
        ]),
      ]),
    ]),
  );
}

function addModelWorkflow() {
  addSlide(
    Root([
      Title(
        "ML PIPELINE",
        "Reusable training flow built for real Kaggle data",
        "The system automatically uses real files from data/, engineers time-aware features, trains candidate models, and saves deployment artifacts.",
      ),
      grid({ width: fill, height: fill, columns: [fr(1), fr(1), fr(1)], rows: [fr(1)], columnGap: 28 }, [
        panel(
          { width: fill, height: fill, fill: C.panel, padding: { x: 28, y: 28 } },
          column({ width: fill, height: fill, gap: 18 }, [
            T("Data handling", { size: 32, bold: true, color: C.white, width: fill }),
            Bullet("Auto-discovery", "Looks for train.csv, test.csv, context.csv, and sample_submission.csv.", "01"),
            Bullet("Target detection", "Detects WBT/wet_bulb/target-like columns, with CLI override support.", "02"),
            Bullet("Missing values", "Median imputation inside the saved sklearn pipeline.", "03"),
          ]),
        ),
        panel(
          { width: fill, height: fill, fill: C.panel, padding: { x: 28, y: 28 } },
          column({ width: fill, height: fill, gap: 18 }, [
            T("Feature factory", { size: 32, bold: true, color: C.white, width: fill }),
            CompactMetric("Total engineered features", "237", "calendar, weather, lag, rolling", C.mint),
            CompactMetric("Lag features", "115", "1, 2, 3, 7, and 10-day memory", C.soft),
            CompactMetric("Rolling features", "92", "mean/std/max trend windows", C.mint2),
          ]),
        ),
        panel(
          { width: fill, height: fill, fill: C.panel, padding: { x: 28, y: 28 } },
          column({ width: fill, height: fill, gap: 14 }, [
            T("Model selection", { size: 32, bold: true, color: C.white, width: fill }),
            TableRow(["Candidate", "Role"], [215, 245], { fill: C.panel2, bold: true, color: C.mint, height: 50, size: 15 }),
            TableRow(["RandomForest", "robust baseline"], [215, 245], { height: 52, size: 15 }),
            TableRow(["HistGradientBoosting", "fast nonlinear"], [215, 245], { height: 52, size: 15 }),
            TableRow(["ExtraTrees", "selected family"], [215, 245], { height: 52, size: 15 }),
            TableRow(["XGBoost/LightGBM", "optional"], [215, 245], { height: 52, size: 15 }),
            CompactMetric("Selected", "FastExtraTrees", "best saved artifact by RMSE", C.soft),
          ]),
        ),
      ]),
    ]),
  );
}

function addExplainabilityRisk() {
  const topFeatures = [
    ["cos_month", 0.150],
    ["month", 0.110],
    ["cos_dayofyear_lag_3", 0.078],
    ["row_day_index_lag_7", 0.051],
    ["CLRSKY_SW_roll_mean_7", 0.049],
    ["cos_dayofyear_lag_10", 0.038],
    ["row_day_index", 0.038],
    ["cos_dayofyear_roll_mean_7", 0.036],
  ];
  addSlide(
    Root([
      Title(
        "EXPLAINABILITY + RISK INTELLIGENCE",
        "Predictions are translated into reasons and action levels",
        "The dashboard uses model feature importances plus wet-bulb thresholds so judges can understand both the forecast and the response.",
      ),
      grid({ width: fill, height: fill, columns: [fr(1.1), fr(0.9)], rows: [fr(1)], columnGap: 42 }, [
        panel(
          { width: fill, height: fill, fill: C.panel, padding: { x: 30, y: 28 } },
          column({ width: fill, height: fill, gap: 15 }, [
            T("Top contributing model features", { size: 32, bold: true, color: C.white, width: fill }),
            ...topFeatures.map(([label, value]) => ImportanceBar(label, value)),
            rule({ width: fill, stroke: C.line, weight: 2 }),
            T("Seasonal cycle, recent day index, and clear-sky solar radiation trends are among the strongest learned drivers.", {
              size: 23,
              color: C.muted,
              width: fill,
            }),
          ]),
        ),
        column({ width: fill, height: fill, gap: 24 }, [
          panel(
            { width: fill, height: fixed(350), fill: C.panel, padding: { x: 26, y: 24 } },
            column({ width: fill, height: fill, gap: 9 }, [
              T("Wet-bulb risk thresholds", { size: 29, bold: true, color: C.white, width: fill }),
              TableRow(["Low", "<24 C", "monitor"], [150, 150, 310], { fill: C.mint2, bold: true, color: C.bg, height: 48, size: 16 }),
              TableRow(["Moderate", "24-27 C", "caution"], [150, 150, 310], { fill: C.mint, bold: true, color: C.bg, height: 48, size: 16 }),
              TableRow(["High", "27-30 C", "prepare"], [150, 150, 310], { fill: C.amber, bold: true, color: C.bg, height: 48, size: 16 }),
              TableRow(["Extreme", ">=30 C", "protect"], [150, 150, 310], { fill: C.red, bold: true, color: C.bg, height: 48, size: 16 }),
            ]),
          ),
          panel(
            { width: fill, height: fill, fill: C.panel2, padding: { x: 28, y: 26 } },
            column({ width: fill, height: fill, gap: 18 }, [
              T("Plain-English explanation", { size: 31, bold: true, color: C.white, width: fill }),
              T("Example: risk is low in the selected forecast because the predicted WBT remains below 24 C, while the model is mainly reading seasonal cycle and recent heat-trend features.", {
                size: 25,
                color: C.muted,
                width: fill,
              }),
              T("For high or extreme peaks, the same module turns elevated humidity, temperature, and rolling heat trends into a concise explanation for users.", {
                size: 25,
                color: C.soft,
                width: fill,
              }),
            ]),
          ),
        ]),
      ]),
    ]),
  );
}

function addValidation() {
  addSlide(
    Root([
      Title("MODEL & VALIDATION", "FastExtraTrees selected on time-aware validation", "The model was trained on real Kaggle train.csv with a chronological split and future-shifted targets."),
      grid({ width: fill, height: fill, columns: [fr(0.95), fr(1.05)], rows: [fr(1)], columnGap: 44 }, [
        column({ width: fill, height: fill, gap: 22 }, [
          row({ width: fill, height: hug, gap: 18 }, [
            Metric("MAE", "0.323", "lower is better", C.mint),
            Metric("RMSE", "0.446", "selection metric", C.mint),
          ]),
          Metric("Validation R2", "0.993", "fit on held-out time split", C.soft),
          panel(
            { width: fill, height: fill, fill: C.panel, padding: { x: 28, y: 24 } },
            T("Validation approach: chronological 80/20 split. The feature pipeline avoids future target leakage.", {
              size: 28,
              bold: true,
              color: C.white,
              width: fill,
            }),
          ),
        ]),
        panel(
          { width: fill, height: fill, fill: C.panel, padding: { x: 30, y: 30 } },
          column({ width: fill, height: fill, gap: 12 }, [
            T("Validation score snapshot", { size: 32, bold: true, color: C.white, width: fill }),
            Bar("R2", "0.993", 552, C.soft),
            Bar("RMSE", "0.446", 248, C.mint),
            Bar("MAE", "0.323", 180, C.mint2),
            rule({ width: fill, stroke: C.line, weight: 2 }),
            T("Trained model artifact: models/heatguard_model.joblib", {
              size: 24,
              color: C.muted,
              width: fill,
            }),
          ]),
        ),
      ]),
    ]),
  );
}

function addDashboard() {
  addSlide(
    Root([
      Title("DASHBOARD EXPERIENCE", "A judge-friendly interface focused on action", "The UI keeps only what matters: location, forecast horizon, risk level, forecast chart, explanation, and advisory."),
      grid({ width: fill, height: fill, columns: [fr(1.25), fr(0.75)], rows: [fr(1)], columnGap: 42 }, [
        panel(
          { width: fill, height: fill, fill: C.panel, padding: { x: 28, y: 18 } },
          column({ width: fill, height: fill, gap: 12 }, [
            row({ width: fill, height: fixed(76), gap: 16 }, [
              panel({ width: fixed(66), height: fixed(66), fill: C.panel2, padding: 10 }, T("HG", { size: 24, bold: true, color: C.mint, width: fill })),
              column({ width: fill, height: hug, gap: 6 }, [
                T("HeatGuard AI", { size: 38, bold: true, color: C.white, width: fill }),
                T("10-Day Wet-Bulb Heat Risk Forecasting & Action Intelligence", { size: 21, bold: true, color: C.mint, width: fill }),
              ]),
              Pill("Track 1", { width: fixed(160) }),
            ]),
            row({ width: fill, height: fixed(124), gap: 16 }, [
              Metric("Peak WBT", "17.2 C", "Day 7", C.mint, { height: 124, valueSize: 34, noteSize: 16, padding: { x: 20, y: 12 }, gap: 6 }),
              Metric("Risk Level", "Low", "WBT threshold", C.mint2, { height: 124, valueSize: 34, noteSize: 16, padding: { x: 20, y: 12 }, gap: 6 }),
              Metric("Reliability", "96%", "validation-derived", C.soft, { height: 124, valueSize: 34, noteSize: 16, padding: { x: 20, y: 12 }, gap: 6 }),
            ]),
            panel(
              { width: fill, height: fill, fill: C.bg2, padding: { x: 28, y: 24 } },
              column({ width: fill, height: fill, gap: 8 }, [
                T("10-Day Wet-Bulb Forecast", { size: 27, bold: true, color: C.white, width: fill }),
                row({ width: fill, height: fixed(56), gap: 0 }, [
                  panel({ width: fill, height: fill, fill: "#0B2A1E", padding: 12 }, T("Low", { size: 18, color: C.soft, width: fill })),
                  panel({ width: fill, height: fill, fill: "#143526", padding: 12 }, T("Moderate", { size: 18, color: C.soft, width: fill })),
                  panel({ width: fill, height: fill, fill: "#332714", padding: 12 }, T("High", { size: 18, color: C.white, width: fill })),
                  panel({ width: fill, height: fill, fill: "#391919", padding: 12 }, T("Extreme", { size: 18, color: C.white, width: fill })),
                ]),
                Bar("Day 1", "15.6", 260, C.mint),
                Bar("Day 3", "15.3", 250, C.mint),
                Bar("Day 5", "13.4", 220, C.mint),
                Bar("Day 7", "17.2", 300, C.soft),
                Bar("Day 10", "16.3", 285, C.mint),
              ]),
            ),
          ]),
        ),
        column({ width: fill, height: fill, gap: 22 }, [
          Bullet("Input", "Select Kaggle grid/location and forecast anchor day.", "01"),
          Bullet("Forecast", "Interactive 10-day WBT risk visualization.", "02"),
          Bullet("Explain", "Plain-English feature-driver summary.", "03"),
          Bullet("Advise", "Audience-specific heat action plan.", "04"),
        ]),
      ]),
    ]),
  );
}

function addAdvisory() {
  addSlide(
    Root([
      Title("GENAI ADVISORY", "Forecasts become operational recommendations", "Gemini powers structured advice when GEMINI_API_KEY is configured; a rule-based fallback keeps the app reliable."),
      grid({ width: fill, height: fill, columns: [fr(1.1), fr(0.9)], rows: [fr(1)], columnGap: 42 }, [
        panel(
          { width: fill, height: fill, fill: C.panel, padding: { x: 28, y: 28 } },
          column({ width: fill, height: fill, gap: 10 }, [
            TableRow(["Audience", "Recommended action focus"], [250, 620], { fill: C.panel2, bold: true, color: C.mint, height: 54 }),
            TableRow(["Citizens", "Hydration, shade, symptom awareness"], [250, 620]),
            TableRow(["Hospitals", "Readiness, triage, cooling capacity"], [250, 620]),
            TableRow(["Schools", "Outdoor activity adjustment, hydration"], [250, 620]),
            TableRow(["Workers", "Rest cycles, shaded breaks, supervisor checks"], [250, 620]),
            TableRow(["Authorities", "Messaging, shelters, water points, surge planning"], [250, 620]),
          ]),
        ),
        column({ width: fill, height: fill, gap: 24 }, [
          Metric("Gemini model", "gemini-1.5-flash", "loaded from env secret", C.mint),
          Metric("Fallback", "Rule-based", "runs when key is missing", C.soft),
          panel(
            { width: fill, height: fill, fill: C.panel2, padding: { x: 28, y: 26 } },
            column({ width: fill, height: fill, gap: 14 }, [
              T("Safety guardrails", { size: 31, bold: true, color: C.white, width: fill }),
              T("No fake government alerts. Recommendations mention uncertainty and remain practical for the selected risk level.", {
                size: 27,
                bold: true,
                color: C.muted,
                width: fill,
              }),
            ]),
          ),
        ]),
      ]),
    ]),
  );
}

function addDeployment() {
  addSlide(
    Root([
      Title("FINAL SUBMISSION PACKAGE", "What goes where", "The project separates scoring artifacts, code repository, deployment files, and judge-facing demo materials."),
      panel(
        { width: fill, height: fill, fill: C.panel, padding: { x: 30, y: 30 } },
        column({ width: fill, height: fill, gap: 10 }, [
          TableRow(["Destination", "Upload", "Notes"], [330, 560, 720], { fill: C.panel2, bold: true, color: C.mint, height: 56 }),
          TableRow(["Kaggle", "submission.csv", "Leaderboard scoring file with row_id and target_day_1...10"], [330, 560, 720], { height: 70 }),
          TableRow(["GitHub", "source code + README", "Do not commit .env, Kaggle CSVs, or large model without LFS"], [330, 560, 720], { height: 70 }),
          TableRow(["Hugging Face Spaces", "app.py, src, assets, model artifacts", "Add GEMINI_API_KEY as a Space secret"], [330, 560, 720], { height: 70 }),
          TableRow(["Judges", "PPT + demo video + live link", "Links to be added after GitHub and Hugging Face are created"], [330, 560, 720], { height: 70 }),
          row({ width: fill, height: fill, gap: 24 }, [
            panel({ width: grow(1), height: fill, fill: C.panel2, padding: { x: 28, y: 26 } }, T("GitHub link: <add after upload>", { size: 30, bold: true, color: C.soft, width: fill })),
            panel({ width: grow(1), height: fill, fill: C.panel2, padding: { x: 28, y: 26 } }, T("Hugging Face link: <add after deploy>", { size: 30, bold: true, color: C.soft, width: fill })),
          ]),
        ]),
      ),
    ]),
  );
}

function addClosing() {
  addSlide(
    Root([
      Title("WHY HEATGUARD AI CAN WIN", "Accurate forecast, interpretable risk, deployable action", "The project demonstrates the complete path from Kaggle-grade forecasting to climate-risk decision support."),
      grid({ width: fill, height: fill, columns: [fr(1), fr(1), fr(1)], rows: [fr(1), fr(0.55)], columnGap: 26, rowGap: 28 }, [
        panel({ width: fill, height: fill, fill: C.panel, padding: { x: 30, y: 28 } }, column({ width: fill, height: fill, gap: 20 }, [
          T("01", { size: 24, bold: true, color: C.mint, width: fill }),
          T("Strong ML pipeline", { size: 38, bold: true, color: C.white, width: fill }),
          T("Time-aware validation, 237 engineered features, and saved reusable model artifacts.", { size: 25, color: C.muted, width: fill }),
        ])),
        panel({ width: fill, height: fill, fill: C.panel, padding: { x: 30, y: 28 } }, column({ width: fill, height: fill, gap: 20 }, [
          T("02", { size: 24, bold: true, color: C.mint, width: fill }),
          T("Decision intelligence", { size: 38, bold: true, color: C.white, width: fill }),
          T("Risk categories, model explanation, and audience-specific advisories.", { size: 25, color: C.muted, width: fill }),
        ])),
        panel({ width: fill, height: fill, fill: C.panel, padding: { x: 30, y: 28 } }, column({ width: fill, height: fill, gap: 20 }, [
          T("03", { size: 24, bold: true, color: C.mint, width: fill }),
          T("Demo-ready system", { size: 38, bold: true, color: C.white, width: fill }),
          T("Streamlit UI, Gemini integration, GitHub-ready repo, and HF deployment support.", { size: 25, color: C.muted, width: fill }),
        ])),
        panel({ width: fill, height: fill, fill: C.panel2, padding: { x: 30, y: 28 }, columnSpan: 3 }, row({ width: fill, height: fill, gap: 30 }, [
          T("Final demo flow:", { size: 31, bold: true, color: C.white, width: fixed(270) }),
          T("select location -> view forecast -> inspect risk -> generate advisory -> show deployment links", {
            size: 30,
            bold: true,
            color: C.soft,
            width: fill,
          }),
        ])),
      ]),
    ]),
  );
}

addCover();
addJudgeRequirements();
addProblem();
addChallengeTarget();
addPipeline();
addFeatures();
addModelWorkflow();
addValidation();
addExplainabilityRisk();
addDashboard();
addAdvisory();
addDeployment();
addClosing();

await fs.mkdir(PREVIEW_DIR, { recursive: true });

const pptxBlob = await PresentationFile.exportPptx(presentation);
await pptxBlob.save(PPTX_PATH);

const inspect = await presentation.inspect();
await fs.writeFile(INSPECT_PATH, inspect.ndjson, "utf8");

for (let i = 0; i < presentation.slides.count; i += 1) {
  const slide = presentation.slides.getItem(i);
  const png = await presentation.export({ format: "png", slide });
  const pngPath = path.join(PREVIEW_DIR, `slide_${String(i + 1).padStart(2, "0")}.png`);
  await fs.writeFile(pngPath, Buffer.from(await png.arrayBuffer()));
}

const reloaded = await Presentation.load(PPTX_PATH);
for (let i = 0; i < reloaded.slides.count; i += 1) {
  const slide = reloaded.slides.getItem(i);
  const png = await reloaded.export({ format: "png", slide });
  const pngPath = path.join(PREVIEW_DIR, `pptx_parity_slide_${String(i + 1).padStart(2, "0")}.png`);
  await fs.writeFile(pngPath, Buffer.from(await png.arrayBuffer()));
}

console.log(`PPTX: ${PPTX_PATH}`);
console.log(`Previews: ${PREVIEW_DIR}`);
console.log(`Inspect: ${INSPECT_PATH}`);
