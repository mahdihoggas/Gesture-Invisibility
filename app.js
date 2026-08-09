import {
  FilesetResolver,
  HandLandmarker,
  ImageSegmenter,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const LM = {
  WRIST: 0,
  INDEX_MCP: 5,
  INDEX_TIP: 8,
  MIDDLE_MCP: 9,
  MIDDLE_TIP: 12,
  RING_MCP: 13,
  RING_TIP: 16,
  PINKY_MCP: 17,
  PINKY_TIP: 20,
};

const LOAD_TIMEOUT_MS = 25000;
const PERSON_THRESHOLD = 0.45;
const BODY_BOX_PADDING = 25;
const BODY_MASK_ERODE = 1;
const BODY_TRIM_PERCENT = 0.02;
const BODY_COL_ROW_FILL = 0.12;
const HEAD_EXTRA_PAD = 40;
const BOX_POS_SMOOTH = 0.55;
const BOX_SIZE_SMOOTH = 0.28;
const BODY_LOST_MAX_FRAMES = 8;
const FIST_HOLD_FRAMES = 6;
const OPEN_HOLD_FRAMES = 4;

const videoEl = document.getElementById("webcam");
const canvas = document.getElementById("sceneCanvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const modeBadge = document.getElementById("modeBadge");
const modeIcon = document.getElementById("modeIcon");
const modeLabel = document.getElementById("modeLabel");
const promptText = document.getElementById("promptText");


const setupOverlay = document.getElementById("setupOverlay");
const bgUpload = document.getElementById("bgUpload");
const captureBgBtn = document.getElementById("captureBgBtn");
const bgPreviewWrap = document.getElementById("bgPreviewWrap");
const bgPreview = document.getElementById("bgPreview");
const startBtn = document.getElementById("startBtn");

const loadingOverlay = document.getElementById("loadingOverlay");
const loaderText = document.getElementById("loaderText");
const loaderRetry = document.getElementById("loaderRetry");

let handLandmarker = null;
let imageSegmenter = null;

let invisible = false;
let fistFrames = 0;
let openFrames = 0;
let backgroundReady = false;
let appStarted = false;
let lastBodyBox = null;
let bodyLostFrames = 0;

let bgCanvas = null;
let bgCtx = null;

const frameCanvas = document.createElement("canvas");
const frameCtx = frameCanvas.getContext("2d", { willReadFrequently: true });

function dist2D(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function isFingerExtended(landmarks, tipIdx, mcpIdx) {
  const wrist = landmarks[LM.WRIST];
  return dist2D(landmarks[tipIdx], wrist) > dist2D(landmarks[mcpIdx], wrist) * 1.06;
}

function isFingerCurled(landmarks, tipIdx, mcpIdx) {
  const wrist = landmarks[LM.WRIST];
  return dist2D(landmarks[tipIdx], wrist) < dist2D(landmarks[mcpIdx], wrist) * 0.92;
}

function isFist(landmarks) {
  const pairs = [
    [LM.INDEX_TIP, LM.INDEX_MCP],
    [LM.MIDDLE_TIP, LM.MIDDLE_MCP],
    [LM.RING_TIP, LM.RING_MCP],
    [LM.PINKY_TIP, LM.PINKY_MCP],
  ];
  let curled = 0;
  for (const [tip, mcp] of pairs) {
    if (isFingerCurled(landmarks, tip, mcp)) curled++;
  }
  return curled >= 4;
}

function isOpenPalm(landmarks) {
  return (
    isFingerExtended(landmarks, LM.INDEX_TIP, LM.INDEX_MCP) &&
    isFingerExtended(landmarks, LM.MIDDLE_TIP, LM.MIDDLE_MCP) &&
    isFingerExtended(landmarks, LM.RING_TIP, LM.RING_MCP) &&
    isFingerExtended(landmarks, LM.PINKY_TIP, LM.PINKY_MCP)
  );
}

function updateBgStatusUI() {}

function updateModeUI() {
  if (invisible) {
    modeBadge.className = "mode-badge vanish-mode";
    modeIcon.textContent = "✊";
    modeLabel.textContent = "Vanished";
    statusDot.className = "status-dot vanish";
    statusText.textContent = "you are invisible";
    promptText.textContent = "Open your hand to come back · purple box still tracks you";
  } else {
    modeBadge.className = "mode-badge visible-mode";
    modeIcon.textContent = "✋";
    modeLabel.textContent = "Visible";
    statusDot.className = "status-dot live";
    statusText.textContent = "you are visible";
    promptText.textContent = "Close fist to vanish · yellow box tracks your body";
  }
}

function updateGestureState(hands) {
  if (!hands.length || !backgroundReady || !appStarted) return;

  const landmarks = hands[0];
  if (isFist(landmarks)) {
    fistFrames++;
    openFrames = 0;
    if (fistFrames >= FIST_HOLD_FRAMES) invisible = true;
  } else if (isOpenPalm(landmarks)) {
    openFrames++;
    fistFrames = 0;
    if (openFrames >= OPEN_HOLD_FRAMES) invisible = false;
  } else {
    fistFrames = Math.max(0, fistFrames - 1);
    openFrames = Math.max(0, openFrames - 1);
  }

  updateModeUI();
}

function fitCanvasToWindow() {
  const stageEl = document.getElementById("stage");
  const vw = stageEl.clientWidth;
  const vh = stageEl.clientHeight;
  const videoAspect = canvas.width / canvas.height;
  const containerAspect = vw / vh;

  let cssWidth;
  let cssHeight;
  if (containerAspect > videoAspect) {
    cssWidth = vw;
    cssHeight = vw / videoAspect;
  } else {
    cssHeight = vh;
    cssWidth = vh * videoAspect;
  }

  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;
}

window.addEventListener("resize", fitCanvasToWindow);

function initFrameBuffer(w, h) {
  frameCanvas.width = w;
  frameCanvas.height = h;

  bgCanvas = document.createElement("canvas");
  bgCanvas.width = w;
  bgCanvas.height = h;
  bgCtx = bgCanvas.getContext("2d", { willReadFrequently: true });
}

function drawCoverImage(targetCtx, image, w, h) {
  const imgAspect = image.width / image.height;
  const canvasAspect = w / h;

  let sx = 0;
  let sy = 0;
  let sw = image.width;
  let sh = image.height;

  if (imgAspect > canvasAspect) {
    sw = image.height * canvasAspect;
    sx = (image.width - sw) / 2;
  } else {
    sh = image.width / canvasAspect;
    sy = (image.height - sh) / 2;
  }

  targetCtx.drawImage(image, sx, sy, sw, sh, 0, 0, w, h);
}

function setBackgroundFromImage(image) {
  if (!bgCanvas) return;

  const w = bgCanvas.width;
  const h = bgCanvas.height;

  bgCtx.save();
  bgCtx.translate(w, 0);
  bgCtx.scale(-1, 1);
  drawCoverImage(bgCtx, image, w, h);
  bgCtx.restore();

  backgroundReady = true;
  startBtn.disabled = false;
  updateBgStatusUI();

  bgPreview.src = bgCanvas.toDataURL("image/jpeg", 0.85);
  bgPreviewWrap.classList.remove("hidden");
}

function setBackgroundFromFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    const img = new Image();
    img.onload = () => setBackgroundFromImage(img);
    img.src = reader.result;
  };
  reader.readAsDataURL(file);
}

function captureBackgroundFromCamera() {
  if (!videoEl.videoWidth) return;

  const snap = document.createElement("canvas");
  snap.width = videoEl.videoWidth;
  snap.height = videoEl.videoHeight;
  const snapCtx = snap.getContext("2d");
  snapCtx.drawImage(videoEl, 0, 0);

  const img = new Image();
  img.onload = () => setBackgroundFromImage(img);
  img.src = snap.toDataURL("image/jpeg", 0.92);
}

async function initWebcam() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("This browser does not support getUserMedia.");
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
    audio: false,
  });
  videoEl.srcObject = stream;

  await new Promise((resolve) => {
    videoEl.onloadedmetadata = () => {
      videoEl.play();
      resolve();
    };
  });

  canvas.width = videoEl.videoWidth;
  canvas.height = videoEl.videoHeight;
  fitCanvasToWindow();
  initFrameBuffer(canvas.width, canvas.height);
}

function withTimeout(promise, ms, timeoutMessage) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(timeoutMessage)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

async function initHandLandmarker(vision) {
  try {
    return await withTimeout(
      HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.6,
        minHandPresenceConfidence: 0.6,
        minTrackingConfidence: 0.6,
      }),
      LOAD_TIMEOUT_MS,
      "Timed out loading HandLandmarker model."
    );
  } catch (gpuErr) {
    console.warn("[VanishCam] GPU HandLandmarker failed, retrying CPU…", gpuErr);
    return withTimeout(
      HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "CPU",
        },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.6,
        minHandPresenceConfidence: 0.6,
        minTrackingConfidence: 0.6,
      }),
      LOAD_TIMEOUT_MS,
      "Timed out loading HandLandmarker model with CPU."
    );
  }
}

async function initImageSegmenter(vision) {
  try {
    return await withTimeout(
      ImageSegmenter.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        outputCategoryMask: false,
        outputConfidenceMasks: true,
      }),
      LOAD_TIMEOUT_MS,
      "Timed out loading body segmenter model."
    );
  } catch (gpuErr) {
    console.warn("[VanishCam] GPU segmenter failed, retrying CPU…", gpuErr);
    return withTimeout(
      ImageSegmenter.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite",
          delegate: "CPU",
        },
        runningMode: "VIDEO",
        outputCategoryMask: false,
        outputConfidenceMasks: true,
      }),
      LOAD_TIMEOUT_MS,
      "Timed out loading body segmenter model with CPU."
    );
  }
}

function getBodyBoxFromMask(mask, maskW, maskH, canvasW, canvasH) {
  const xs = [];
  const ys = [];

  for (let y = 0; y < maskH; y++) {
    for (let x = 0; x < maskW; x++) {
      if (mask[y * maskW + x] > PERSON_THRESHOLD) {
        xs.push(x);
        ys.push(y);
      }
    }
  }

  if (xs.length < 100) return null;

  xs.sort((a, b) => a - b);
  ys.sort((a, b) => a - b);

  const xStart = Math.floor(xs.length * BODY_TRIM_PERCENT);
  const xEnd = Math.floor(xs.length * (1 - BODY_TRIM_PERCENT)) - 1;
  const yStart = Math.floor(ys.length * BODY_TRIM_PERCENT);
  const yEnd = Math.floor(ys.length * (1 - BODY_TRIM_PERCENT)) - 1;

  let minX = xs[xStart];
  let maxX = xs[Math.max(xStart, xEnd)];
  let minY = ys[yStart];
  let maxY = ys[Math.max(yStart, yEnd)];

  const colCounts = new Uint16Array(maskW);
  const rowCounts = new Uint16Array(maskH);
  for (let i = 0; i < xs.length; i++) {
    colCounts[xs[i]]++;
    rowCounts[ys[i]]++;
  }

  let peakCol = 0;
  let peakRow = 0;
  for (let x = minX; x <= maxX; x++) peakCol = Math.max(peakCol, colCounts[x]);
  for (let y = minY; y <= maxY; y++) peakRow = Math.max(peakRow, rowCounts[y]);

  const colMin = peakCol * BODY_COL_ROW_FILL;
  const rowMin = peakRow * BODY_COL_ROW_FILL;

  while (minX < maxX && colCounts[minX] < colMin) minX++;
  while (minX < maxX && colCounts[maxX] < colMin) maxX--;
  while (minY < maxY && rowCounts[minY] < rowMin) minY++;
  while (minY < maxY && rowCounts[maxY] < rowMin) maxY--;

  minX = Math.min(maskW - 1, minX + BODY_MASK_ERODE);
  minY = Math.min(maskH - 1, minY + BODY_MASK_ERODE);
  maxX = Math.max(0, maxX - BODY_MASK_ERODE);
  maxY = Math.max(0, maxY - BODY_MASK_ERODE);
  if (maxX <= minX || maxY <= minY) return null;

  const scaleX = canvasW / maskW;
  const scaleY = canvasH / maskH;

  const videoMinX = minX * scaleX;
  const videoMaxX = (maxX + 1) * scaleX;
  const videoMinY = minY * scaleY;
  const videoMaxY = (maxY + 1) * scaleY;

  const mirroredX = canvasW - videoMaxX;
  const width = videoMaxX - videoMinX;
  const height = videoMaxY - videoMinY;

  const bx = Math.max(0, mirroredX - BODY_BOX_PADDING);
  const by = Math.max(0, videoMinY - BODY_BOX_PADDING - HEAD_EXTRA_PAD);
  const bw = Math.min(canvasW - bx, width + BODY_BOX_PADDING * 2);
  const bh = Math.min(canvasH - by, height + BODY_BOX_PADDING * 2 + HEAD_EXTRA_PAD);

  return { x: bx, y: by, width: bw, height: bh };
}

function smoothBodyBox(next) {
  if (!next) {
    bodyLostFrames++;
    if (bodyLostFrames > BODY_LOST_MAX_FRAMES) lastBodyBox = null;
    return lastBodyBox;
  }

  bodyLostFrames = 0;
  if (!lastBodyBox) {
    lastBodyBox = { ...next };
    return lastBodyBox;
  }

  const smooth = (prev, target, factor) => prev + (target - prev) * factor;
  let x = smooth(lastBodyBox.x, next.x, BOX_POS_SMOOTH);
  let y = smooth(lastBodyBox.y, next.y, BOX_POS_SMOOTH);
  let width = smooth(lastBodyBox.width, next.width, BOX_SIZE_SMOOTH);
  let height = smooth(lastBodyBox.height, next.height, BOX_SIZE_SMOOTH);

  width = Math.min(width, next.width * 1.03);
  height = Math.min(height, next.height * 1.03);

  lastBodyBox = { x, y, width, height };
  return lastBodyBox;
}

function detectBodyBox(nowMs, w, h) {
  if (!imageSegmenter) return null;

  const segResult = imageSegmenter.segmentForVideo(videoEl, nowMs);
  const maskObj = segResult.confidenceMasks?.[0];
  const mask = maskObj?.getAsFloat32Array();
  if (!mask || !maskObj.width || !maskObj.height) return smoothBodyBox(null);

  const rawBox = getBodyBoxFromMask(mask, maskObj.width, maskObj.height, w, h);
  return smoothBodyBox(rawBox);
}

function drawBodyDetector(box) {
  const { x, y, width, height } = box;
  const color = invisible ? "#9b59d4" : "#f5c518";
  const label = invisible ? "BODY · INVISIBLE" : "BODY DETECTED";

  ctx.save();
  ctx.shadowColor = color;
  ctx.shadowBlur = invisible ? 12 : 6;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash(invisible ? [8, 5] : []);
  ctx.strokeRect(x, y, width, height);
  ctx.setLineDash([]);

  const cornerLen = Math.min(36, width * 0.18, height * 0.18);
  ctx.lineWidth = 3;
  ctx.shadowBlur = 8;
  const corners = [
    [x, y, 1, 1],
    [x + width, y, -1, 1],
    [x, y + height, 1, -1],
    [x + width, y + height, -1, -1],
  ];
  for (const [cx, cy, dx, dy] of corners) {
    ctx.beginPath();
    ctx.moveTo(cx, cy + cornerLen * dy);
    ctx.lineTo(cx, cy);
    ctx.lineTo(cx + cornerLen * dx, cy);
    ctx.stroke();
  }

  ctx.shadowBlur = 0;
  ctx.font = "11px 'IBM Plex Mono', monospace";
  ctx.fillStyle = color;
  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  ctx.fillText(label, x + 4, Math.max(14, y - 6));
  ctx.restore();
}

function drawMirroredFrame(targetCtx, w, h) {
  targetCtx.save();
  targetCtx.translate(w, 0);
  targetCtx.scale(-1, 1);
  targetCtx.drawImage(videoEl, 0, 0, w, h);
  targetCtx.restore();
}

function renderFrame(nowMs) {
  const w = canvas.width;
  const h = canvas.height;
  const box = detectBodyBox(nowMs, w, h);

  if (invisible && backgroundReady) {
    ctx.drawImage(bgCanvas, 0, 0, w, h);
  } else {
    drawMirroredFrame(frameCtx, w, h);
    ctx.drawImage(frameCanvas, 0, 0, w, h);
  }

  if (box) drawBodyDetector(box);
}

function renderLoop() {
  if (videoEl.readyState >= 2 && handLandmarker && appStarted) {
    const nowMs = performance.now();
    const handResult = handLandmarker.detectForVideo(videoEl, nowMs);
    updateGestureState(handResult.landmarks || []);
    renderFrame(nowMs);
  }
  requestAnimationFrame(renderLoop);
}

function showSetupOverlay() {
  setupOverlay.classList.remove("hidden");
  appStarted = false;
  invisible = false;
  updateModeUI();
}

function hideSetupOverlay() {
  setupOverlay.classList.add("hidden");
  appStarted = true;
  invisible = false;
  lastBodyBox = null;
  bodyLostFrames = 0;
  fistFrames = 0;
  openFrames = 0;
  updateModeUI();
  statusText.textContent = "ready — open hand to appear";
}

function showLoaderError(message) {
  loaderText.textContent = message;
  loaderText.style.color = "#e0533d";
  loaderRetry.classList.remove("hidden");
}

function resetLoaderUI() {
  loadingOverlay.classList.remove("hidden");
  loaderText.style.color = "";
  loaderText.textContent = "loading AI models…";
  loaderRetry.classList.add("hidden");
}

async function boot() {
  resetLoaderUI();
  invisible = false;
  lastBodyBox = null;
  bodyLostFrames = 0;
  fistFrames = 0;
  openFrames = 0;
  backgroundReady = false;
  appStarted = false;
  startBtn.disabled = true;
  bgPreviewWrap.classList.add("hidden");
  updateBgStatusUI();
  updateModeUI();

  let settled = false;
  const watchdog = setTimeout(() => {
    if (!settled) {
      showLoaderError("Loading is taking too long. Press retry or check your connection.");
    }
  }, LOAD_TIMEOUT_MS * 3);

  try {
    if (!videoEl.srcObject) await initWebcam();

    loaderText.textContent = "loading hand + body AI models…";

    const vision = await withTimeout(
      FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      ),
      LOAD_TIMEOUT_MS,
      "Timed out loading MediaPipe runtime."
    );

    handLandmarker = await initHandLandmarker(vision);
    imageSegmenter = await initImageSegmenter(vision);

    settled = true;
    clearTimeout(watchdog);
    loadingOverlay.classList.add("hidden");
    showSetupOverlay();
    requestAnimationFrame(renderLoop);
  } catch (err) {
    settled = true;
    clearTimeout(watchdog);
    if (err?.name === "NotAllowedError") {
      showLoaderError("Camera permission denied. Enable it in browser settings and press retry.");
    } else if (err?.name === "NotFoundError") {
      showLoaderError("No webcam found.");
    } else {
      showLoaderError(err?.message || "Error starting the app.");
    }
  }
}

bgUpload.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  if (file) setBackgroundFromFile(file);
});

captureBgBtn.addEventListener("click", captureBackgroundFromCamera);

startBtn.addEventListener("click", () => {
  if (backgroundReady) hideSetupOverlay();
});


const passcodeOverlay = document.getElementById("passcodeOverlay");
const passcodeForm = document.getElementById("passcodeForm");
const passcodeInput = document.getElementById("passcodeInput");
const passcodeError = document.getElementById("passcodeError");

function verifyPasscode(e) {
  if (e) e.preventDefault();
  const code = passcodeInput.value.trim();
  if (code === "1609") {
    passcodeOverlay.remove();
    boot();
  } else {
    passcodeError.textContent = "Incorrect passcode";
    passcodeInput.style.borderColor = "#e0533d";
    passcodeInput.value = "";
    passcodeInput.focus();
  }
}

if (passcodeForm) {
  passcodeForm.addEventListener("submit", verifyPasscode);
}
if (passcodeInput) {
  passcodeInput.focus();
}


