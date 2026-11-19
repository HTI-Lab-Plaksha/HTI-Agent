import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// DOM Elements
const avatarContainer = document.getElementById('avatarContainer');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingText = document.getElementById('loadingText');
const agentStatus = document.getElementById('agentStatus');
const chatMessages = document.getElementById('chatMessages');
const chatForm = document.getElementById('chatForm');
const userPromptInput = document.getElementById('userPromptInput');
const btnCameraHead = document.getElementById('btnCameraHead');
const btnCameraBody = document.getElementById('btnCameraBody');
const btnStudioTrigger = document.getElementById('btnStudioTrigger');
const btnOpenStudio = document.getElementById('btnOpenStudio');
const studioModal = document.getElementById('studioModal');
const btnCloseStudio = document.getElementById('btnCloseStudio');
const modalDropZone = document.getElementById('modalDropZone');
const glbFileInput = document.getElementById('glbFileInput');
const btnToggleVeda = document.getElementById('btnToggleVeda');
const quickChips = document.querySelectorAll('.chip-btn');

let scene = null;
let camera = null;
let renderer = null;
let controls = null;
let currentModel = null;
let clock = new THREE.Clock();

let avatarMesh = null;
let morphMeshes = [];      // every mesh part that carries the mouth blendshapes (skin, lips, mouth interior)
let bones = {};
let initialBoneRotations = {};

// Lip-sync values: [jawOpen, mouthWide, mouthRound, upperLipUp], each 0..1
const MOUTH_SHAPES = ['jawOpen', 'mouthWide', 'mouthRound', 'upperLipUp'];
const BLINK_SHAPE = 'eyeBlink';
let blinkVal = 0;          // 0 = eyes open, 1 = closed
let mouthVals = [0, 0, 0, 0];
let specData = null;
let delayNode = null;
let wsGainNode = null;
let mouthAnalyzer = null;
let proceduralJaw = null;   // only used if the browser cannot decode the audio
const LIP_DELAY = 0.03;

// Natural Eye Blinking (exact static ease curve)
let isBlinking = false;
let blinkStartTime = 0;
let nextBlinkTime = 2.5;

// Interaction & Posture State
let currentState = 'idle'; // 'idle', 'active' (thinking), 'listening', 'speaking'
let mouseX = 0, mouseY = 0;
let targetHeadRotY = 0, targetHeadRotX = 0;
let isPortraitView = true;

let isSpeaking = false;
let speechElapsed = 0;
let currentUtterance = null;
let gestureSwitchInterval = null;

const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);

/** Spectrum (dB per FFT bin) -> mouth shapes [jaw, wide, round, up]. */
class MouthAnalyzer {
  constructor() {
    this.peak = -40;          // running loudest level (dB); floor of -40 keeps hiss/noise silent
    this.hist = [];           // recent frame levels for syllable contrast
    this.cenMed = 1300;       // adaptive brightness centre (Hz)
    this.cenSpread = 350;
    this.vals = [0, 0, 0, 0];
  }

  process(db, binHz, dt) {
    const n = db.length;
    const bin = (hz) => Math.max(1, Math.round(hz / binHz));
    const band = (a, b) => {
      let s = 0;
      const i1 = Math.min(n, bin(b));
      for (let i = bin(a); i < i1; i++) s += Math.pow(10, db[i] / 10);
      return s;
    };
    const eLo = band(120, 900), eMid = band(900, 2300), eHi = band(2300, 4200), eFr = band(4200, 8000);
    const tot = eLo + eMid + eHi + eFr + 1e-20;
    const lev = 10 * Math.log10(tot);

    this.peak = Math.max(lev, this.peak - 4 * dt, -40);
    const gate = this.peak - 24;
    const level = clamp01((lev - gate) / (this.peak - gate + 1e-6));

    const voiced = eLo / tot;
    const fric = (eFr + 0.5 * eHi) / tot;

    // brightness (spectral centroid): bright = "ee", dark = "oo"; F1 region = how open the vowel is
    let cs = 0, cw = 0, fs = 0, fw = 0;
    const c1 = Math.min(n, bin(3600));
    for (let i = bin(200); i < c1; i++) { const p = Math.pow(10, db[i] / 10); cs += p * i * binHz; cw += p; }
    const m1 = Math.min(n, bin(1100));
    for (let i = bin(250); i < m1; i++) { const p = Math.pow(10, db[i] / 10); fs += p * i * binHz; fw += p; }
    const cen = cs / (cw + 1e-20), f1 = fs / (fw + 1e-20);
    if (level > 0.3) {
      const a = Math.min(1, dt / 1.5);
      this.cenMed += (cen - this.cenMed) * a;
      this.cenSpread += (Math.max(150, Math.abs(cen - this.cenMed) * 1.6) - this.cenSpread) * a;
    }
    const z = (cen - this.cenMed) / this.cenSpread;
    const f1n = clamp01((f1 - 380) / 320);

    // syllable contrast: between the recent valley (consonant) and peak (vowel)
    this.hist.push(lev);
    if (this.hist.length > 14) this.hist.shift();
    let mn = 1e9, mx = -1e9;
    for (const v of this.hist) { if (v < mn) mn = v; if (v > mx) mx = v; }
    const rel = clamp01((lev - mn) / (mx - mn + 4));

    const vo = clamp01(voiced * 1.5);
    const jaw = level * (0.22 + 0.48 * rel + 0.30 * f1n) * (0.45 + 0.55 * vo);
    const wide = clamp01(0.75 * z) * level * (0.4 + 0.6 * vo);
    const rnd = clamp01(-0.85 * z) * level * clamp01(voiced * 1.4);
    const up = clamp01(2.2 * fric - 0.15) * clamp01(level * 1.4);
    const target = [0.95 * jaw, 0.55 * wide, 0.75 * rnd, 0.65 * up];

    // light smoothing: fast attack, slightly slower release
    const tau = [[0.012, 0.035], [0.03, 0.05], [0.03, 0.05], [0.02, 0.04]];
    for (let k = 0; k < 4; k++) {
      const t = target[k] > this.vals[k] ? tau[k][0] : tau[k][1];
      this.vals[k] = clamp01(this.vals[k] + (target[k] - this.vals[k]) * (1 - Math.exp(-dt / t)));
    }
    return this.vals;
  }
}

// Agent WebSocket & Web Audio API State
let agentWs = null;
let wsAudioCtx = null;
let wsAudioAnalyser = null;
let activeAudioSource = null;
let activeAudioElement = null;
let audioQueue = [];
let isAudioPlaying = false;
let lastAppendedMessage = { text: '', time: 0 };
let activeSpeechSessionId = 0;
let isVedaActive = false;

function updateVedaToggleButton(active) {
  isVedaActive = Boolean(active);
  if (!btnToggleVeda) return;
  const iconSpan = btnToggleVeda.querySelector('.btn-toggle-icon');
  const textSpan = btnToggleVeda.querySelector('.btn-toggle-text');
  if (isVedaActive) {
    btnToggleVeda.classList.remove('paused');
    btnToggleVeda.classList.add('running');
    if (iconSpan) iconSpan.textContent = '⏹';
    if (textSpan) textSpan.textContent = 'Stop Veda';
    btnToggleVeda.title = "Veda is Active (Watching Camera & Greeting Visitors). Click to Pause.";
  } else {
    btnToggleVeda.classList.remove('running');
    btnToggleVeda.classList.add('paused');
    if (iconSpan) iconSpan.textContent = '▶';
    if (textSpan) textSpan.textContent = 'Start Veda';
    btnToggleVeda.title = "Veda is Paused. Click to Start watching camera and interacting.";
  }
}

// Initialize Web Audio Context for real-time MP3 decoding & lip-sync
function initAudioContext() {
  if (!wsAudioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass) {
      wsAudioCtx = new AudioContextClass();
      wsAudioAnalyser = wsAudioCtx.createAnalyser();
      wsAudioAnalyser.fftSize = 1024;
      wsAudioAnalyser.smoothingTimeConstant = 0;
      wsAudioAnalyser.minDecibels = -120;
      specData = new Float32Array(wsAudioAnalyser.frequencyBinCount);
      mouthAnalyzer = new MouthAnalyzer();

      delayNode = wsAudioCtx.createDelay(0.5);
      delayNode.delayTime.value = LIP_DELAY;

      wsGainNode = wsAudioCtx.createGain();
      wsGainNode.gain.value = 1.0;

      // Source -> Analyser (early) AND Source -> Delay -> Gain -> Speakers
      delayNode.connect(wsGainNode);
      wsGainNode.connect(wsAudioCtx.destination);
    }
  }
  if (wsAudioCtx && wsAudioCtx.state === 'suspended') {
    wsAudioCtx.resume();
  }
}

function base64ToArrayBuffer(base64) {
  const binaryString = window.atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
}

// Initialize 3D Avatar Engine with Black & White Studio Aesthetic
async function initEngine() {
  try {
    loadingText.textContent = "Setting up monochrome 3D environment...";

    const width = avatarContainer.clientWidth || window.innerWidth;
    const height = avatarContainer.clientHeight || window.innerHeight;

    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 50);
    camera.position.set(0, 1.50, 0.62);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.setClearColor(0x0a0a0a, 1);
    avatarContainer.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 0.40;
    controls.maxDistance = 1.15;
    controls.maxPolarAngle = Math.PI / 2;
    controls.minPolarAngle = Math.PI / 3;
    controls.target.set(0, 1.48, 0.05);
    controls.update();

    // Black & White High-Contrast Studio Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.3);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, 2.1);
    keyLight.position.set(2, 4, 3);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 2048;
    keyLight.shadow.mapSize.height = 2048;
    scene.add(keyLight);

    const frontLight = new THREE.DirectionalLight(0xffffff, 1.6);
    frontLight.position.set(0, 1.8, 2.2);
    scene.add(frontLight);

    const fillLight = new THREE.DirectionalLight(0xd4d4d4, 1.1);
    fillLight.position.set(-2, 3, 2);
    scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0xffffff, 2.4);
    rimLight.position.set(0, 3.5, -3);
    scene.add(rimLight);

    // Deep neutral floor shadow receiver
    const shadowGeo = new THREE.PlaneGeometry(3, 3);
    const shadowMat = new THREE.ShadowMaterial({ opacity: 0.35 });
    const shadowMesh = new THREE.Mesh(shadowGeo, shadowMat);
    shadowMesh.rotation.x = -Math.PI / 2;
    shadowMesh.position.y = 0;
    shadowMesh.receiveShadow = true;
    scene.add(shadowMesh);

    // Load initial avatar
    await loadAvatarAndAnimations();

    // Start render loop
    animate();

    // Setup Modal & Drag and Drop Upload
    initStudioModal();
    initDragAndDrop();

    // Connect to Agent WebSocket backend
    connectAgentWebSocket();

    window.addEventListener('resize', onWindowResize);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('click', () => {
      initAudioContext();
      if (wsAudioCtx && wsAudioCtx.state === 'suspended') {
        wsAudioCtx.resume();
      }
    });

  } catch (error) {
    console.error("Engine initialization error:", error);
    loadingText.textContent = "Error initializing 3D viewer. Check browser console.";
  }
}

// Load Avatar and Animations
async function loadAvatarAndAnimations(customUrl = null) {
  const loader = new GLTFLoader();
  const modelUrl = customUrl || ('../final/meera.glb?t=' + Date.now());
  const fallbackUrl = '../final/meera.glb';

  loadingText.textContent = "Loading 3D avatar...";
  loadingOverlay.classList.remove('hidden');

  if (currentModel) {
    scene.remove(currentModel);
    currentModel = null;
    if (mixer) mixer.stopAllAction();
  }

  let gltf;
  try {
    gltf = await loadGLTF(loader, modelUrl);
  } catch (e) {
    console.warn("Could not load model, loading fallback:", e);
    gltf = await loadGLTF(loader, fallbackUrl);
  }

  await applyLoadedGLTF(gltf);
}

async function applyLoadedGLTF(gltf) {
  if (currentModel) {
    scene.remove(currentModel);
  }

  currentModel = gltf.scene;
  currentModel.position.set(0, 0, 0);
  scene.add(currentModel);

  morphMeshes = [];
  bones = {};
  initialBoneRotations = {};

  // Traverse and locate mesh, morph targets, and bones (exact static/app.js rigging)
  currentModel.traverse((child) => {
    if (child.isMesh) {
      avatarMesh = child;
      child.frustumCulled = false;
      child.castShadow = true;
      child.receiveShadow = true;

      if (child.material) {
        child.material.roughness = Math.max(0.35, child.material.roughness || 0.4);
        child.material.envMapIntensity = 1.0;
      }

      // Collect EVERY part that has mouth blendshapes (the model is split in 3 parts)
      if (child.morphTargetDictionary && (MOUTH_SHAPES.some(n => n in child.morphTargetDictionary) || BLINK_SHAPE in child.morphTargetDictionary)) {
        morphMeshes.push(child);
      }
    }

    if (child.isBone) {
      const name = child.name;
      bones[name] = child;
      initialBoneRotations[name] = {
        x: child.rotation.x,
        y: child.rotation.y,
        z: child.rotation.z
      };
    }
  });

  console.log('[AVATAR] Model rigged successfully. Bones count:', Object.keys(bones).length,
              '| mouth-shape meshes:', morphMeshes.length);

  loadingOverlay.classList.add('hidden');
  agentStatus.textContent = "Veda Ready";
}

function loadGLTF(loader, url) {
  return new Promise((resolve, reject) => {
    loader.load(url, resolve, undefined, reject);
  });
}

function playAction(action) {
  if (!action) return;
  if (currentAction && currentAction !== action) {
    currentAction.stop();
  }
  currentAction = action;
  action.reset();
  action.setEffectiveTimeScale(1);
  action.setEffectiveWeight(1);
  action.play();
}

function crossFadeToAction(newAction, duration = 0.25) {
  if (!newAction) return;
  if (newAction === currentAction) {
    newAction.setEffectiveTimeScale(1);
    newAction.setEffectiveWeight(1);
    if (!newAction.isRunning()) newAction.play();
    return;
  }

  const prevAction = currentAction;
  currentAction = newAction;

  newAction.reset();
  newAction.setEffectiveTimeScale(1);
  newAction.setEffectiveWeight(1);
  newAction.fadeIn(duration);
  newAction.play();

  if (prevAction) {
    prevAction.fadeOut(duration);
    setTimeout(() => {
      if (prevAction !== currentAction && prevAction.isRunning()) {
        prevAction.stop();
      }
    }, duration * 1000 + 40);
  }
}

function setCameraPortrait() {
  if (camera && controls) {
    camera.position.set(0, 1.50, 0.62);
    controls.target.set(0, 1.48, 0.05);
    controls.update();
  }
}

function setCameraFullBody() {
  setCameraPortrait();
}

function onMouseMove(e) {
  mouseX = (e.clientX / window.innerWidth) * 2 - 1;
  mouseY = -(e.clientY / window.innerHeight) * 2 + 1;
}

// Append Chat Message
function appendMessage(sender, text) {
  if (!text || !text.trim()) return;
  const bubble = document.createElement('div');
  bubble.className = `message-bubble ${sender}`;

  const meta = document.createElement('div');
  meta.className = 'bubble-meta';

  const name = document.createElement('span');
  name.className = 'sender-name';
  name.textContent = sender === 'agent' ? 'Veda' : 'You';

  const time = document.createElement('span');
  time.className = 'timestamp';
  time.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  meta.appendChild(name);
  meta.appendChild(time);

  const content = document.createElement('p');
  content.className = 'bubble-text';
  content.textContent = text;

  bubble.appendChild(meta);
  bubble.appendChild(content);

  chatMessages.appendChild(bubble);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ================= Agent WebSocket & Web Audio Streaming =================
function connectAgentWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  try {
    agentWs = new WebSocket(wsUrl);

    agentWs.onopen = () => {
      console.log("[Agent WS] Connected to backend agent:", wsUrl);
      agentWs.send(JSON.stringify({
        event: "audio_mode",
        data: { web_audio: true }
      }));
      const statusIndicator = document.querySelector('.status-indicator');
      if (statusIndicator) {
        statusIndicator.classList.remove('speaking');
        statusIndicator.classList.add('online');
      }
      agentStatus.textContent = "Agent Ready • Online";
    };

    agentWs.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        handleAgentEvent(payload);
      } catch (err) {
        console.warn("[Agent WS] Error parsing message:", err);
      }
    };

    agentWs.onclose = () => {
      console.log("[Agent WS] Disconnected. Reconnecting in 2.5s...");
      setTimeout(connectAgentWebSocket, 2500);
    };

    agentWs.onerror = (err) => {
      console.warn("[Agent WS] Socket error:", err);
      agentWs.close();
    };
  } catch (err) {
    console.warn("[Agent WS] Connection attempt failed:", err);
    setTimeout(connectAgentWebSocket, 3000);
  }
}

function handleAgentEvent(msg) {
  const event = msg.event;
  const data = msg.data || {};

  switch (event) {
    case "speak_audio":
      handleSpeakAudio(data);
      break;

    case "user_message":
      if (data.text && (data.text !== lastAppendedMessage.text || (Date.now() - lastAppendedMessage.time > 1500))) {
        lastAppendedMessage = { text: data.text, time: Date.now() };
        appendMessage('user', data.text);
      }
      break;

    case "ai_response":
      if (data.text) {
        appendMessage('agent', data.text);
      }
      break;

    case "state_change":
      const statusIndicator = document.querySelector('.status-indicator');
      if (data.state === "speaking") {
        if (statusIndicator) {
          statusIndicator.classList.remove('online');
          statusIndicator.classList.add('speaking');
        }
        agentStatus.textContent = data.text || "Veda Speaking...";
      } else if (data.state === "active") {
        if (statusIndicator) {
          statusIndicator.classList.remove('speaking');
          statusIndicator.classList.add('online');
        }
        agentStatus.textContent = data.text || "Listening...";
        if (animations.idle && !isAudioPlaying) {
          crossFadeToAction(animations.idle, 0.6);
        }
      } else if (data.state === "idle") {
        if (statusIndicator) {
          statusIndicator.classList.remove('speaking');
          statusIndicator.classList.add('online');
        }
        agentStatus.textContent = "Agent Ready";
        if (animations.idle && !isAudioPlaying) {
          crossFadeToAction(animations.idle, 0.6);
        }
      }
      break;

    case "face_detected":
      if (data.name) {
        agentStatus.textContent = `${data.name} detected`;
      }
      break;

    case "veda_status":
      updateVedaToggleButton(Boolean(data.active));
      if (!data.active) {
        if (activeAudioSource) {
          try { activeAudioSource.stop(); } catch(e) {}
          activeAudioSource = null;
        }
        audioQueue = [];
        isAudioPlaying = false;
        isSpeaking = false;
        const statusIndicator = document.querySelector('.status-indicator');
        if (statusIndicator) {
          statusIndicator.classList.remove('speaking');
          statusIndicator.classList.remove('online');
        }
        agentStatus.textContent = "Agent Paused (Standby)";
        if (animations.idle) {
          crossFadeToAction(animations.idle, 0.4);
        }
      } else {
        const statusIndicator = document.querySelector('.status-indicator');
        if (statusIndicator) {
          statusIndicator.classList.add('online');
        }
        agentStatus.textContent = "Agent Ready • Watching Camera";
      }
      break;
  }
}

async function handleSpeakAudio(payload) {
  if (!payload || !payload.audio_base64) return;
  initAudioContext();

  const b64 = payload.audio_base64.trim();

  audioQueue.push({
    base64: b64,
    text: payload.text,
    duration: payload.duration || 3.0
  });

  if (!isAudioPlaying) {
    playNextAudioInQueue();
  }
}

async function playNextAudioInQueue() {
  if (audioQueue.length === 0) {
    isAudioPlaying = false;
    isSpeaking = false;
    proceduralJaw = null;
    const statusIndicator = document.querySelector('.status-indicator');
    if (statusIndicator) {
      statusIndicator.classList.remove('speaking');
      statusIndicator.classList.add('online');
    }
    agentStatus.textContent = "Agent Ready • Listening";
    if (animations.idle && currentAction !== animations.idle) {
      playAction(animations.idle);
    }
    return;
  }

  isAudioPlaying = true;
  isSpeaking = true;
  speechElapsed = 0;
  proceduralJaw = null;

  const item = audioQueue.shift();

  const statusIndicator = document.querySelector('.status-indicator');
  if (statusIndicator) {
    statusIndicator.classList.remove('online');
    statusIndicator.classList.add('speaking');
  }
  agentStatus.textContent = "Veda Speaking...";

  // Warm smile on greeting
  const textLower = (item.text || '').toLowerCase();
  if (textLower.includes('namaste') || textLower.includes('pranam') || textLower.includes('welcome') || textLower.includes('hello') || textLower.startsWith('hi')) {
    targetSmile = 0.28;
  }

  // Ensure idle posture is active (no distorted arm gestures)
  if (animations.idle && (!currentAction || currentAction !== animations.idle)) {
    playAction(animations.idle);
  }

  // Stop any active buffer source
  if (activeAudioSource) {
    try { activeAudioSource.stop(); } catch (e) {}
    activeAudioSource = null;
  }

  if (!wsAudioCtx) initAudioContext();
  if (wsAudioCtx && wsAudioCtx.state === 'suspended') {
    await wsAudioCtx.resume();
  }

  try {
    const arrayBuffer = base64ToArrayBuffer(item.base64);
    const audioBuffer = await wsAudioCtx.decodeAudioData(arrayBuffer);
    const source = wsAudioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(wsAudioAnalyser);
    source.connect(delayNode);
    activeAudioSource = source;

    source.onended = () => {
      activeAudioSource = null;
      playNextAudioInQueue();
    };

    source.start(0);
  } catch (err) {
    console.warn("[WebAudio] decodeAudioData failed, running procedural cadence fallback:", err);
    const fallbackMs = (item.duration || 3.0) * 1000;
    const start = performance.now();
    const interval = setInterval(() => {
      const elapsed = performance.now() - start;
      if (elapsed >= fallbackMs) {
        clearInterval(interval);
        proceduralJaw = null;
        playNextAudioInQueue();
      } else {
        const cycle = (elapsed / 1000) * 5.5;
        proceduralJaw = Math.max(0, Math.sin(cycle * Math.PI)) * 0.6;
      }
    }, 16);
  }
}

// User Prompt Handler (routes to Pydantic AI agent or local TTS fallback)
function handleUserPrompt(text, gestureKey = null) {
  if (!text || !text.trim()) return;
  text = text.trim();
  userPromptInput.value = '';

  initAudioContext();

  lastAppendedMessage = { text: text, time: Date.now() };
  appendMessage('user', text);

  if (agentWs && agentWs.readyState === WebSocket.OPEN) {
    agentStatus.textContent = "Thinking...";
    if (animations.thinking) {
      crossFadeToAction(animations.thinking, 0.35);
    }
    agentWs.send(JSON.stringify({
      event: "user_text",
      text: text
    }));
  } else {
    // Offline fallback: browser TTS
    speakText(text, gestureKey);
  }
}

// Local Fallback Speak Function (if backend socket is offline)
async function speakText(text, gestureKey = null) {
  if (!text || !text.trim()) return;

  // Never use browser client speech synthesis if backend is active and connected
  if (agentWs && agentWs.readyState === WebSocket.OPEN) {
    return;
  }

  const sessionId = ++activeSpeechSessionId;

  const statusIndicator = document.querySelector('.status-indicator');
  if (statusIndicator) {
    statusIndicator.classList.remove('online');
    statusIndicator.classList.add('speaking');
  }
  agentStatus.textContent = "Veda Speaking...";

  isSpeaking = true;
  speechElapsed = 0;
  if (gestureSwitchInterval) {
    clearInterval(gestureSwitchInterval);
    gestureSwitchInterval = null;
  }

  if (animations.idle && currentAction !== animations.idle) {
    crossFadeToAction(animations.idle, 0.4);
  }
  targetSmile = 0.20;

  if ('speechSynthesis' in window) {
    if (currentUtterance) {
      currentUtterance.onend = null;
      currentUtterance.onerror = null;
      currentUtterance = null;
    }

    try {
      window.speechSynthesis.cancel();
    } catch (e) {}

    const utterance = new SpeechSynthesisUtterance(text);
    currentUtterance = utterance;

    const voices = window.speechSynthesis.getVoices();
    // Strictly select confirmed female voices - NEVER fall back to male voice (like Alex)
    const voice = voices.find(v => {
      const n = v.name.toLowerCase();
      return n.includes('female') || n.includes('samantha') || n.includes('victoria') ||
             n.includes('karen') || n.includes('moira') || n.includes('tessa') ||
             n.includes('zira') || (n.includes('google') && (n.includes('uk english female') || n.includes('hi-in') || n.includes('en-in')));
    });

    if (!voice) {
      console.warn("[TTS] No confirmed female voice found in browser. Skipping voice synthesis to avoid male voice.");
      const approxDuration = Math.max(2.5, text.length * 0.06);
      setTimeout(stopSpeaking, approxDuration * 1000);
      return;
    }
    utterance.voice = voice;

    utterance.rate = 0.98;
    utterance.pitch = 1.04;

    const stopSpeaking = () => {
      if (sessionId !== activeSpeechSessionId) return;
      isSpeaking = false;
      currentUtterance = null;

      if (statusIndicator) {
        statusIndicator.classList.remove('speaking');
        statusIndicator.classList.add('online');
      }
      agentStatus.textContent = "Agent Ready";

      if (animations.idle && currentAction !== animations.namaste) {
        crossFadeToAction(animations.idle, 0.6);
      }
    };

    utterance.onend = stopSpeaking;
    utterance.onerror = stopSpeaking;

    setTimeout(() => {
      if (sessionId === activeSpeechSessionId) {
        window.speechSynthesis.speak(utterance);
      }
    }, 25);
  }
}

// Render Loop (Exact static/app.js physics: FFT mouth, ease eye blinking, spine breathing & mouse gaze)
function animate() {
  requestAnimationFrame(animate);

  const delta = clock.getDelta();
  const elapsedTime = clock.getElapsedTime();

  if (controls) controls.update();

  // 1. Real-Time Lip Sync (analyses the audio BEFORE it reaches the speakers)
  const dt = Math.min(Math.max(delta, 0.001), 0.1);
  if (proceduralJaw !== null) {
    const j = proceduralJaw;
    const t = [j, 0.2 * j, 0, 0];
    for (let k = 0; k < 4; k++) mouthVals[k] += (t[k] - mouthVals[k]) * 0.35;
  } else if (wsAudioAnalyser && mouthAnalyzer && (isSpeaking || isAudioPlaying)) {
    wsAudioAnalyser.getFloatFrequencyData(specData);
    mouthVals = mouthAnalyzer.process(specData, wsAudioCtx.sampleRate / wsAudioAnalyser.fftSize, dt);
  } else {
    for (let k = 0; k < 4; k++) mouthVals[k] *= 0.8;
  }

  // Apply the mouth shapes to every part of the model that has them
  for (const m of morphMeshes) {
    for (let k = 0; k < 4; k++) {
      const idx = m.morphTargetDictionary[MOUTH_SHAPES[k]];
      if (idx !== undefined) m.morphTargetInfluences[idx] = mouthVals[k];
    }
  }

  // 2. Natural eye blinking (eyelids are part of avatar.glb, driven by the 'eyeBlink' shape)
  //    close ~70 ms, hold ~30 ms, open ~120 ms; every 2.5-5 s; about 1 in 5 is a quick double blink
  if (!isBlinking && elapsedTime >= nextBlinkTime) {
    isBlinking = true;
    blinkStartTime = elapsedTime;
    const double = Math.random() < 0.2;
    nextBlinkTime = double ? elapsedTime + 0.40 : elapsedTime + 2.5 + Math.random() * 2.5;
  }
  if (isBlinking) {
    const t = elapsedTime - blinkStartTime;
    const ease = (x) => x * x * (3 - 2 * x);
    if (t < 0.07) blinkVal = ease(t / 0.07);
    else if (t < 0.10) blinkVal = 1;
    else if (t < 0.22) blinkVal = 1 - ease((t - 0.10) / 0.12);
    else { blinkVal = 0; isBlinking = false; }
  }
  for (const m of morphMeshes) {
    const bi = m.morphTargetDictionary[BLINK_SHAPE];
    if (bi !== undefined) m.morphTargetInfluences[bi] = blinkVal;
  }

  // 3. Natural Breathing & Organic Life Dynamics
  const breathRate = 1.4;
  const breathCycle = Math.sin(elapsedTime * breathRate);

  // Spine breathing expansion
  const spine = bones['mixamorig:Spine2'] || bones['mixamorig:Spine1'];
  if (spine && initialBoneRotations[spine.name]) {
    spine.rotation.x = initialBoneRotations[spine.name].x + breathCycle * 0.014;
  }

  // Head subtle micro-movements & mouse parallax
  const head = bones['mixamorig:Head'];
  if (head && initialBoneRotations['mixamorig:Head']) {
    const init = initialBoneRotations['mixamorig:Head'];

    let stateOffsetX = 0;
    let stateOffsetY = 0;
    let stateOffsetZ = 0;

    if (currentState === 'listening') {
      stateOffsetX = 0.035;
      stateOffsetZ = 0.03;
    } else if (currentState === 'active') {
      stateOffsetX = -0.025;
      stateOffsetY = 0.02;
    } else if (isSpeaking || isAudioPlaying) {
      stateOffsetX = Math.sin(elapsedTime * 6.0) * (0.015 + mouthVals[0] * 0.025);
      stateOffsetZ = Math.cos(elapsedTime * 3.0) * 0.01;
    }

    targetHeadRotY = init.y + (mouseX * 0.15) + stateOffsetY;
    targetHeadRotX = init.x + (mouseY * 0.08) + stateOffsetX + breathCycle * 0.006;
    const targetHeadRotZ = init.z + stateOffsetZ;

    head.rotation.y += (targetHeadRotY - head.rotation.y) * 0.06;
    head.rotation.x += (targetHeadRotX - head.rotation.x) * 0.06;
    head.rotation.z += (targetHeadRotZ - head.rotation.z) * 0.06;
  }

  renderer.render(scene, camera);
}

function onWindowResize() {
  const w = avatarContainer.clientWidth;
  const h = avatarContainer.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}

// Modal and Upload System
function initStudioModal() {
  btnOpenStudio.addEventListener('click', () => studioModal.classList.remove('hidden'));
  btnStudioTrigger.addEventListener('click', () => studioModal.classList.remove('hidden'));
  btnCloseStudio.addEventListener('click', () => studioModal.classList.add('hidden'));
}

function initDragAndDrop() {
  const dropBox = document.querySelector('.drop-zone-box');

  ['dragenter', 'dragover'].forEach(name => {
    dropBox.addEventListener(name, (e) => {
      e.preventDefault();
      dropBox.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropBox.addEventListener(name, (e) => {
      e.preventDefault();
      dropBox.classList.remove('dragover');
    });
  });

  dropBox.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) handleAvatarFile(files[0]);
  });

  glbFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleAvatarFile(e.target.files[0]);
  });
}

async function handleAvatarFile(file) {
  if (!file.name.toLowerCase().endsWith('.glb') && !file.name.toLowerCase().endsWith('.gltf')) {
    alert("Please upload a .glb 3D avatar file.");
    return;
  }

  studioModal.classList.add('hidden');
  loadingText.textContent = "Processing and loading new HD avatar...";
  loadingOverlay.classList.remove('hidden');

  try {
    const arrayBuffer = await file.arrayBuffer();

    fetch('/api/upload-avatar', {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: arrayBuffer
    }).then(res => res.json()).then(res => console.log("[Backend] Avatar saved:", res));

    const loader = new GLTFLoader();
    loader.parse(
      arrayBuffer,
      '',
      (gltf) => {
        applyLoadedGLTF(gltf);
        console.log("New HD avatar successfully applied!");
      },
      (err) => {
        console.error("Error parsing GLB file:", err);
        loadingOverlay.classList.add('hidden');
        alert("Failed to parse 3D file. Please ensure it is a valid GLB.");
      }
    );

  } catch (err) {
    console.error("Error handling file:", err);
    loadingOverlay.classList.add('hidden');
  }
}

// Event Listeners
chatForm.addEventListener('submit', (e) => {
  e.preventDefault();
  handleUserPrompt(userPromptInput.value);
});

if (btnCameraHead) btnCameraHead.addEventListener('click', setCameraPortrait);
if (btnCameraBody) btnCameraBody.addEventListener('click', setCameraPortrait);

// Global user interaction unlock for browser audio
function unlockAudioPlayback() {
  if (wsAudioCtx && wsAudioCtx.state === 'suspended') {
    wsAudioCtx.resume();
  }
}
window.addEventListener('click', unlockAudioPlayback);
window.addEventListener('keydown', unlockAudioPlayback);
window.addEventListener('touchstart', unlockAudioPlayback);

if (btnToggleVeda) {
  btnToggleVeda.addEventListener('click', () => {
    initAudioContext();
    const nextState = !isVedaActive;
    updateVedaToggleButton(nextState);
    if (agentWs && agentWs.readyState === WebSocket.OPEN) {
      agentWs.send(JSON.stringify({
        event: "toggle_active",
        data: { active: nextState }
      }));
    } else {
      fetch('/api/toggle-active', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ active: nextState })
      }).catch(err => console.warn('[Toggle Veda API error]', err));
    }
  });
}

quickChips.forEach(chip => {
  chip.addEventListener('click', () => {
    const text = chip.getAttribute('data-text');
    handleUserPrompt(text);
  });
});

window.addEventListener('DOMContentLoaded', () => {
  initEngine();
});
