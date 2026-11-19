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
const btnWave = document.getElementById('btnWave');
const btnGestureExplain = document.getElementById('btnGestureExplain');
const btnGestureWelcome = document.getElementById('btnGestureWelcome');
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
let mixer = null;
let clock = new THREE.Clock();

const animations = {
  idle: null,
  namaste: null,
  talking: null,
  talking_explain: null,
  talking_bothhands: null,
  talking_welcome: null,
  thinking: null,
  waving: null,
  thankful: null
};

let currentAction = null;
let facialMeshes = [];

// Speech & Natural Mouth Cadence State
let isSpeaking = false;
let speechElapsed = 0;
let currentUtterance = null;
let gestureSwitchInterval = null;

// Target and Current Morph Weights for Smooth 60FPS Lerping
let targetJawOpen = 0;
let targetVisemeAA = 0;
let targetVisemeO = 0;
let targetSmile = 0.08;

let currentJawOpen = 0;
let currentVisemeAA = 0;
let currentVisemeO = 0;
let currentSmile = 0.08;

// Natural Eye Blink State
let blinkTimer = 2.5 + Math.random() * 2.5;
let isBlinking = false;
let blinkProgress = 0;

// Agent WebSocket & Web Audio API State
let agentWs = null;
let wsAudioCtx = null;
let wsAudioAnalyser = null;
let wsAudioDataArray = null;
let activeAudioSource = null;
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
      wsAudioAnalyser.fftSize = 256;
      wsAudioAnalyser.smoothingTimeConstant = 0.45;
      wsAudioDataArray = new Uint8Array(wsAudioAnalyser.frequencyBinCount);
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

    camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100);
    setCameraPortrait();

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.35;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.setClearColor(0x0a0a0a, 1);
    avatarContainer.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 1.48, 0);
    controls.maxPolarAngle = Math.PI / 2 + 0.1;
    controls.minDistance = 0.6;
    controls.maxDistance = 1.35;

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
    if (mixer) mixer.stopAllAction();
  }

  currentModel = gltf.scene;
  facialMeshes = [];

  // Normalize bone names for seamless Mixamo/ARKit track binding
  currentModel.traverse((child) => {
    if (child.name) {
      child.name = child.name.replace(/:/g, '_');
    }
    if (child.isMesh) {
      child.castShadow = true;
      child.receiveShadow = true;
      if (child.material) {
        child.material.flatShading = false;
        child.material.needsUpdate = true;
      }
      if (child.morphTargetDictionary && child.morphTargetInfluences) {
        facialMeshes.push(child);
      }
    }
  });

  // Reset animations state
  for (const k in animations) {
    animations[k] = null;
  }
  currentAction = null;

  scene.add(currentModel);
  mixer = new THREE.AnimationMixer(currentModel);

  // 1. Bind native embedded animations from meera.glb
  if (gltf.animations && gltf.animations.length > 0) {
    gltf.animations.forEach((clip) => {
      clip.tracks.forEach(track => {
        if (track.name.includes(':')) {
          track.name = track.name.replace(/:/g, '_');
        }
      });
      const baseName = clip.name.split('.')[0].toLowerCase();
      if (animations.hasOwnProperty(baseName) && !animations[baseName]) {
        console.log(`[Native Embedded Animation Bound]: ${clip.name} -> animations.${baseName}`);
        const action = mixer.clipAction(clip);
        if (baseName === 'idle' || baseName.startsWith('talking')) {
          action.setLoop(THREE.LoopRepeat);
        } else {
          action.setLoop(THREE.LoopOnce);
          action.clampWhenFinished = true;
        }
        animations[baseName] = action;
      }
    });
  }

  // 2. Load animation clips from anims folder
  await loadAnimationClips();

  // If idle animation was bound, play it immediately
  if (!currentAction && animations.idle) {
    playAction(animations.idle);
  }

  loadingOverlay.classList.add('hidden');
  agentStatus.textContent = facialMeshes.length > 0 ? "HD Persona Active • Ready" : "Avatar Active • Gestures Ready";
}

// Helper to set morph target value synchronously across all facial meshes
function setMorphInfluence(targetName, value) {
  for (let i = 0; i < facialMeshes.length; i++) {
    const mesh = facialMeshes[i];
    const idx = mesh.morphTargetDictionary[targetName];
    if (idx !== undefined) {
      mesh.morphTargetInfluences[idx] = value;
    }
  }
}

async function loadAnimationClips() {
  const loader = new GLTFLoader();
  const cacheBust = Date.now();
  const animNames = [
    { key: 'idle', file: `../final/anims/Idle.glb?t=${cacheBust}` },
    { key: 'waving', file: `../final/anims/Waving.glb?t=${cacheBust}` }
  ];

  for (const anim of animNames) {
    if (animations[anim.key]) continue;
    try {
      const animGltf = await loadGLTF(loader, anim.file);
      if (animGltf.animations && animGltf.animations.length > 0) {
        let clip = animGltf.animations.find(a => a.name.toLowerCase().includes(anim.key.toLowerCase())) 
                || animGltf.animations[0];
                
        clip = retargetClip(clip, currentModel);
        const action = mixer.clipAction(clip);
        
        if (anim.key === 'idle' || anim.key.startsWith('talking')) {
          action.setLoop(THREE.LoopRepeat);
        } else {
          action.setLoop(THREE.LoopOnce);
          action.clampWhenFinished = true;
        }

        animations[anim.key] = action;
      }
    } catch (err) {
      console.warn(`Could not load animation ${anim.key}:`, err);
    }
  }

  mixer.addEventListener('finished', (e) => {
    if (e.action !== animations.idle && animations.idle) {
      crossFadeToAction(animations.idle, 0.6);
    }
  });

  if (!currentAction && animations.idle) {
    playAction(animations.idle);
  }

  window.animations = animations;
  window.playAction = playAction;
  window.crossFadeToAction = crossFadeToAction;
}

// Retarget animation tracks from Mixamo or custom clips to Meera's skeleton
function retargetClip(clip, model) {
  if (!clip || !clip.tracks || !model) return clip;
  
  const boneMap = new Map();
  model.traverse((node) => {
    if (node.name) {
      const original = node.name;
      boneMap.set(original.toLowerCase(), original);
      const clean = original.replace(/^mixamorig\d*[:_]?/i, '').toLowerCase();
      boneMap.set(clean, original);
    }
  });

  const remappedTracks = [];
  clip.tracks.forEach((track) => {
    const dotIndex = track.name.lastIndexOf('.');
    if (dotIndex === -1) return;
    const targetNodeName = track.name.substring(0, dotIndex);
    const propertyName = track.name.substring(dotIndex);

    if (!propertyName.includes('quaternion')) return;

    const simpleTarget = targetNodeName.split('/').pop().replace(/:/g, '_');
    const cleanTarget = simpleTarget.replace(/^mixamorig\d*[:_]?/i, '').toLowerCase();

    const excludedBones = [
      'hips', 'armature', 'root', 
      'leftupleg', 'rightupleg', 'leftleg', 'rightleg', 
      'leftfoot', 'rightfoot', 'lefttoebase', 'righttoebase'
    ];
    if (excludedBones.includes(cleanTarget)) return;

    const matchedBone = boneMap.get(cleanTarget) || 
                         boneMap.get(simpleTarget.toLowerCase()) || 
                         boneMap.get(targetNodeName.toLowerCase());

    if (matchedBone) {
      const clonedTrack = track.clone();
      clonedTrack.name = matchedBone.replace(/:/g, '_') + propertyName;
      remappedTracks.push(clonedTrack);
    }
  });

  if (remappedTracks.length > 0) {
    clip.tracks = remappedTracks;
  }
  return clip;
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

// Camera Presets - Strictly Portrait Only
function setCameraPortrait() {
  if (camera && controls) {
    camera.position.set(0, 1.54, 1.02);
    controls.target.set(0, 1.48, 0);
    controls.minDistance = 0.6;
    controls.maxDistance = 1.35;
    controls.update();
  }
  if (btnCameraHead) btnCameraHead.classList.add('active');
  if (btnCameraBody) btnCameraBody.classList.remove('active');
}

function setCameraFullBody() {
  // Enforce portrait only as requested
  setCameraPortrait();
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
        if (animations.idle && !isAudioPlaying && currentAction !== animations.waving) {
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
  initAudioContext();
  if (!wsAudioCtx) return;

  const audioData = base64ToArrayBuffer(payload.audio_base64);
  try {
    const audioBuffer = await wsAudioCtx.decodeAudioData(audioData.slice(0));
    audioQueue.push({
      buffer: audioBuffer,
      text: payload.text,
      duration: payload.duration
    });
    if (!isAudioPlaying) {
      playNextAudioInQueue();
    }
  } catch (err) {
    console.warn("[WebAudio] Failed to decode MP3 chunk:", err);
  }
}

function playNextAudioInQueue() {
  if (audioQueue.length === 0) {
    isAudioPlaying = false;
    isSpeaking = false;
    const statusIndicator = document.querySelector('.status-indicator');
    if (statusIndicator) {
      statusIndicator.classList.remove('speaking');
      statusIndicator.classList.add('online');
    }
    agentStatus.textContent = "Agent Ready • Listening";
    if (animations.idle && currentAction !== animations.namaste) {
      crossFadeToAction(animations.idle, 0.6);
    }
    return;
  }

  isAudioPlaying = true;
  isSpeaking = true;
  speechElapsed = 0;

  const item = audioQueue.shift();
  const source = wsAudioCtx.createBufferSource();
  source.buffer = item.buffer;
  source.connect(wsAudioAnalyser);
  wsAudioAnalyser.connect(wsAudioCtx.destination);
  activeAudioSource = source;

  const statusIndicator = document.querySelector('.status-indicator');
  if (statusIndicator) {
    statusIndicator.classList.remove('online');
    statusIndicator.classList.add('speaking');
  }
  agentStatus.textContent = "Veda Speaking...";

  // Only perform gentle wave for greeting/namaste; keep avatar strictly in natural idle posture otherwise
  const textLower = (item.text || '').toLowerCase();
  if (textLower.includes('namaste') || textLower.includes('pranam') || textLower.includes('welcome') || textLower.includes('hello') || textLower.includes('swagat') || textLower.startsWith('hi')) {
    if (animations.waving) {
      crossFadeToAction(animations.waving, 0.25);
    }
    targetSmile = 0.22;
  } else {
    if (animations.idle && currentAction !== animations.idle && currentAction !== animations.waving) {
      crossFadeToAction(animations.idle, 0.5);
    }
  }

  source.onended = () => {
    activeAudioSource = null;
    playNextAudioInQueue();
  };

  source.start(0);
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

  let startAction = null;
  if (gestureKey === 'waving' || gestureKey === 'namaste') {
    startAction = animations.waving;
  } else if (animations.idle) {
    startAction = animations.idle;
  }

  if (startAction) crossFadeToAction(startAction, 0.25);
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

// Facial Expressions, Blinking, and Real-time Lip-Sync Loop
function updateFacialExpressions(delta) {
  if (facialMeshes.length === 0) return;

  // 1. Natural Organic Eye Blinking
  blinkTimer -= delta;
  if (blinkTimer <= 0 && !isBlinking) {
    isBlinking = true;
    blinkProgress = 0;
    blinkTimer = 3.2 + Math.random() * 2.8;
  }
  if (isBlinking) {
    blinkProgress += delta / 0.16;
    if (blinkProgress >= 1.0) {
      isBlinking = false;
      setMorphInfluence('eyeBlinkLeft', 0);
      setMorphInfluence('eyeBlinkRight', 0);
      setMorphInfluence('eyesClosed', 0);
      setMorphInfluence('blink', 0);
    } else {
      const blinkWeight = Math.sin(blinkProgress * Math.PI);
      setMorphInfluence('eyeBlinkLeft', blinkWeight);
      setMorphInfluence('eyeBlinkRight', blinkWeight);
      setMorphInfluence('eyesClosed', blinkWeight);
      setMorphInfluence('blink', blinkWeight);
    }
  }

  // 2. High-Fidelity Speech Mouth Cadence driven by Audio Analyser or Procedural Syllables
  if (isSpeaking || isAudioPlaying) {
    speechElapsed += delta;
    let mouthOpening = 0;

    if (wsAudioAnalyser && wsAudioDataArray && isAudioPlaying) {
      wsAudioAnalyser.getByteFrequencyData(wsAudioDataArray);
      let sum = 0;
      const maxBin = Math.min(64, wsAudioDataArray.length);
      for (let i = 4; i < maxBin; i++) {
        sum += wsAudioDataArray[i];
      }
      const avg = sum / (maxBin - 4);
      if (avg > 12) {
        mouthOpening = Math.min(0.55, (avg / 100.0) * 0.55);
      } else {
        // Procedural syllable cadence fallback during quieter syllables
        const syllableFreq = 3.6;
        const phase = (speechElapsed * syllableFreq * Math.PI * 2) % (Math.PI * 2);
        let wave = Math.sin(phase);
        mouthOpening = wave > 0 ? Math.pow(wave, 1.35) * 0.32 : 0;
      }
    } else {
      // Procedural syllable cadence for Web Speech API
      const syllableFreq = 3.6;
      const phase = (speechElapsed * syllableFreq * Math.PI * 2) % (Math.PI * 2);
      let wave = Math.sin(phase);
      mouthOpening = wave > 0 ? Math.pow(wave, 1.35) * 0.42 : 0;
    }

    targetJawOpen = mouthOpening;
    targetVisemeAA = mouthOpening * 0.70;
    targetVisemeO = mouthOpening * 0.35;
    targetSmile = 0.18;
  } else {
    targetJawOpen = 0;
    targetVisemeAA = 0;
    targetVisemeO = 0;
    targetSmile = 0.08;
  }

  // Smooth 60FPS dampening / interpolation
  const lerpSpeed = (isSpeaking || isAudioPlaying) ? 18.0 : 12.0;
  const t = Math.min(1.0, delta * lerpSpeed);
  
  currentJawOpen = THREE.MathUtils.lerp(currentJawOpen, targetJawOpen, t);
  currentVisemeAA = THREE.MathUtils.lerp(currentVisemeAA, targetVisemeAA, t);
  currentVisemeO = THREE.MathUtils.lerp(currentVisemeO, targetVisemeO, t);
  currentSmile = THREE.MathUtils.lerp(currentSmile, targetSmile, t);

  // Apply to all facial meshes synchronously
  setMorphInfluence('jawOpen', currentJawOpen);
  setMorphInfluence('mouthOpen', currentJawOpen * 0.85);
  setMorphInfluence('viseme_aa', currentVisemeAA);
  setMorphInfluence('viseme_O', currentVisemeO);
  setMorphInfluence('mouthSmile', currentSmile);
  setMorphInfluence('mouthSmileLeft', currentSmile);
  setMorphInfluence('mouthSmileRight', currentSmile);
}

// Render Loop
function animate() {
  requestAnimationFrame(animate);
  const delta = clock.getDelta();
  if (mixer) mixer.update(delta);
  if (controls) controls.update();

  updateFacialExpressions(delta);

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

if (btnWave) {
  btnWave.addEventListener('click', () => {
    if (animations.waving) crossFadeToAction(animations.waving, 0.25);
    handleUserPrompt("Namaste! Welcome to the Human Technology Interaction Lab at Plaksha University!", 'waving');
  });
}

if (btnGestureExplain) {
  btnGestureExplain.addEventListener('click', () => {
    handleUserPrompt("Can you explain how affective computing, bio-sensing, and 3D avatars work at HTI Lab?");
  });
}

if (btnGestureWelcome) {
  btnGestureWelcome.addEventListener('click', () => {
    if (animations.waving) crossFadeToAction(animations.waving, 0.25);
    handleUserPrompt("A warm welcome to everyone visiting our HTI Lab today!", 'waving');
  });
}

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
    let gesture = null;
    if (text.includes('Namaste') || text.includes('Welcome') || text.includes('Hello')) {
      gesture = 'waving';
    }
    handleUserPrompt(text, gesture);
  });
});

window.addEventListener('DOMContentLoaded', () => {
  initEngine();
});
