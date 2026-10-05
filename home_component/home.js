// EnergyPulse home console.
// A Streamlit component: Python sends one day of meter + detection data,
// this file plays it on a 3D model of the home and sends the user's
// test-mode switches back.
import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const AMBER = 0xf5a83c;
const FAN_ICON = "<circle cx='12' cy='12' r='1.6'/><path d='M12 10.4c0-4 1-6.4 3-6.4 2.4 0 2.6 4 .2 6.2M13.4 12.8c3.4 2 5 4 4 5.8-1.2 2-4.8.2-5.6-2.9M10.6 12.8c-3.4 2-6 2.3-7 .6-1.2-2.1 2.2-4.3 5.3-3.4'/>";

// ---------------------------------------------------------------- Streamlit
const post = (type, extra = {}) =>
  window.parent.postMessage({ isStreamlitMessage: true, type, ...extra }, "*");
const sendValue = (value) => post("streamlit:setComponentValue", { value, dataType: "json" });
const setHeight = () => post("streamlit:setFrameHeight", { height: document.documentElement.scrollHeight });

const GUTTER = 118, TOP = 16, LANE = 15, GAP = 4;   // timeline geometry

// ---------------------------------------------------------------- state
const S = {
  data: null, version: null, playhead: 0, playing: true, speed: 1,
  selected: null, mode: "live", switches: {}, pending: false,
  prefix: {}, lastPanel: 0, lastMinute: -1, pulseAt: 0,
  fans: {}, fanKwh: 0, fanCost: 0, devKwh: {}, devCost: {}, member: null, lastActivity: null, hover: null, colors: {},
};
const SPEEDS = [["1 min/s", 1], ["5 min/s", 5], ["20 min/s", 20]];
const INTRO_SEC = 1.9;      // one opening move: the model rises and the camera settles
const TEST_RUN_MIN = { geyser: 25, water_pump: 20, washing_machine: 60, microwave: 4, ac: 180 };

// ---------------------------------------------------------------- icons
const ICONS = {
  fan_living: FAN_ICON, fan_bedroom: FAN_ICON,
  meter: '<path d="M5 3h14v18H5z"/><path d="M8 7h8v4H8z"/><path d="M9 15h.01M12 15h.01M15 15h.01"/>',
  fridge: '<path d="M7 3h10v18H7z"/><path d="M7 10h10M10 6v2M10 13v3"/>',
  ac: '<path d="M3 6h18v7H3z"/><path d="M6 10h12M8 16c0 1.5-1 1.5-1 3M12 16c0 1.5-1 1.5-1 3M16 16c0 1.5-1 1.5-1 3"/>',
  geyser: '<path d="M8 3h8v13a4 4 0 0 1-8 0z"/><path d="M10 21v-1M14 21v-1M11 8h2"/>',
  washing_machine: '<path d="M5 3h14v18H5z"/><circle cx="12" cy="13.5" r="4"/><path d="M8 6h.01M11 6h.01"/>',
  water_pump: '<circle cx="10" cy="13" r="5"/><path d="M15 13h5v-4M10 8V4h4M10 13h.01"/>',
  microwave: '<path d="M3 6h18v12H3z"/><path d="M6 9h9v6H6zM18 9v.01M18 12v.01M18 15v.01"/>',
};
const KIND_ICONS = {
  fan: FAN_ICON,
  light: '<path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-4 10.5c.7.7 1 1.3 1 2.5h6c0-1.2.3-1.8 1-2.5A6 6 0 0 0 12 3z"/>',
  tv: '<path d="M3 5h18v12H3zM8 21h8M12 17v4"/>',
  generic: '<path d="M9 3v5M15 3v5M6 8h12v4a6 6 0 0 1-12 0zM12 18v3"/>',
};
const icon = (key) =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[key] || ""}</svg>`;

// ---------------------------------------------------------------- formatting
const hhmm = (m) => {
  m = Math.max(0, Math.floor(m)) % 1440;
  return `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
};
const dur = (min) => {
  min = Math.round(min);
  return min < 60 ? `${min} min` : `${Math.floor(min / 60)} h ${String(min % 60).padStart(2, "0")} min`;
};
const rs = (v) => `Rs. ${v < 100 ? v.toFixed(1) : Math.round(v).toLocaleString("en-IN")}`;

// ================================================================= 3D SCENE
const AMBER_COLOR = new THREE.Color(AMBER);
const W = 12, D = 9, OX = -W / 2, OZ = -D / 2;
const canvas = $("scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
const LITE = (navigator.hardwareConcurrency || 8) <= 2 || (navigator.deviceMemory || 8) <= 2;
renderer.setPixelRatio(LITE ? 1 : Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = !LITE;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;

const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-10, 10, 6, -6, 0.1, 100);
camera.position.set(13, 13.5, 15);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0.2, 0.2, 0.3);
controls.enablePan = false;
controls.enableDamping = true;
controls.minPolarAngle = 0.55; controls.maxPolarAngle = 1.12;
// No azimuth limits: the model turns a full 360 degrees.
controls.minZoom = 0.8; controls.maxZoom = 2.4;

const hemi = new THREE.HemisphereLight(0xc4d2ff, 0x161b29, 1.05);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xfff0dc, 1.9);
sun.position.set(-7, 14, 3);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -11, right: 11, top: 11, bottom: -11, near: 1, far: 40 });
sun.shadow.bias = -0.0004; sun.shadow.radius = 4;
scene.add(sun);

// Two looks for the same model. Every themed material is registered so a
// switch of theme recolours the scene without rebuilding it.
const PALETTE = {
  dark: { slab: 0x1b2136, fLiving: 0x2b3350, fKitchen: 0x283048, fBedroom: 0x302f52, fBath: 0x253449,
          fUtility: 0x232b42, furn: 0x46507a, soft: 0x5d6890, pale: 0x8f9abd, rug: 0x4a4178,
          appliance: 0xc3cadb, trace: 0x4a5475, ring: 0xe6e9f0, leaf: 0x4f9a76,
          glass: 0x8a98ff, glassOpacity: 0.2, edge: 0xaab4ff, edgeOpacity: 0.85,
          sky: 0xc4d2ff, groundLight: 0x161b29, hemi: 1.0, sun: 1.7, shadow: 0.4, lamp: 2.8, exposure: 1.15 },
  light: { slab: 0xdfe3ee, fLiving: 0xfbfbfd, fKitchen: 0xf3f5f9, fBedroom: 0xf8f6fb, fBath: 0xeef4f8,
           fUtility: 0xf1f3f7, furn: 0xe2e6ee, soft: 0xeceff5, pale: 0xffffff, rug: 0xe4ddf3,
           appliance: 0xf4f5f8, trace: 0xc3c9d8, ring: 0x171b28, leaf: 0x58b081,
           glass: 0x7f9df0, glassOpacity: 0.3, edge: 0x7f9df0, edgeOpacity: 0.75,
           sky: 0xffffff, groundLight: 0xc7cde0, hemi: 1.15, sun: 2.1, shadow: 0.14, lamp: 0.35, exposure: 1.0 },
};
let P = PALETTE.dark;
const themed = [];
const std = (color, extra = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.85, metalness: 0, ...extra });
const tm = (key, extra = {}) => { const m = std(PALETTE.dark[key], extra); themed.push([m, key]); return m; };

function box(w, h, d, x, y, z, mat, parent = scene) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  m.position.set(x + w / 2 + OX, y + h / 2, z + d / 2 + OZ);   // x, z are the near corner
  m.castShadow = true; m.receiveShadow = true;
  parent.add(m);
  return m;
}
function cyl(r, h, x, y, z, mat, parent = scene, rot = null) {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, h, 28), mat);
  m.position.set(x + OX, y, z + OZ);
  if (rot) m.rotation.set(...rot);
  m.castShadow = true; m.receiveShadow = true;
  parent.add(m);
  return m;
}

// ground + slab
const shadowMat = new THREE.ShadowMaterial({ opacity: 0.38 });
const ground = new THREE.Mesh(new THREE.PlaneGeometry(60, 60), shadowMat);
ground.rotation.x = -Math.PI / 2; ground.position.y = -0.3; ground.receiveShadow = true;
scene.add(ground);
box(W + 0.5, 0.3, D + 0.5, -0.25, -0.3, -0.25, tm("slab"));

// floors
const ROOMS = [
  ["Living room", 0, 0, 7, 5, "fLiving"], ["", 7, 4, 1.5, 1, "fLiving"],
  ["Kitchen", 7, 0, 5, 4, "fKitchen"], ["Bedroom", 0, 5, 5.5, 4, "fBedroom"],
  ["Bathroom", 5.5, 5, 3, 4, "fBath"], ["Utility", 8.5, 4, 3.5, 5, "fUtility"],
];
for (const [, x, z, w, d, c] of ROOMS) box(w, 0.04, d, x, 0, z, tm(c, { roughness: 0.95 })).castShadow = false;
box(2.9, 0.02, 1.9, 0.75, 0.04, 2.0, tm("rug", { roughness: 1 })).castShadow = false;      // living rug
box(2.9, 0.02, 3.0, 1.05, 0.04, 5.95, tm("rug", { roughness: 1 })).castShadow = false;     // bedroom rug

// Walls are tinted glass, so every room reads from any angle.
const T = 0.12, FULL = 1.5, MID = 1.5, LOW = 0.62;
const glassWall = new THREE.MeshPhysicalMaterial({
  color: 0x8a98ff, transparent: true, opacity: 0.2, roughness: 0.12, metalness: 0,
  side: THREE.DoubleSide, depthWrite: false });
const edgeMat = new THREE.LineBasicMaterial({ color: 0xaab4ff, transparent: true, opacity: 0.85 });
function wall(x1, z1, x2, z2, h) {
  const horiz = z1 === z2, len = Math.abs(horiz ? x2 - x1 : z2 - z1) + T;
  const geo = new THREE.BoxGeometry(horiz ? len : T, h, horiz ? T : len);
  geo.translate(0, h / 2, 0);                          // scale from the floor up
  const mesh = new THREE.Mesh(geo, glassWall);
  mesh.position.set((x1 + x2) / 2 + OX, 0.04, (z1 + z2) / 2 + OZ);
  mesh.renderOrder = 2;
  mesh.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), edgeMat));
  scene.add(mesh);
  return mesh;
}
const INNER = [
  [7, 0, 7, 2.4], [7, 3.4, 7, 4], [7, 4, 8.5, 4], [8.5, 4, 10, 4], [11, 4, 12, 4],
  [0, 5, 3.5, 5], [4.5, 5, 5.5, 5], [5.5, 5, 6.5, 5], [7.4, 5, 8.5, 5], [5.5, 5, 5.5, 9], [8.5, 4, 8.5, 9],
];
for (const [x1, z1, x2, z2] of INNER) wall(x1, z1, x2, z2, MID);

// Outer walls on the camera's side lower a little so pins in front stay clear.
const OUTER = [
  [0, 0, 12, 0, 0, -1], [0, 0, 0, 1, -1, 0], [0, 2, 0, 9, -1, 0], [12, 0, 12, 9, 1, 0], [0, 9, 12, 9, 0, 1],
];
const outerWalls = OUTER.map(([x1, z1, x2, z2, nx, nz]) => ({ mesh: wall(x1, z1, x2, z2, FULL), nx, nz, s: 1, panes: [] }));
const wallMat = tm("pale");      // solid white pieces (pillows)

// furniture: quiet context only
const furn = tm("furn"), soft = tm("soft"), pale = tm("pale");
const screenMat = std(0x10131c);
box(2.6, 0.42, 0.9, 0.9, 0.04, 3.9, furn); box(2.6, 0.75, 0.22, 0.9, 0.04, 4.6, furn);   // sofa
box(0.7, 0.12, 0.7, 1.0, 0.46, 3.95, pale); box(0.7, 0.12, 0.7, 2.7, 0.46, 3.95, pale);  // cushions
box(1.2, 0.22, 0.7, 1.6, 0.06, 2.5, soft);                                               // coffee table
box(2.4, 0.38, 0.4, 1.0, 0.04, 0.12, furn); box(1.5, 0.75, 0.06, 1.45, 0.5, 0.14, screenMat); // tv
box(1.2, 0.45, 1.9, 4.6, 0.04, 0.8, soft);                                               // dining
for (const [x, z] of [[4.25, 1.0], [4.25, 2.1], [5.85, 1.0], [5.85, 2.1]]) box(0.32, 0.3, 0.32, x, 0.04, z, furn); // stools
box(2.2, 0.36, 2.5, 1.4, 0.06, 6.2, furn); box(2.0, 0.14, 2.3, 1.5, 0.42, 6.3, pale);    // bed
box(0.8, 0.12, 0.45, 1.6, 0.56, 6.4, wallMat); box(0.8, 0.12, 0.45, 2.55, 0.56, 6.4, wallMat);
box(0.45, 0.38, 0.45, 3.75, 0.04, 6.25, furn);                                           // bedside table
box(0.6, 1.05, 1.7, 4.8, 0.04, 6.9, furn);                                               // wardrobe
box(3.6, 0.55, 0.7, 8.3, 0.04, 0.1, furn); box(0.7, 0.55, 2.4, 11.2, 0.04, 0.8, furn);   // kitchen counters
box(0.62, 0.03, 0.5, 8.55, 0.59, 0.2, screenMat);                                        // hob
box(1.3, 0.34, 1.5, 5.7, 0.04, 7.4, pale); box(0.45, 0.4, 0.6, 7.9, 0.04, 8.2, pale);    // tub, wc
const leafMat = tm("leaf", { flatShading: true });
for (const [x, z] of [[6.4, 0.5], [0.5, 4.6], [11.4, 5.0]]) {
  cyl(0.16, 0.3, x, 0.19, z, furn);
  const leaf = new THREE.Mesh(new THREE.IcosahedronGeometry(0.34, 1), leafMat);
  leaf.position.set(x + OX, 0.72, z + OZ); leaf.castShadow = true; scene.add(leaf);
}

// lamps: the warm light that makes the dark model feel lived in
const shadeMat = new THREE.MeshStandardMaterial({ color: 0xffe2b8, emissive: 0xffc27a, emissiveIntensity: 0.9, roughness: 0.6 });
const lamps = [];
function lamp(x, z, base, reach) {
  cyl(0.025, base, x, 0.04 + base / 2, z, furn);
  cyl(0.15, 0.2, x, 0.14 + base, z, shadeMat).castShadow = false;
  const l = new THREE.PointLight(0xffc98a, 2.6, reach, 1.7);
  l.position.set(x + OX, base + 0.1, z + OZ); scene.add(l); lamps.push(l);
}
lamp(6.25, 4.4, 1.0, 5); lamp(3.97, 6.47, 0.42, 3.5);

// switched devices (fans, lights, TV, anything else the household listed):
// built from the list Python sends, drawn glowing or spinning when on
const fanUnits = {};
function buildDevice(f) {
  const [x, z] = f.pos, g = new THREE.Group();
  const body = tm("appliance", { roughness: 0.5 });
  const u = { group: g, kind: f.kind, speed: 0, glow: 0, lightMax: 0 };
  const glowMat = (color, emissive) => new THREE.MeshStandardMaterial({ color, emissive, emissiveIntensity: 0, roughness: 0.45 });
  const addLight = (color, lx, ly, lz, reach, max) => {
    u.light = new THREE.PointLight(color, 0, reach, 1.6); u.lightMax = max;
    u.light.position.set(lx + OX, ly, lz + OZ); scene.add(u.light);
  };
  if (f.kind === "fan") {
    cyl(0.018, 0.4, x, 1.72, z, body, g);
    cyl(0.1, 0.08, x, 1.5, z, body, g);
    const rotor = new THREE.Group();
    rotor.position.set(x + OX, 1.48, z + OZ);
    for (let i = 0; i < 3; i++) {
      const arm = new THREE.Group(); arm.rotation.y = (i * Math.PI * 2) / 3;
      const blade = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.014, 0.13), body);
      blade.position.x = 0.4; blade.castShadow = true; arm.add(blade); rotor.add(arm);
    }
    g.add(rotor); u.rotor = rotor;
    u.anchor = new THREE.Vector3(x + OX, 2.0, z + OZ);
  } else if (f.kind === "light") {
    cyl(0.012, 0.3, x, 1.75, z, body, g);
    u.mat = glowMat(0xd9dce6, 0xffd9a0);
    cyl(0.2, 0.07, x, 1.57, z, u.mat, g).castShadow = false;
    addLight(0xffd9a0, x, 1.38, z, 5.5, 7);
    u.anchor = new THREE.Vector3(x + OX, 1.98, z + OZ);
  } else if (f.kind === "tv") {
    u.mat = glowMat(0x10131c, 0x7fb4ff);
    box(1.42, 0.67, 0.02, x - 0.71, 0.54, z - 0.005, u.mat, g).castShadow = false;
    addLight(0x7fb4ff, x, 0.9, z + 0.7, 3.5, 4);
    u.anchor = new THREE.Vector3(x + OX, 1.5, z + 0.1 + OZ);
  } else {
    box(0.46, 0.5, 0.46, x - 0.23, 0.04, z - 0.23, tm("furn"), g);
    u.mat = glowMat(0xd9dce6, AMBER);
    box(0.34, 0.26, 0.3, x - 0.17, 0.54, z - 0.15, u.mat, g);
    addLight(AMBER, x, 1.0, z, 3, 3.5);
    u.anchor = new THREE.Vector3(x + OX, 1.2, z + OZ);
  }
  scene.add(g);
  g.traverse((o) => { o.userData.key = f.key; });
  fanUnits[f.key] = u;
}

function applyTheme(name) {
  P = PALETTE[name] || PALETTE.dark;
  for (const [mat, key] of themed) mat.color.setHex(P[key]);
  hemi.color.setHex(P.sky); hemi.groundColor.setHex(P.groundLight); hemi.intensity = P.hemi;
  sun.intensity = P.sun; shadowMat.opacity = P.shadow;
  glassWall.color.setHex(P.glass); glassWall.opacity = P.glassOpacity;
  edgeMat.color.setHex(P.edge); edgeMat.opacity = P.edgeOpacity;
  lamps.forEach((l) => { l.intensity = P.lamp; });
  shadeMat.emissiveIntensity = name === "light" ? 0.25 : 0.9;
  renderer.toneMappingExposure = P.exposure;
}

// ---------------------------------------------------------------- appliances
// Each entry: where it sits, how its pin is anchored, and the floor route
// the supply takes from the meter.
const HUB = [[0.3, 2.6], [6.6, 2.6]];
const KIT = [...HUB, [6.6, 2.9], [7.6, 2.9]];
const UTIL = [...KIT, [10.5, 2.9], [10.5, 4.5]];
const LAYOUT = {
  fridge: { anchor: [7.62, 1.5, 0.52], path: [...KIT, [7.62, 0.95]] },
  microwave: { anchor: [10.2, 1.0, 0.45], path: [...KIT, [10.2, 2.9], [10.2, 0.85]] },
  washing_machine: { anchor: [9.3, 0.95, 4.75], path: [...UTIL, [9.75, 4.75]] },
  water_pump: { anchor: [11.2, 0.6, 8.2], path: [...UTIL, [10.5, 8.2], [10.9, 8.2]] },
  ac: { anchor: [0.3, 1.45, 5.9], path: [[0.3, 2.6], [4.0, 2.6], [4.0, 5.65], [0.5, 5.65]] },
  geyser: { anchor: [5.9, 1.2, 6.0], path: [...HUB, [6.6, 4.5], [6.95, 4.5], [6.95, 5.6], [5.95, 5.6]] },
};
const units = {};   // key -> { group, mat, light, traceMat, dots, anchor, pathLen, pts }
const ringMat = new THREE.MeshBasicMaterial({ color: 0xe6e9f0, transparent: true, opacity: 0.9 });

function buildAppliance(key) {
  const g = new THREE.Group();
  const mat = tm("appliance", { roughness: 0.55, emissive: 0x000000, emissiveIntensity: 0 });
  const dark = std(0x1a1f2d, { roughness: 0.4 });
  if (key === "fridge") {
    box(0.8, 1.4, 0.75, 7.22, 0.04, 0.14, mat, g);
    box(0.8, 0.02, 0.02, 7.22, 0.95, 0.89, dark, g); box(0.04, 0.3, 0.04, 7.9, 1.05, 0.9, dark, g);
  } else if (key === "microwave") {
    box(0.62, 0.34, 0.42, 9.9, 0.59, 0.24, mat, g); box(0.4, 0.22, 0.02, 9.95, 0.65, 0.66, dark, g);
  } else if (key === "washing_machine") {
    box(0.72, 0.85, 0.72, 8.95, 0.04, 4.2, mat, g);
    const door = new THREE.Mesh(new THREE.TorusGeometry(0.2, 0.045, 12, 32), dark);
    door.position.set(9.31 + OX, 0.45, 4.93 + OZ); g.add(door);
  } else if (key === "water_pump") {
    box(0.8, 0.12, 0.5, 10.8, 0.04, 7.95, dark, g);
    cyl(0.2, 0.55, 11.15, 0.36, 8.2, mat, g, [0, 0, Math.PI / 2]);
    cyl(0.13, 0.3, 10.85, 0.42, 8.2, mat, g);
  } else if (key === "ac") {
    box(0.26, 0.32, 1.2, 0.07, 1.0, 5.3, mat, g); box(0.02, 0.05, 1.05, 0.33, 1.03, 5.38, dark, g);
  } else if (key === "geyser") {
    cyl(0.24, 0.62, 5.83, 0.82, 6.0, mat, g);
    cyl(0.03, 0.5, 5.83, 0.28, 6.0, dark, g);
  }
  scene.add(g);

  const L = LAYOUT[key];
  const light = new THREE.PointLight(0xffffff, 0, 5.5, 1.6);
  light.position.set(L.anchor[0] + OX + 0.35, Math.max(0.6, L.anchor[1] - 0.5), L.anchor[2] + OZ + 0.35);
  scene.add(light);

  // supply trace on the floor
  const traceMat = new THREE.MeshBasicMaterial({ color: 0x4a5475 });
  const pts = L.path.map(([x, z]) => new THREE.Vector3(x + OX, 0.085, z + OZ));
  let pathLen = 0;
  const segs = [];
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i], b = pts[i + 1], len = a.distanceTo(b);
    pathLen += len;
    const seg = new THREE.Mesh(new THREE.BoxGeometry(Math.abs(b.x - a.x) + 0.05, 0.012, Math.abs(b.z - a.z) + 0.05), traceMat);
    seg.position.copy(a).lerp(b, 0.5);
    scene.add(seg); segs.push(seg);
  }
  const dots = [];
  const dotMat = new THREE.MeshBasicMaterial({ color: AMBER });
  for (let i = 0; i < 7; i++) {
    const d = new THREE.Mesh(new THREE.SphereGeometry(0.055, 10, 10), dotMat);
    d.visible = false; scene.add(d); dots.push(d);
  }
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.62, 0.68, 48), ringMat);
  ring.rotation.x = -Math.PI / 2;
  ring.position.set(L.anchor[0] + OX, 0.09, L.anchor[2] + OZ); ring.visible = false; scene.add(ring);

  g.traverse((o) => { o.userData.key = key; });
  units[key] = { group: g, mat, light, traceMat, dots, pts, pathLen, ring, segs, glow: 0, phase: Math.random(),
                 anchor: new THREE.Vector3(L.anchor[0] + OX, L.anchor[1], L.anchor[2] + OZ) };
}
Object.keys(LAYOUT).forEach(buildAppliance);

// main meter on the wall by the entrance
const meterBody = box(0.1, 0.56, 0.42, 0.08, 0.72, 2.4, tm("appliance", { roughness: 0.5 }));
meterBody.userData.key = "meter";
const meterGlow = box(0.02, 0.16, 0.26, 0.18, 1.02, 2.48, new THREE.MeshBasicMaterial({ color: AMBER }));
const meterLight = new THREE.PointLight(AMBER, 3, 3.5, 1.8);
meterLight.position.set(0.6 + OX, 1.0, 2.6 + OZ); scene.add(meterLight);
const meterAnchor = new THREE.Vector3(0.2 + OX, 1.45, 2.6 + OZ);

function pointOnPath(u, t) {       // t in 0..1 along the route
  let dist = t * u.pathLen;
  for (let i = 0; i < u.pts.length - 1; i++) {
    const len = u.pts[i].distanceTo(u.pts[i + 1]);
    if (dist <= len) return u.pts[i].clone().lerp(u.pts[i + 1], dist / len);
    dist -= len;
  }
  return u.pts[u.pts.length - 1].clone();
}

// ---------------------------------------------------------------- pins
const pinsEl = $("pins");
const pins = {};
function makePin(key, label, extra = "") {
  const el = document.createElement("button");
  el.type = "button"; el.className = `pin ${extra}`;
  el.innerHTML = `<span class="dot">${icon(key)}</span><span class="v"></span>`;
  el.setAttribute("aria-label", label);
  el.addEventListener("click", () => {
    if (key === "meter") return;
    select(key);
    if (isFan(key)) toggleFan(key);
  });
  el.addEventListener("pointerenter", (e) => showTip(key, e.clientX, e.clientY));
  el.addEventListener("pointerleave", () => showTip(null));
  pinsEl.appendChild(el);
  pins[key] = el;
}
const roomLabels = ROOMS.filter((r) => r[0]).map(([name, x, z, w, d]) => {
  const el = document.createElement("span");
  el.textContent = name;
  el.className = "room";
  pinsEl.appendChild(el);
  return { el, pos: new THREE.Vector3(x + w / 2 + OX, 0.06, z + d * 0.72 + OZ) };
});
const v3 = new THREE.Vector3();
function place(el, pos, above) {
  v3.copy(pos).project(camera);
  const x = (v3.x * 0.5 + 0.5) * canvas.clientWidth, y = (-v3.y * 0.5 + 0.5) * canvas.clientHeight;
  el.style.transform = above ? `translate(${x}px, ${y}px) translate(-17px, calc(-100% - 8px))`
                             : `translate(${x}px, ${y}px) translate(-50%, -50%)`;
}

// ---------------------------------------------------------------- hover + click
const ray = new THREE.Raycaster(); let down = null;
const tip = $("tip");
function pick(e) {
  const r = canvas.getBoundingClientRect();
  ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
  const targets = [...Object.values(units).filter((u) => u.group.visible).map((u) => u.group),
                   ...Object.values(fanUnits).filter((f) => f.group.visible).map((f) => f.group), meterBody];
  const hit = ray.intersectObjects(targets, true)[0];
  return hit ? hit.object.userData.key : null;
}
const fanKw = () => (S.data ? S.data.fans.reduce((sum, f) => sum + (S.fans[f.key] ? f.kw : 0), 0) : 0);
const isFan = (key) => !!fanUnits[key];
const devByKey = (key) => (S.data ? S.data.fans.find((f) => f.key === key) : null);

// What the pointer is over: is it on, what it is, and its figures.
function describe(key) {
  const d = S.data, m = Math.max(0, Math.min(d.live_edge - 1, Math.floor(S.playhead)));
  if (key === "meter") {
    return `<b>Main meter</b><span class="what">Single-phase household meter</span>
      <span class="on">${(d.mains[m] + fanKw()).toFixed(2)} kW right now</span>
      <span>${(S.prefix.kwh[m + 1] + S.fanKwh).toFixed(2)} units today, ${rs(S.prefix.cost[m + 1] + S.fanCost)}</span>`;
  }
  if (isFan(key)) {
    const f = d.fans.find((x) => x.key === key), on = !!S.fans[key];
    return `<b>${f.name}</b><span class="what">${f.product}</span>
      <span class="${on ? "on" : "off"}">${on ? `On, ${(f.kw * 1000).toFixed(0)} W` : "Off"}</span>
      <span>${f.room}. Click to switch it ${on ? "off" : "on"}.</span>
      <span>This session: ${(S.devKwh[key] || 0).toFixed(3)} units, ${rs(S.devCost[key] || 0)}</span>`;
  }
  const a = byKey(key); if (!a) return "";
  const on = !!a.on[m], p = S.prefix[key];
  const status = on ? `On, ${a.kw[m].toFixed(2)} kW` : a.locked ? "Always on, compressor resting" : "Off";
  return `<b>${a.name}</b><span class="what">${a.product}</span>
    <span class="${on ? "on" : "off"}" style="--c:${a.color}">${status}</span>
    <span>Rated ${a.rated_kw} kW. ${a.category_label}. ${a.locked ? "Cannot be switched off." : "Click, then use its switch in the list."}</span>
    <span>Today: ${dur(p.on[m + 1])} running, ${p.kwh[m + 1].toFixed(2)} units, ${rs(p.cost[m + 1])}</span>`;
}
function showTip(key, clientX, clientY) {
  if (!key || !S.data) { tip.hidden = true; S.hover = null; return; }
  S.hover = key; tip.innerHTML = describe(key); tip.hidden = false;
  const box = $("viewport").getBoundingClientRect();
  const x = Math.min(clientX - box.left + 16, box.width - tip.offsetWidth - 8);
  const y = Math.min(clientY - box.top + 16, box.height - tip.offsetHeight - 8);
  tip.style.transform = `translate(${Math.max(8, x)}px, ${Math.max(8, y)}px)`;
}
canvas.addEventListener("pointermove", (e) => {
  if (e.buttons) { showTip(null); return; }
  const key = pick(e);
  canvas.style.cursor = key ? "pointer" : "";
  showTip(key, e.clientX, e.clientY);
});
canvas.addEventListener("pointerleave", () => showTip(null));
canvas.addEventListener("pointerdown", (e) => { down = [e.clientX, e.clientY]; showTip(null); });
canvas.addEventListener("pointerup", (e) => {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
  const key = pick(e);
  if (isFan(key)) { select(key); toggleFan(key); }
  else if (key && key !== "meter") select(key);
});

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  const aspect = w / h, vh = Math.max(12.2, 17.5 / aspect);
  camera.left = (-vh * aspect) / 2; camera.right = (vh * aspect) / 2;
  camera.top = vh / 2; camera.bottom = -vh / 2;
  camera.updateProjectionMatrix();
  drawTimeline(true); setHeight();
}
new ResizeObserver(resize).observe($("viewport"));

// ================================================================= DATA
function byKey(key) { return S.data.appliances.find((a) => a.key === key); }

const canon = (sw) => JSON.stringify(Object.keys(sw || {}).sort().map((k) => [k, sw[k]]));
function load(data) {
  // A reply to an older click can arrive after a newer click; wait for the newer one.
  if (S.pending && S.wantMode === data.mode && data.mode === "test"
      && canon(data.switches) !== canon(S.switches)) return;
  const first = !S.data;
  const modeChanged = S.data && S.data.mode !== data.mode;
  S.data = data; S.version = data.version; S.mode = data.mode; S.pending = false;
  S.switches = JSON.parse(JSON.stringify(data.switches || {}));
  $("console").dataset.mode = S.mode;

  // prefix sums so "today so far" is a lookup
  const n = data.mains.length, cum = (arr, f) => {
    const out = new Float64Array(n + 1);
    for (let i = 0; i < n; i++) out[i + 1] = out[i] + f(arr[i], i);
    return out;
  };
  S.prefix = { kwh: cum(data.mains, (v) => v / 60), cost: cum(data.mains, (v, i) => (v / 60) * data.rates[i]) };
  for (const a of data.appliances) {
    S.prefix[a.key] = {
      kwh: cum(a.kw, (v) => v / 60), cost: cum(a.kw, (v, i) => (v / 60) * data.rates[i]),
      on: cum(a.on, (v) => v),
    };
    const u = units[a.key];
    if (u) { u.color = new THREE.Color(a.color); u.light.color.copy(u.color); }
    if (!pins[a.key]) makePin(a.key, a.name);
    pins[a.key].style.setProperty("--c", a.color);
  }
  if (!pins.meter) makePin("meter", "Main meter", "is-meter");
  document.documentElement.dataset.theme = data.theme || "dark";
  applyTheme(data.theme || "dark");
  const css = getComputedStyle(document.documentElement);
  S.colors = Object.fromEntries(["plaster", "mist", "faint", "current", "lane"].map((k) => [k, css.getPropertyValue(`--${k}`).trim()]));
  ringMat.color.setHex(P.ring);
  for (const f of data.fans) {
    ICONS[f.key] = KIND_ICONS[f.kind] || KIND_ICONS.generic;
    if (!fanUnits[f.key]) buildDevice(f);
    if (!pins[f.key]) { makePin(f.key, f.name); pins[f.key].style.setProperty("--c", "var(--current)"); }
  }
  for (const [key, u] of Object.entries(fanUnits)) {
    const has = data.fans.some((f) => f.key === key);
    u.group.visible = has;
    if (!has && u.light) u.light.intensity = 0;
    if (pins[key]) pins[key].style.display = has ? "" : "none";
  }
  if (first) S.fans = { ...(data.fan_state || {}) };
  const sel = $("member");
  if (sel.options.length !== data.members.length) {
    sel.innerHTML = data.members.map((n) => `<option>${n}</option>`).join("");
    if (S.member && data.members.includes(S.member)) sel.value = S.member;
  }
  S.member = sel.value;
  const newest = data.activity[0] ? `${data.activity[0].time}|${data.activity[0].text}` : null;
  if (!first && newest && newest !== S.lastActivity) {
    toast(`${data.activity[0].text}. ${data.household_size > 1
      ? `All ${data.household_size} household members can see this.`
      : "Add family members on the Family page to share this with them."}`,
      / on$/.test(data.activity[0].text) ? "success" : "info");
  }
  S.lastActivity = newest;
  // The house only shows appliances this household has.
  for (const [key, u] of Object.entries(units)) {
    const has = data.appliances.some((a) => a.key === key);
    u.group.visible = has; u.segs.forEach((s) => { s.visible = has; });
    if (!has) { u.light.intensity = 0; u.ring.visible = false; u.dots.forEach((d) => { d.visible = false; }); }
    if (pins[key]) pins[key].style.display = has ? "" : "none";
  }
  if (!data.appliances.some((a) => a.key === S.selected) && !data.fans.some((f) => f.key === S.selected))
    S.selected = data.appliances[0].key;

  if (first || data.source.real) S.playhead = data.source.real ? data.live_edge - 1 : data.start_minute;
  if (first && data.appliances.some((a) => a.key === "ac")) S.selected = "ac";
  if (modeChanged && S.mode === "test") S.playhead = Math.min(S.playhead, data.live_edge - 2);

  $("source").textContent = data.source.label;
  $("source").classList.toggle("is-real", !!data.source.real);
  $("hint").textContent = S.mode === "test"
    ? "You are switching the appliances. The model sees only the meter total and has to find them."
    : "Glowing appliances are what the meter reading gives away. Every device has a switch in the list.";
  document.querySelectorAll(".seg [data-mode]").forEach((b) => b.classList.toggle("is-on", b.dataset.mode === S.mode));
  $("speeds").style.display = data.source.real ? "none" : "";
  $("play").style.display = data.source.real ? "none" : "";
  const tlH = TOP + 12 + data.appliances.length * (LANE + GAP);
  $("activity").innerHTML = data.activity.length
    ? data.activity.slice(0, 4).map((x) => `<li><time>${x.time}</time>${x.text}</li>`).join("")
    : "<li>Nothing switched yet. Switch any device to see it appear here for the whole household.</li>";
  $("timeline").style.height = `${tlH}px`;
  buildList(); S.lastMinute = -1; S.lastPanel = 0;
  if (first) {
    $("console").classList.remove("is-waiting");
    $("loading").classList.add("is-done");
    setTimeout(() => $("loading").remove(), 500);
  }
}

function select(key) {
  S.selected = key; S.lastPanel = 0;
  document.querySelectorAll("#list li").forEach((li) => li.classList.toggle("is-selected", li.dataset.key === key));
}

// ================================================================= PANEL
function buildList() {
  const ul = $("list"); ul.innerHTML = "";
  const group = (text) => { const li = document.createElement("li"); li.className = "group"; li.textContent = text; ul.appendChild(li); };
  const row = (key, name, desc, color, onSwitch, locked) => {
    const li = document.createElement("li");
    li.dataset.key = key; li.style.setProperty("--c", color);
    li.innerHTML = `<span class="badge">${icon(key)}</span>
      <div><div class="name">${name}</div><div class="desc">${desc}</div><div class="state"></div></div>`;
    li.addEventListener("click", () => select(key));
    const sw = document.createElement("button");
    sw.type = "button"; sw.className = "switch"; sw.setAttribute("role", "switch");
    sw.setAttribute("aria-label", `Switch ${name}${locked ? " (always on)" : ""}`);
    if (locked) { sw.disabled = true; sw.title = "Always on. It must not be switched off."; }
    sw.addEventListener("click", (e) => { e.stopPropagation(); select(key); onSwitch(); });
    li.appendChild(sw);
    ul.appendChild(li);
  };
  group("Found from the main meter");
  for (const a of S.data.appliances) row(a.key, a.name, a.product, a.color, () => toggle(a.key), a.locked);
  if (S.data.fans.length) group("Shown from their switch");
  for (const f of S.data.fans) row(f.key, f.name, `${f.product}, ${f.room.toLowerCase()}`, "var(--fill)", () => toggleFan(f.key), false);
  $("quick").innerHTML = "";
  select(S.selected);
}

let toastTimer = null;
function toast(text, kind = "info") {
  const el = $("toast");
  el.className = `toast ${kind}`; el.textContent = text; el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.classList.add("is-leaving");
    setTimeout(() => { el.hidden = true; el.classList.remove("is-leaving"); }, 220);
  }, 5000);
}
$("member").addEventListener("change", (e) => { S.member = e.target.value; });

function toggleFan(key) {
  S.fans[key] = !S.fans[key]; S.lastPanel = 0;
  sendValue({ mode: S.mode, switches: S.switches, fans: S.fans, nonce: Date.now(),
              event: { member: S.member, device: key, on: S.fans[key], minute: Math.floor(S.playhead) } });
}

function userOn(key, m) {             // test mode: what the user asked for
  const a = byKey(key);
  if (a.locked) return true;
  return (S.switches[key] || []).some(([s, e]) => m >= s && m < e);
}

function toggle(key) {
  const m = Math.floor(S.playhead);
  if (S.mode !== "test") {
    // Taking over from the replay: start from what is running right now.
    S.switches = {};
    for (const a of S.data.appliances) {
      if (!a.locked && a.on[m]) S.switches[a.key] = [[m, Math.min(S.data.mains.length, m + (TEST_RUN_MIN[a.key] || 30))]];
    }
    toast("You are now switching the appliances yourself (Test mode). The model has to find them from the meter total.", "info");
  }
  const list = (S.switches[key] = S.switches[key] || []);
  const live = list.find(([s, e]) => m >= s && m < e);
  if (live) { live[1] = m; if (live[1] <= live[0]) list.splice(list.indexOf(live), 1); }
  else list.push([m, Math.min(S.data.mains.length, m + (TEST_RUN_MIN[key] || 30))]);
  S.pending = true; S.wantMode = "test"; S.lastPanel = 0;
  sendValue({ mode: "test", switches: S.switches, fans: S.fans, nonce: Date.now(),
              event: { member: S.member, device: key, on: !live, minute: m } });
}

document.querySelectorAll(".seg [data-mode]").forEach((b) => b.addEventListener("click", () => {
  if (b.dataset.mode === S.mode) return;
  S.pending = true; S.wantMode = b.dataset.mode;
  sendValue({ mode: b.dataset.mode, switches: b.dataset.mode === "test" ? S.switches : {}, fans: S.fans, nonce: Date.now() });
}));

SPEEDS.forEach(([label, v], i) => {
  const b = document.createElement("button");
  b.type = "button"; b.textContent = label; if (i === 0) b.classList.add("is-on");
  b.addEventListener("click", () => {
    S.speed = v; [...$("speeds").children].forEach((c) => c.classList.toggle("is-on", c === b));
  });
  $("speeds").appendChild(b);
});
$("play").addEventListener("click", () => {
  S.playing = !S.playing;
  if (S.playing && S.playhead >= S.data.live_edge - 1) S.playhead = 0;
});

function updatePanel(m) {
  const d = S.data;
  $("clock").textContent = `${d.date_label}, ${hhmm(m)}`;
  const reg = (S.prefix.kwh[m + 1] + S.fanKwh).toFixed(2).padStart(6, "0");
  $("register").innerHTML = [...reg].map((c, i) =>
    c === "." ? "<i></i>" : `<span class="${i > reg.indexOf(".") ? "dec" : ""}">${c}</span>`).join("");
  $("cost").textContent = rs(S.prefix.cost[m + 1] + S.fanCost);
  const period = d.periods.find((p) => m >= p.start && m < p.end);
  $("rate").innerHTML = `<span class="dot" style="background:${period.color}"></span>
    <span><b>${period.label}</b> until ${hhmm(period.end)}, Rs. ${d.rates[m].toFixed(2)} per unit</span>`;
  $("play").classList.toggle("is-paused", !S.playing);
  $("play").setAttribute("aria-label", S.playing ? "Pause" : "Play");
  $("transportNote").textContent = d.source.real ? "Following the meter feed"
    : S.pending ? "Updating the meter signal..." : `Replay of a simulated day. 1 second = ${S.speed} min`;

  let running = 0, found = 0;
  for (const a of d.appliances) {
    const li = document.querySelector(`#list li[data-key="${a.key}"]`);
    const on = !!a.on[m], p = S.prefix[a.key];
    li.classList.toggle("is-on", on);
    const state = li.querySelector(".state");
    if (S.mode === "test") {
      const asked = userOn(a.key, m), truth = a.truth ? !!a.truth[m] : asked;
      if (truth) { running++; if (on) found++; }
      state.textContent = a.locked ? (on ? "Always on, detected" : "Always on, compressor resting")
        : asked ? (on ? `Detected, ${a.kw[m].toFixed(2)} kW` : "On, not detected yet")
        : (on ? "Detected, but you have it off" : "Off");
    } else {
      state.textContent = on ? `On, ${a.kw[m].toFixed(2)} kW`
        : a.locked ? "Always on, compressor resting" : "Off";
      state.textContent += ` \u00b7 ${rs(p.cost[m + 1])} today`;
    }
    li.querySelector(".switch").setAttribute("aria-checked",
      String(a.locked || (S.mode === "test" ? userOn(a.key, m) : on)));
  }
  for (const f of d.fans) {
    const li = document.querySelector(`#list li[data-key="${f.key}"]`), on = !!S.fans[f.key];
    if (!li) continue;
    li.classList.toggle("is-on", on);
    li.querySelector(".switch").setAttribute("aria-checked", String(on));
    li.querySelector(".state").textContent = on
      ? `On, ${(f.kw * 1000).toFixed(0)} W \u00b7 ${rs(S.devCost[f.key] || 0)} this session` : "Off";
  }
  $("score").textContent = S.mode === "test"
    ? (running ? `Found ${found} of ${running} drawing power` : "Nothing switched on") : "state and cost today";

  // selected appliance
  const a = byKey(S.selected);
  if (!a) {
    const f = devByKey(S.selected), on = !!S.fans[f.key];
    $("detail").innerHTML = `<h3>${f.name}</h3><div class="kind">${f.description}</div>
      <p>${on ? `On now, drawing ${(f.kw * 1000).toFixed(0)} W.` : "Off now."} This session: ${(S.devKwh[f.key] || 0).toFixed(3)} units, ${rs(S.devCost[f.key] || 0)}.</p>
      <p class="never">Left on for 8 hours a day it would use about ${(f.kw * 8 * 30).toFixed(1)} units a month, about ${rs(f.kw * 8 * 30 * d.rates[m])}. Its load is added to the meter reading while it is on.</p>`;
  } else {
  const p = S.prefix[a.key];
  let html = `<h3>${a.name}</h3><div class="kind">${a.product}. ${a.category_label}. ${a.note}</div>
    <p>Today so far: ${dur(p.on[m + 1])} drawing power, ${p.kwh[m + 1].toFixed(2)} units, ${rs(p.cost[m + 1])}.</p>`;
  if (a.tips.length) {
    html += a.tips.map((t) => `<div class="tip"><b>${t.title}</b><span>${t.detail} <em>Saves about Rs. ${t.saving} a month.</em></span></div>`).join("");
  } else {
    html += `<p class="never">${a.no_tip}</p>`;
  }
  $("detail").innerHTML = html;
  }

  // long-run alert
  const al = d.alerts.find((x) => m >= x.warn_at && m <= x.end + 20);
  const alertEl = $("alert");
  if (al) {
    if (alertEl.hidden && S.lastAlert !== al.key + al.start) {
      S.lastAlert = al.key + al.start;
      toast(`${al.name} has been running longer than normal.`, "warning");
    }
    alertEl.hidden = false;
    alertEl.innerHTML = `<strong>${al.name} has been on for ${dur(Math.min(m, al.end) - al.start)}.</strong> A normal run is ${al.normal}. ${al.advice}`;
  } else alertEl.hidden = true;
  if (S.hover) tip.innerHTML = describe(S.hover);
  setHeight();
}

// ================================================================= CANVASES
function fit(c) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2), w = c.clientWidth, h = c.clientHeight;
  if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
  const ctx = c.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
  return [ctx, w, h];
}

function drawWave(m) {
  const [ctx, w, h] = fit($("wave")), d = S.data, span = 120, from = Math.max(0, m - span + 1);
  const peak = Math.max(1, ...d.mains.slice(from, m + 1)) * 1.1;
  const X = (i) => ((i - (m - span + 1)) / (span - 1)) * w, Y = (v) => h - 2 - (v / peak) * (h - 6);
  ctx.beginPath();
  for (let i = from; i <= m; i++) ctx.lineTo(X(i), Y(d.mains[i]));
  ctx.strokeStyle = S.colors.current; ctx.lineWidth = 1.6; ctx.lineJoin = "round"; ctx.stroke();
  ctx.lineTo(X(m), h); ctx.lineTo(X(from), h); ctx.closePath();
  const g = ctx.createLinearGradient(0, 0, 0, h);
  g.addColorStop(0, "rgba(245,168,60,.28)"); g.addColorStop(1, "rgba(245,168,60,0)");
  ctx.fillStyle = g; ctx.fill();
  ctx.fillStyle = S.colors.current; ctx.beginPath(); ctx.arc(X(m) - 2, Y(d.mains[m]), 3, 0, 7); ctx.fill();
}

function drawTimeline(force) {
  if (!S.data) return;
  const m = Math.floor(S.playhead);
  if (!force && m === S.tlMinute) return;
  S.tlMinute = m;
  const [ctx, w] = fit($("timeline")), d = S.data, n = d.mains.length;
  const X = (min) => GUTTER + (min / n) * (w - GUTTER - 4);
  ctx.font = '12px "Hanken Grotesk", system-ui, sans-serif'; ctx.textBaseline = "middle";

  for (const p of d.periods) {                         // tariff band
    ctx.fillStyle = p.color; ctx.globalAlpha = p.mult === 1 ? 0.25 : 0.85;
    ctx.fillRect(X(p.start), 0, X(p.end) - X(p.start), 4);
  }
  ctx.globalAlpha = 1; ctx.fillStyle = S.colors.faint;
  for (let hr = 0; hr <= 24; hr += 3) {
    const x = X(hr * 60); ctx.textAlign = hr === 24 ? "right" : hr === 0 ? "left" : "center";
    ctx.fillText(String(hr).padStart(2, "0"), x, 11);
  }
  ctx.textAlign = "left";
  d.appliances.forEach((a, row) => {
    const y = TOP + 6 + row * (LANE + GAP);
    ctx.fillStyle = a.key === S.selected ? S.colors.plaster : S.colors.mist;
    ctx.fillText(a.name.replace(" (Geyser)", ""), 0, y + LANE / 2);
    ctx.fillStyle = S.colors.lane; ctx.fillRect(GUTTER, y, w - GUTTER - 4, LANE);
    const bars = (arr, upto, paint) => {
      let s = -1;
      for (let i = 0; i <= upto; i++) {
        const on = i < upto && arr[i];
        if (on && s < 0) s = i;
        if (!on && s >= 0) { paint(X(s), Math.max(1.5, X(i) - X(s))); s = -1; }
      }
    };
    if (S.mode === "test" && a.truth && !a.locked) {     // what was really on: outline
      ctx.strokeStyle = a.color; ctx.lineWidth = 1;
      bars(a.truth, n, (x, bw) => ctx.strokeRect(x + 0.5, y + 0.5, bw, LANE - 1));
    }
    ctx.fillStyle = a.color;                              // what the model found: filled
    bars(a.on, Math.min(n, m + 1), (x, bw) => ctx.fillRect(x, y + 2, bw, LANE - 4));
  });
  const px = X(m);
  ctx.fillStyle = S.colors.plaster; ctx.fillRect(px - 0.5, 4, 1.5, TOP + 6 + d.appliances.length * (LANE + GAP));
}
const tl = $("timeline");
const seek = (e) => {
  const r = tl.getBoundingClientRect(), n = S.data.mains.length;
  const min = ((e.clientX - r.left - GUTTER) / (r.width - GUTTER - 4)) * n;
  S.playhead = Math.max(0, Math.min(S.data.live_edge - 1, min)); S.lastPanel = 0;
};
tl.addEventListener("pointerdown", (e) => { if (S.data && !S.data.source.real) { tl.setPointerCapture(e.pointerId); seek(e); tl.onpointermove = seek; } });
tl.addEventListener("pointerup", () => { tl.onpointermove = null; });

// ================================================================= LOOP
const HOME_OFFSET = camera.position.clone().sub(controls.target);
const UP = new THREE.Vector3(0, 1, 0);
let introT = reduceMotion ? 1 : 0, shownKw = 0;
function intro(dt) {
  if (introT >= 1) return;
  introT = Math.min(1, introT + dt / INTRO_SEC);
  const k = 1 - Math.pow(1 - introT, 3);                    // ease-out
  camera.position.copy(HOME_OFFSET).applyAxisAngle(UP, (1 - k) * 0.5).add(controls.target);
  camera.zoom = 0.62 + 0.38 * k; camera.updateProjectionMatrix();
  scene.scale.y = 0.04 + 0.96 * k;
  controls.enabled = introT >= 1;
  pinsEl.style.opacity = introT >= 1 ? 1 : 0;
}
pinsEl.style.transition = "opacity .5s ease";
if (!reduceMotion) { pinsEl.style.opacity = 0; scene.scale.y = 0.04; controls.enabled = false; }

let last = performance.now();
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (S.data) intro(dt);
  controls.update();
  if (S.data) {
    const d = S.data, edge = d.live_edge - 1;
    if (d.source.real) S.playhead = edge;
    else if (S.playing) {
      S.playhead += dt * S.speed;
      if (S.playhead >= edge) { S.playhead = edge; S.playing = false; }
    }
    const m = Math.max(0, Math.min(edge, Math.floor(S.playhead)));

    for (const a of d.appliances) {
      const u = units[a.key]; if (!u) continue;
      const on = !!a.on[m], kw = a.kw[m];
      u.glow += ((on ? 1 : 0) - u.glow) * Math.min(1, dt * 6);
      u.mat.emissive.copy(u.color); u.mat.emissiveIntensity = u.glow * 0.9;
      u.light.intensity = u.glow * (P === PALETTE.light ? 4 : 9);
      u.traceMat.color.setHex(P.trace).lerp(AMBER_COLOR, u.glow * 0.6);
      u.ring.visible = a.key === S.selected;
      if (!reduceMotion) u.phase = (u.phase + (dt * (1.6 + kw * 1.2)) / u.pathLen) % 1;
      u.dots.forEach((dot, i) => {
        dot.visible = u.glow > 0.4;
        if (dot.visible) dot.position.copy(pointOnPath(u, (u.phase + i / u.dots.length) % 1));
      });
      const pin = pins[a.key], asked = S.mode === "test" ? userOn(a.key, m) : on;
      pin.classList.toggle("is-on", on);
      pin.classList.toggle("is-selected", a.key === S.selected);
      pin.classList.toggle("is-wrong", S.mode === "test" && !a.locked && asked !== on);
      pin.querySelector(".v").textContent = on ? `${kw.toFixed(2)} kW` : "";
      place(pin, u.anchor, true);
    }
    // fans: spin, pin, and their share of the meter reading
    const extra = fanKw();
    if (S.fanMinute !== undefined && m > S.fanMinute && m - S.fanMinute <= 30) {
      for (const f of d.fans) {
        if (!S.fans[f.key]) continue;
        const kwh = (f.kw * (m - S.fanMinute)) / 60, cost = kwh * d.rates[m];
        S.devKwh[f.key] = (S.devKwh[f.key] || 0) + kwh; S.devCost[f.key] = (S.devCost[f.key] || 0) + cost;
        S.fanKwh += kwh; S.fanCost += cost;
      }
    }
    S.fanMinute = m;
    for (const f of d.fans) {
      const u = fanUnits[f.key], on = !!S.fans[f.key];
      if (!u) continue;
      u.glow += ((on ? 1 : 0) - u.glow) * Math.min(1, dt * 5);
      if (u.rotor) {
        u.speed += ((on ? 11 : 0) - u.speed) * Math.min(1, dt * 1.8);
        if (!reduceMotion) u.rotor.rotation.y -= u.speed * dt;
      }
      if (u.mat) u.mat.emissiveIntensity = u.glow * (u.kind === "tv" ? 0.9 : 1.3);
      if (u.light) u.light.intensity = u.glow * u.lightMax * (P === PALETTE.light ? 0.5 : 1);
      const pin = pins[f.key];
      pin.classList.toggle("is-on", on);
      pin.classList.toggle("is-selected", f.key === S.selected);
      pin.querySelector(".v").textContent = on ? `${(f.kw * 1000).toFixed(0)} W` : "";
      place(pin, u.anchor, true);
    }

    // outer walls on the camera's side drop so the rooms stay in view
    const cdx = camera.position.x - controls.target.x, cdz = camera.position.z - controls.target.z;
    const clen = Math.hypot(cdx, cdz) || 1;
    for (const wall of outerWalls) {
      const facing = (wall.nx * cdx + wall.nz * cdz) / clen > 0.12;
      wall.s += ((facing ? LOW / FULL : 1) - wall.s) * Math.min(1, dt * 7);
      wall.mesh.scale.y = wall.s;
      wall.panes.forEach((p) => { p.visible = wall.s > 0.75; });
    }

    const total = d.mains[m] + extra;
    shownKw += (total - shownKw) * Math.min(1, dt * 7);
    $("kw").textContent = shownKw.toFixed(2);
    pins.meter.querySelector(".v").textContent = `${total.toFixed(2)} kW`;
    place(pins.meter, meterAnchor, true);
    roomLabels.forEach((r) => place(r.el, r.pos, false));
    meterLight.intensity = (1.5 + Math.min(4, total) * 1.2) * (P === PALETTE.light ? 0.35 : 1);

    // impulse LED: faster with load, like the one on a real meter
    const gapMs = Math.max(140, Math.min(3000, 1500 / Math.max(0.05, total)));
    if (now - S.pulseAt > gapMs) { S.pulseAt = now; $("pulse").classList.add("is-lit"); }
    else if (now - S.pulseAt > 80) $("pulse").classList.remove("is-lit");

    if (m !== S.lastMinute || now - S.lastPanel > 400) {
      S.lastMinute = m; S.lastPanel = now;
      updatePanel(m); drawWave(m); drawTimeline(true);
    }
  }
  renderer.render(scene, camera);
}

window.addEventListener("message", (e) => {
  if (e.data && e.data.type === "streamlit:render") {
    const data = e.data.args && e.data.args.data;
    if (data && data.version !== S.version) load(data);
  }
});
post("streamlit:componentReady", { apiVersion: 1 });
resize();
requestAnimationFrame(frame);
window.__home = { S, load };   // for tests
