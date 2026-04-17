// ─────────────────────────────────────────────────────────
//  PERCEPTRON MONITOR — Processing Sketch
//  Real-time visualization of your analog perceptron circuit.
//
//  HOW TO RUN:
//  1. Install Processing from https://processing.org
//  2. Upload perceptron_reader.ino to your Arduino first
//  3. Open this file in Processing and click Run (▶)
//  4. If no Arduino is found, it runs in DEMO MODE automatically
//
//  IF THE WRONG PORT IS SELECTED:
//  Check the console output — it lists all available ports.
//  Change ports[0] to ports[1], ports[2] etc. and re-run.
// ─────────────────────────────────────────────────────────

import processing.serial.*;

// ── Serial ────────────────────────────────────────────────
Serial port;
boolean demo = true;

// ── Live data ─────────────────────────────────────────────
float w1 = 0.5, w2 = 0.5, b = 0.5, sigma = 0.5;
int   x1 = 0, x2 = 0, fired = 0;
float firedGlow = 0;

// ── Oscilloscope history ──────────────────────────────────
float[] history = new float[180];
int     hIdx    = 0;

// ── Colors ────────────────────────────────────────────────
final color BG     = #070604;
final color AMBER  = #FFA500;
final color DIM    = #5A3800;
final color BRIGHT = #FFD080;
final color GREEN  = #39FF14;
final color RED    = #FF3333;

// ─────────────────────────────────────────────────────────
void setup() {
  size(920, 570);
  smooth(8);

  for (int i = 0; i < history.length; i++) history[i] = 0;

  // Try to connect to first available serial port
  String[] ports = Serial.list();
  println("─── Available serial ports ───");
  for (int i = 0; i < ports.length; i++) println(i + ": " + ports[i]);
  println("──────────────────────────────");

  if (ports.length > 0) {
    try {
      port = new Serial(this, ports[6], 9600);
      port.bufferUntil('\n');
      demo = false;
      println("Connected to: " + ports[6]);
    } catch (Exception e) {
      println("Could not open " + ports[6] + " — running in demo mode.");
    }
  } else {
    println("No serial ports found — running in demo mode.");
  }

  textFont(createFont("Courier New Bold", 12, true));
  frameRate(30);
}

// ─────────────────────────────────────────────────────────
void serialEvent(Serial p) {
  String s = trim(p.readStringUntil('\n'));
  if (s == null || s.equals("PERCEPTRON_READY")) return;

  String[] v = split(s, ',');
  if (v.length != 7) return;

  x1    = int(v[0]);
  x2    = int(v[1]);
  w1    = float(v[2]);
  w2    = float(v[3]);
  b     = float(v[4]);
  sigma = float(v[5]);
  fired = int(v[6]);

  history[hIdx] = sigma;
  hIdx = (hIdx + 1) % history.length;
}

// ─────────────────────────────────────────────────────────
void draw() {
  if (demo) animateDemo();

  firedGlow = lerp(firedGlow, fired, 0.12);

  background(BG);
  drawGrid();
  drawHeader();
  drawNetwork();
  drawMetrics();
  drawScope();
  drawScanlines();
}

// ── Demo mode: animated fake data ────────────────────────
void animateDemo() {
  float t = millis() / 1000.0;
  w1    = 0.5 + 0.45 * sin(t * 0.70);
  w2    = 0.5 + 0.45 * sin(t * 0.50 + 1.2);
  b     = 0.4 + 0.35 * sin(t * 0.30);
  x1    = sin(t * 0.90) > 0.0 ? 1 : 0;
  x2    = cos(t * 0.65) > 0.1 ? 1 : 0;
  sigma = constrain((w1 * x1 + w2 * x2 + b * 0.5) / 1.5, 0, 1);
  fired = sigma > 0.55 ? 1 : 0;

  history[hIdx] = sigma;
  hIdx = (hIdx + 1) % history.length;
}

// ── Background dot grid ───────────────────────────────────
void drawGrid() {
  stroke(18, 14, 4);
  strokeWeight(1);
  for (int x = 0; x < width;  x += 30) line(x, 0, x, height);
  for (int y = 0; y < height; y += 30) line(0, y, width, y);
}

// ── Header bar ────────────────────────────────────────────
void drawHeader() {
  noStroke();
  fill(12, 9, 2);
  rect(0, 0, width, 36);
  stroke(DIM);
  strokeWeight(1);
  line(0, 36, width, 36);

  noStroke();
  fill(BRIGHT);
  textSize(13);
  textAlign(LEFT, CENTER);
  text("PERCEPTRON MONITOR  v1.0", 14, 18);

  textAlign(CENTER, CENTER);
  fill(DIM);
  text(demo ? "[ DEMO MODE — connect Arduino to see live data ]"
            : "[ LIVE — reading from Arduino ]", width / 2, 18);

  textAlign(RIGHT, CENTER);
  fill(demo ? DIM : GREEN);
  text(demo ? "● DEMO" : "● LIVE", width - 14, 18);
}

// ── Neural network diagram ────────────────────────────────
void drawNetwork() {
  // Node positions
  float ix1 = 170, iy1 = 160;
  float ix2 = 170, iy2 = 295;
  float ibx = 170, iby = 420;
  float sx  = 430, sy  = 268;
  float ox  = 660, oy  = 268;

  // Weight connections
  weightLine(ix1, iy1, sx, sy, w1, x1 == 1, "w\u2081=" + nf(w1, 1, 2));
  weightLine(ix2, iy2, sx, sy, w2, x2 == 1, "w\u2082=" + nf(w2, 1, 2));
  weightLine(ibx, iby, sx, sy, b,  true,     "b =" + nf(b,  1, 2));

  // Σ → Output line
  stroke(lerpColor(DIM, RED, firedGlow));
  strokeWeight(lerp(1.5, 5, firedGlow));
  line(sx + 29, sy, ox - 31, oy);

  // Nodes (drawn on top of lines)
  inputNode(ix1, iy1, x1, "x\u2081");
  inputNode(ix2, iy2, x2, "x\u2082");
  biasNode(ibx, iby);
  sumNode(sx, sy, sigma);
  outputNode(ox, oy, fired);
}

// ── Individual node drawers ───────────────────────────────
void inputNode(float x, float y, int active, String lbl) {
  if (active == 1) {
    noStroke();
    for (int r = 46; r > 20; r -= 6) { fill(57, 255, 20, 12); ellipse(x, y, r*2, r*2); }
  }
  color nc = active == 1 ? GREEN : DIM;
  strokeWeight(2);
  stroke(nc);
  fill(active == 1 ? color(8, 35, 8) : color(14, 11, 4));
  ellipse(x, y, 46, 46);
  fill(nc);
  textSize(16);
  textAlign(CENTER, CENTER);
  text(active == 1 ? "1" : "0", x, y);
  noStroke();
  fill(BRIGHT);
  textSize(12);
  textAlign(RIGHT, CENTER);
  text(lbl, x - 28, y);
  fill(nc);
  textSize(10);
  textAlign(CENTER, TOP);
  text(active == 1 ? "ON" : "OFF", x, y + 27);
}

void biasNode(float x, float y) {
  strokeWeight(1.5);
  stroke(color(150, 100, 0, 140));
  fill(color(18, 13, 3));
  ellipse(x, y, 38, 38);
  noStroke();
  fill(DIM);
  textSize(13);
  textAlign(CENTER, CENTER);
  text("b", x, y);
  fill(DIM);
  textSize(10);
  textAlign(RIGHT, CENTER);
  text("bias", x - 23, y);
}

void sumNode(float x, float y, float val) {
  noStroke();
  for (int r = 60; r > 25; r -= 7) { fill(255, 165, 0, int(val * 18)); ellipse(x, y, r*2, r*2); }
  strokeWeight(2.5);
  stroke(lerpColor(DIM, AMBER, val));
  fill(color(int(val * 35), int(val * 22), 0));
  ellipse(x, y, 58, 58);
  noStroke();
  fill(BRIGHT);
  textSize(24);
  textAlign(CENTER, CENTER);
  text("\u03A3", x, y);
  fill(DIM);
  textSize(10);
  textAlign(CENTER, TOP);
  text(nf(val, 1, 2), x, y + 33);
}

void outputNode(float x, float y, int fire) {
  if (fire == 1) {
    noStroke();
    for (int r = 74; r > 30; r -= 8) { fill(255, 60, 60, 10); ellipse(x, y, r*2, r*2); }
  }
  color nc = fire == 1 ? RED : DIM;
  strokeWeight(3);
  stroke(nc);
  fill(fire == 1 ? color(55, 4, 4) : color(14, 11, 4));
  ellipse(x, y, 62, 62);
  noStroke();
  fill(nc);
  textSize(11);
  textAlign(CENTER, CENTER);
  text(fire == 1 ? "LED\nON" : "LED\nOFF", x, y);
  if (firedGlow > 0.15) {
    fill(RED);
    textSize(13);
    textAlign(CENTER, TOP);
    text("FIRED!", x, y + 37);
  }
}

void weightLine(float x1c, float y1c, float x2c, float y2c, float w, boolean act, String lbl) {
  float mx = (x1c + x2c) / 2.0;
  float my = (y1c + y2c) / 2.0;
  int   a  = int(map(w, 0, 1, 40, 230));
  color lc = act ? color(255, 165, 0, a) : color(90, 55, 0, a);
  stroke(lc);
  strokeWeight(map(w, 0, 1, 1.0, 4.5));
  noFill();
  bezier(x1c + 23, y1c,  mx, y1c,  mx, y2c,  x2c - 29, y2c);
  noStroke();
  fill(act ? BRIGHT : DIM);
  textSize(11);
  textAlign(CENTER, BOTTOM);
  text(lbl, mx - 5, my - 14);
}

// ── Right metrics panel ───────────────────────────────────
void drawMetrics() {
  float px = 730, py = 45, pw = 168, ph = 395;
  noStroke();
  fill(10, 8, 2);
  rect(px - 8, py, pw, ph, 5);
  stroke(DIM);
  strokeWeight(1);
  noFill();
  rect(px - 8, py, pw, ph, 5);

  noStroke();
  fill(BRIGHT);
  textSize(11);
  textAlign(LEFT, TOP);
  text("\u2500\u2500\u2500 WEIGHTS \u2500\u2500\u2500", px, py + 10);
  metricBar(px, py + 35,  pw - 20, w1,    "w\u2081", x1 == 1 ? GREEN : AMBER);
  metricBar(px, py + 72,  pw - 20, w2,    "w\u2082", x2 == 1 ? GREEN : AMBER);
  metricBar(px, py + 109, pw - 20, b,     "b ",  BRIGHT);

  text("\u2500\u2500\u2500 SUM \u2500\u2500\u2500\u2500", px, py + 148);
  metricBar(px, py + 173, pw - 20, sigma, "\u03A3 ", lerpColor(AMBER, RED, firedGlow));

  text("\u2500\u2500 OUTPUT \u2500\u2500\u2500", px, py + 215);
  fill(firedGlow > 0.5 ? RED : DIM);
  textSize(34);
  textAlign(CENTER, CENTER);
  text(fired == 1 ? "ON" : "OFF", px + pw / 2 - 10, py + 275);

  noStroke();
  fill(DIM);
  textSize(10);
  textAlign(LEFT, TOP);
  text("x1=" + x1 + "  x2=" + x2, px, py + 318);
  text("w1=" + nf(w1, 1, 3),       px, py + 333);
  text("w2=" + nf(w2, 1, 3),       px, py + 348);
  text("b =" + nf(b,  1, 3),       px, py + 363);
  text("\u03A3 =" + nf(sigma, 1, 3), px, py + 378);
}

void metricBar(float x, float y, float w, float val, String lbl, color col) {
  stroke(DIM);
  strokeWeight(1);
  noFill();
  rect(x, y, w, 19, 3);
  noStroke();
  fill(col);
  rect(x + 1, y + 1, (w - 2) * val, 17, 2);
  noStroke();
  fill(BRIGHT);
  textSize(10);
  textAlign(LEFT, CENTER);
  text(lbl + " " + nf(val, 1, 2), x, y - 9);
}

// ── Oscilloscope trace ────────────────────────────────────
void drawScope() {
  float sx = 10, sy = 462, sw = 710, sh = 92;

  noStroke();
  fill(4, 7, 2);
  rect(sx, sy, sw, sh, 5);
  stroke(DIM);
  strokeWeight(1);
  noFill();
  rect(sx, sy, sw, sh, 5);

  // Grid lines
  stroke(20, 28, 12);
  strokeWeight(1);
  for (int i = 1; i < 4; i++) line(sx, sy + sh * i / 4.0, sx + sw, sy + sh * i / 4.0);
  for (int i = 1; i < 7; i++) line(sx + sw * i / 7.0, sy, sx + sw * i / 7.0, sy + sh);

  // Threshold line (dashed manually)
  float ty = sy + sh - 0.4 * sh;
  stroke(DIM);
  strokeWeight(1);
  for (int xi = int(sx); xi < int(sx + sw) - 4; xi += 9) line(xi, ty, xi + 5, ty);

  noStroke();
  fill(DIM);
  textSize(9);
  textAlign(LEFT, CENTER);
  text("\u03B8=0.40", sx + 6, ty - 8);
  textAlign(LEFT, TOP);
  text("\u03A3 OUTPUT HISTORY", sx + 6, sy + 5);

  // Waveform
  stroke(GREEN);
  strokeWeight(2);
  noFill();
  beginShape();
  for (int i = 0; i < history.length; i++) {
    int   idx = (hIdx + i) % history.length;
    float hx  = sx + map(i, 0, history.length - 1, 4, sw - 4);
    float hy  = sy + sh - 5 - history[idx] * (sh - 10);
    vertex(hx, hy);
  }
  endShape();
}

// ── CRT scanline overlay ──────────────────────────────────
void drawScanlines() {
  stroke(0, 0, 0, 32);
  strokeWeight(2);
  for (int y = 0; y < height; y += 4) line(0, y, width, y);
}
