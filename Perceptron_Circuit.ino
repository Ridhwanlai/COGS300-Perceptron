// ─────────────────────────────────────────────────────────
//  PERCEPTRON READER — Arduino Sketch
//  Reads your analog perceptron circuit and streams
//  live data to the Processing visualizer over serial.
// ─────────────────────────────────────────────────────────
//
//  EXTRA WIRES TO ADD TO YOUR BREADBOARD:
//
//  1. Arduino GND  →  Breadboard GND rail (blue)
//
//  2. DIP switch top pins: DISCONNECT from +9V rail
//     Reconnect both to Arduino 5V pin instead.
//     (Row 26 and Row 27 will now be 0V or 5V — safe for Arduino)
//
//  3. Pot 3 left terminal (Row 37): DISCONNECT from +9V rail
//     Reconnect to Arduino 5V pin instead.
//     (Bias wiper now ranges 0–5V — safe for Arduino)
//
//  4. Arduino D2  →  Row 26  (x₁, DIP switch 1 output)
//  5. Arduino D4  →  Row 27  (x₂, DIP switch 2 output)
//  6. Arduino A0  →  Row 31  (Pot 1 wiper, weight w₁)
//  7. Arduino A1  →  Row 35  (Pot 2 wiper, weight w₂)
//  8. Arduino A2  →  Row 39  (Pot 3 wiper, bias b)
//
//  9. TL074 output voltage divider (IMPORTANT — 9V would fry A3):
//     Row 5 left (TL074 Pin 1)  →  220Ω  →  [Junction A]
//     [Junction A]              →  220Ω  →  GND rail
//     Arduino A3                →  [Junction A]
//     This scales 0–9V down to 0–4.5V before hitting A3.
//     Uses 2 of your remaining 220Ω resistors.
//
// ─────────────────────────────────────────────────────────

const int PIN_X1   = 2;
const int PIN_X2   = 4;
const int PIN_W1   = A0;
const int PIN_W2   = A1;
const int PIN_BIAS = A2;
const int PIN_SUM  = A3;  // TL074 output (through voltage divider)

void setup() {
  Serial.begin(9600);
  pinMode(PIN_X1, INPUT);
  pinMode(PIN_X2, INPUT);
  Serial.println("PERCEPTRON_READY");
}

void loop() {
  int x1   = digitalRead(PIN_X1);
  int x2   = digitalRead(PIN_X2);
  float w1   = analogRead(PIN_W1)   / 1023.0;
  float w2   = analogRead(PIN_W2)   / 1023.0;
  float bias = analogRead(PIN_BIAS) / 1023.0;

  // TL074 output, scaled 0–4.5V → 0.0–1.0
  // (voltage divider already brought it into safe range)
  float sumV = analogRead(PIN_SUM) / 1023.0;

  // TL074 is inverting: LOW output means the weighted sum
  // exceeded the threshold. Adjust 0.40 if needed.
  int fired = (sumV < 0.40) ? 1 : 0;

  // CSV format: x1,x2,w1,w2,bias,sumV,fired
  Serial.print(x1);       Serial.print(",");
  Serial.print(x2);       Serial.print(",");
  Serial.print(w1, 3);    Serial.print(",");
  Serial.print(w2, 3);    Serial.print(",");
  Serial.print(bias, 3);  Serial.print(",");
  Serial.print(sumV, 3);  Serial.print(",");
  Serial.println(fired);

  delay(100);  // 10 readings per second
}
