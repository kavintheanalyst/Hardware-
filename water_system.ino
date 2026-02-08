// -------- PINS --------
#define SOIL_PIN A0

#define TANK_LOW  2    // Bottom float (EMPTY)
#define TANK_HIGH 3    // Top float (FULL)

#define IRRIGATION_RELAY 7
#define REFILL_RELAY     8

// -------- SETTINGS --------
int threshold = 550;   // Adjust after calibration

void setup() {

  Serial.begin(9600);

  pinMode(IRRIGATION_RELAY, OUTPUT);
  pinMode(REFILL_RELAY, OUTPUT);

  pinMode(TANK_LOW, INPUT_PULLUP);
  pinMode(TANK_HIGH, INPUT_PULLUP);

  // Relays OFF (active LOW)
  digitalWrite(IRRIGATION_RELAY, HIGH);
  digitalWrite(REFILL_RELAY, HIGH);
}

void loop() {

  int soil = analogRead(SOIL_PIN);

  bool tankEmpty = digitalRead(TANK_LOW) == LOW;
  bool tankFull  = digitalRead(TANK_HIGH) == LOW;

  bool soilDry = soil > threshold;

  // -------- TANK REFILL SYSTEM --------
  if (tankEmpty && !tankFull) {
    digitalWrite(REFILL_RELAY, LOW);   // Refill ON
  }

  if (tankFull) {
    digitalWrite(REFILL_RELAY, HIGH);  // Refill OFF
  }

  // -------- PLANT IRRIGATION SYSTEM --------
  if (soilDry && !tankEmpty) {
    digitalWrite(IRRIGATION_RELAY, LOW);   // Irrigation ON
  } else {
    digitalWrite(IRRIGATION_RELAY, HIGH);  // Irrigation OFF
  }

  // -------- DEBUG --------
  Serial.print("Soil=");
  Serial.print(soil);

  Serial.print(" Empty=");
  Serial.print(tankEmpty);

  Serial.print(" Full=");
  Serial.print(tankFull);

  Serial.print(" Irrigation=");
  Serial.print((soilDry && !tankEmpty) ? "ON" : "OFF");

  Serial.print(" Refill=");
  Serial.println((tankEmpty && !tankFull) ? "ON" : "OFF");

  delay(1000);
}
