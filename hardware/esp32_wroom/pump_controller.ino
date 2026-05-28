#include <WiFi.h>
#include <WebServer.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include <ESPmDNS.h>

const char* ssid = "Galaxy";
const char* password = "30263026";

#define RELAY_PIN     26
#define MOISTURE_PIN  34  

WebServer server(80);

bool pumpOn = false;

// Convert raw ADC (0-4095) to moisture percentage
// 4095 = completely dry, 0 = completely wet (sensor reads inverse)
int getMoisturePercent() {
  int raw = analogRead(MOISTURE_PIN);
  // 4095 = dry = 0%, ~1400 = fully wet = 100%
  int percent = map(raw, 4095, 1400, 0, 100);
  return constrain(percent, 0, 100);
}

void handlePumpOn() {
  digitalWrite(RELAY_PIN, LOW);
  pumpOn = true;
  server.send(200, "application/json", "{\"pump\":\"on\"}");
}

void handlePumpOff() {
  digitalWrite(RELAY_PIN, HIGH);
  pumpOn = false;
  server.send(200, "application/json", "{\"pump\":\"off\"}");
}

void handleStatus() {
  int moisture = getMoisturePercent();
  String json = "{\"pump\":\"" + String(pumpOn ? "on" : "off") +
                "\",\"moisture\":" + String(moisture) + "}";
  server.send(200, "application/json", json);
}

void handleAutoWater() {
  int moisture = getMoisturePercent();
  
  // Auto mode: pump on below 30%, off above 60% — prevents rapid switching
  if (moisture < 30) {
    digitalWrite(RELAY_PIN, LOW);
    pumpOn = true;
    String json = "{\"action\":\"pump_on\",\"reason\":\"soil_dry\",\"moisture\":" + String(moisture) + "}";
    server.send(200, "application/json", json);
  } else if (moisture > 70) {
    digitalWrite(RELAY_PIN, HIGH);
    pumpOn = false;
    String json = "{\"action\":\"pump_off\",\"reason\":\"soil_wet\",\"moisture\":" + String(moisture) + "}";
    server.send(200, "application/json", json);
  } else {
    // In range — no change
    String json = "{\"action\":\"no_change\",\"moisture\":" + String(moisture) + "}";
    server.send(200, "application/json", json);
  }
}

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
  Serial.begin(115200);

  pinMode(RELAY_PIN, OUTPUT);
  // AO pin needs no pinMode — analogRead works directly

  digitalWrite(RELAY_PIN, HIGH); // pump off on boot

  WiFi.setSleep(false);
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("ESP32-WROOM IP: ");
  Serial.println(WiFi.localIP());
  
  if (MDNS.begin("agriguard-wroom")) {
  Serial.println("mDNS started: agriguard-wroom.local");
}

  server.on("/pump/on", HTTP_GET, handlePumpOn);
  server.on("/pump/off", HTTP_GET, handlePumpOff);
  server.on("/status", HTTP_GET, handleStatus);
  server.on("/auto", HTTP_GET, handleAutoWater);
  server.begin();
  Serial.println("Server started");
}

void loop() {
  server.handleClient();
}
