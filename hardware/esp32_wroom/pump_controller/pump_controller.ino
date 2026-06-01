#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include <WiFiClientSecure.h>

const char* ssid = "Galaxy";
const char* password = "30263026";
const char* backendURL = "https://agriguard-g3hq.onrender.com/api/esp/register";

#define RELAY_PIN       26
#define MOISTURE_PIN    34
#define AUTO_INTERVAL   300000 // 5 minutes

WebServer server(80);

bool pumpOn = false;
bool autoMode = false;
unsigned long lastAutoCheck = 0;

int getMoisturePercent() {
  int raw = analogRead(MOISTURE_PIN);
  // 4095 = dry = 0%, ~1400 = fully wet = 100%
  int percent = map(raw, 4095, 1400, 0, 100);
  return constrain(percent, 0, 100);
}

void runAutoLogic() {
  int moisture = getMoisturePercent();
  if (moisture < 30) {
    digitalWrite(RELAY_PIN, LOW);
    pumpOn = true;
  } else if (moisture > 70) {
    digitalWrite(RELAY_PIN, HIGH);
    pumpOn = false;
  }
}

void registerWithBackend() {
  WiFiClientSecure client;
  client.setInsecure();
  
  HTTPClient http;
  String payload = "{\"deviceId\":\"esp-wroom\",\"ip\":\"" + WiFi.localIP().toString() + "\"}";

  Serial.println("Registering with: " + String(backendURL));
  Serial.println("Payload: " + payload);
  
  http.begin(client, backendURL);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(payload);
  Serial.printf("Response code: %d\n", code);
  Serial.println("Response: " + http.getString());
  http.end();
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
                "\",\"moisture\":" + String(moisture) +
                ",\"autoMode\":" + String(autoMode ? "true" : "false") + "}";
  server.send(200, "application/json", json);
}

void handleAutoOn() {
  autoMode = true;
  server.send(200, "application/json", "{\"autoMode\":true}");
  Serial.println("Auto mode enabled");
}

void handleAutoOff() {
  autoMode = false;
  // turn pump off when disabling auto — safer default
  digitalWrite(RELAY_PIN, HIGH);
  pumpOn = false;
  server.send(200, "application/json", "{\"autoMode\":false}");
  Serial.println("Auto mode disabled");
}

void handleAutoCheck() {
  int moisture = getMoisturePercent();
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
    String json = "{\"action\":\"no_change\",\"moisture\":" + String(moisture) + "}";
    server.send(200, "application/json", json);
  }
}

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
  Serial.begin(115200);

  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, HIGH); // pump off on boot

  // Static IP — always gets same address regardless of reboots
  // IPAddress local_IP(10, 31, 234, 201);
  // IPAddress gateway(10, 31, 234, 1);
  // IPAddress subnet(255, 255, 255, 0);
  // WiFi.config(local_IP, gateway, subnet);

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
  Serial.println(WiFi.localIP());
  Serial.println("Gateway: ");
  Serial.println(WiFi.gatewayIP());

  registerWithBackend();

  server.on("/pump/on", HTTP_GET, handlePumpOn);
  server.on("/pump/off", HTTP_GET, handlePumpOff);
  server.on("/status", HTTP_GET, handleStatus);
  server.on("/auto/on", HTTP_GET, handleAutoOn);
  server.on("/auto/off", HTTP_GET, handleAutoOff);
  server.on("/auto/check", HTTP_GET, handleAutoCheck);
  server.begin();
  Serial.println("Server started");
}

void loop() {
  server.handleClient();

  // Run auto logic every 5 minutes when auto mode is enabled
  if (autoMode && millis() - lastAutoCheck >= AUTO_INTERVAL) {
    lastAutoCheck = millis();
    runAutoLogic();
    Serial.printf("Auto check — moisture: %d%% pump: %s\n",
      getMoisturePercent(), pumpOn ? "on" : "off");
  }
}