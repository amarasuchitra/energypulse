// EnergyPulse main-meter node
// ESP32 + PZEM-004T v3.0 (with its clip-on CT clamp round the main live wire).
// Sends total household power to the EnergyPulse receiver once every 10 s.
//
// SAFETY: the CT clamp is non-invasive, but the PZEM voltage terminals carry
// mains voltage. Have a licensed electrician fit it at the distribution board.
//
// Library: "PZEM004Tv30" by Jakub Mandula (Arduino Library Manager).
// Wiring:  PZEM TX -> GPIO16 (RX2),  PZEM RX -> GPIO17 (TX2),  5V, GND.

#include <WiFi.h>
#include <HTTPClient.h>
#include <PZEM004Tv30.h>

const char* WIFI_SSID = "your-wifi-name";
const char* WIFI_PASS = "your-wifi-password";
// IP address of the computer running:  python live_meter.py serve
const char* RECEIVER  = "http://192.168.1.50:8600/reading";

PZEM004Tv30 pzem(Serial2, 16, 17);

void setup() {
  Serial.begin(115200);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println("\nWiFi connected");
}

void loop() {
  float watts = pzem.power();
  if (!isnan(watts) && WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(RECEIVER);
    http.addHeader("Content-Type", "application/json");
    String body = "{\"power_w\":" + String(watts, 1) + ",\"device\":\"esp32-pzem\"}";
    int code = http.POST(body);
    Serial.printf("%.1f W -> HTTP %d\n", watts, code);
    http.end();
  }
  delay(10000);
}
