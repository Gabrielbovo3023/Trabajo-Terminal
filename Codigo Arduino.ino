//Codigo final envio de video, recibe mensajes y se visualiza en el OLED

#include "WifiCam.hpp"
#include <WiFi.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

static const char* WIFI_SSID = "wifi2";
static const char* WIFI_PASS = "gafas123";

esp32cam::Resolution initialResolution;

#define I2C_SDA 1
#define I2C_SCL 3
#define OLED_RESET 13
Adafruit_SSD1306 display(OLED_RESET);

WebServer server(80);

void setup() {
  Serial.begin(115200);
  Serial.println();
  delay(2000);

  // Inicia la comunicación I2C para el display OLED
  Wire.begin(I2C_SDA, I2C_SCL);
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);

  WiFi.persistent(false);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  if (WiFi.waitForConnectResult() != WL_CONNECTED) {
    Serial.println("WiFi failure");
    delay(5000);
    ESP.restart();
  }
  Serial.println("WiFi connected");

  {
    using namespace esp32cam;

    initialResolution = Resolution::find(1024, 768);

    Config cfg;
    cfg.setPins(pins::AiThinker);
    cfg.setResolution(initialResolution);
    cfg.setJpeg(80);

    bool ok = Camera.begin(cfg);
    if (!ok) {
      Serial.println("camera initialize failure");
      delay(5000);
      ESP.restart();
    }
    Serial.println("camera initialize success");
  }

  Serial.println("camera starting");
  Serial.print("http://");
  Serial.println(WiFi.localIP());

  addRequestHandlers();
  
  server.on("/tu_ruta", HTTP_GET, []() {
    String message = server.arg("message");
    Serial.print("Mensaje recibido: ");
    Serial.println(message);
    
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(WHITE);
    display.setCursor(0,0);
    display.println(message);
    display.display();  
    server.send(200, "text/plain", "Mensaje recibido correctamente");
  });
  server.begin();
}

void loop() {
  server.handleClient();
}
