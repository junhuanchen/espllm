#pragma once

#ifdef ESP_PLATFORM

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "driver/usb_serial_jtag.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

class IdfSerialCompat {
    int pending_byte_ = -1;

    void write_text_(const char* text, size_t length) const {
        size_t start = 0;
        for (size_t i = 0; i < length; ++i) {
            if (text[i] != '\n') continue;
            if (i > start) {
                usb_serial_jtag_write_bytes(text + start, i - start, pdMS_TO_TICKS(20));
            }
            // ESP-IDF writes raw bytes, unlike Arduino Serial. Convert LF to
            // CRLF so terminals that require a carriage return do not drift.
            if (i > 0 && text[i - 1] == '\r') {
                usb_serial_jtag_write_bytes("\n", 1, pdMS_TO_TICKS(20));
            } else {
                usb_serial_jtag_write_bytes("\r\n", 2, pdMS_TO_TICKS(20));
            }
            start = i + 1;
        }
        if (start < length) {
            usb_serial_jtag_write_bytes(text + start, length - start, pdMS_TO_TICKS(20));
        }
    }

public:
    void begin(unsigned long) {
        usb_serial_jtag_driver_config_t config = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
        usb_serial_jtag_driver_install(&config);
    }

    int available() {
        if (pending_byte_ >= 0) return 1;
        uint8_t byte = 0;
        if (usb_serial_jtag_read_bytes(&byte, 1, 0) == 1) pending_byte_ = byte;
        return pending_byte_ >= 0;
    }

    int read() {
        if (pending_byte_ >= 0) {
            int byte = pending_byte_;
            pending_byte_ = -1;
            return byte;
        }
        uint8_t byte = 0;
        return usb_serial_jtag_read_bytes(&byte, 1, 0) == 1 ? byte : -1;
    }

    void print(const char* text) const { write_text_(text, strlen(text)); }
    void print(char value) const { write_text_(&value, 1); }
    void println() const { print("\n"); }
    void println(const char* text) const { print(text); println(); }
    int printf(const char* format, ...) const {
        va_list args;
        va_start(args, format);
        char output[256];
        int written = vsnprintf(output, sizeof(output), format, args);
        va_end(args);
        if (written <= 0) return written;
        size_t length = (written < (int)sizeof(output)) ? (size_t)written : sizeof(output) - 1;
        write_text_(output, length);
        return written;
    }
};

struct IdfEspCompat {
    size_t getPsramSize() const { return heap_caps_get_total_size(MALLOC_CAP_SPIRAM); }
    size_t getFreeHeap() const { return heap_caps_get_free_size(MALLOC_CAP_8BIT); }
};

static IdfSerialCompat Serial;
static IdfEspCompat ESP;

static inline unsigned long millis() { return (unsigned long)(esp_timer_get_time() / 1000); }
static inline void delay(unsigned long ms) { vTaskDelay(pdMS_TO_TICKS(ms)); }
static inline void yield() { vTaskDelay(1); }

#define pgm_read_byte(address) (*(const uint8_t*)(address))
#define pgm_read_dword(address) (*(const uint32_t*)(address))
#define PROGMEM

#endif
