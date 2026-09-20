# ESP32-S3 ESP-IDF Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and flash ESP-LLM on an ESP32-S3 N16R8 using ESP-IDF 5.5.5.

**Architecture:** Preserve the inference engine and generated model header. Add an ESP-IDF project and a narrow compatibility layer for serial I/O, timing, flash reads, and PSRAM allocation.

**Tech Stack:** ESP-IDF 5.5.5, FreeRTOS, UART0, ESP32-S3 octal PSRAM.

---

### Task 1: Add ESP-IDF project metadata

**Files:** `CMakeLists.txt`, `main/CMakeLists.txt`, `sdkconfig.defaults`, `partitions_s3_16MB_idf.csv`

Configure an ESP32-S3 target with 16 MB QIO flash, octal PSRAM, a single factory application partition, and the minimal IDF component dependencies.

### Task 2: Add IDF compatibility layer

**Files:** `src/idf_compat.hpp`, `src/inference.hpp`, `src/main.cpp`

Replace only Arduino runtime dependencies with UART0, FreeRTOS timing/yield, direct memory-mapped flash reads, heap capability APIs, and an `app_main` entry point. Keep transformer inference behavior unchanged.

### Task 3: Build and device-verify

Run `idf.py set-target esp32s3`, `idf.py build`, inspect the application size against the 0xFE0000 partition, then flash `COM3` and inspect the 115200-baud boot log for an 8 MB PSRAM allocation and successful arena setup.
