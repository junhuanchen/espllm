idf_component_register(
    SRCS "../src/main.cpp"
    INCLUDE_DIRS "../src"
    REQUIRES esp_driver_usb_serial_jtag esp_timer esp_psram esp_hw_support
)
