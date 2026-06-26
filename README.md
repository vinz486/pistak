# Pistak 🚀

Pistak is a TUI (Terminal User Interface) application designed to easily run an OpenAI-compatible API server using OpenVINO. It dynamically analyzes your hardware and suggests the best Hugging Face models tailored to your system's capabilities (NPU, GPU, CPU).

## Features
- **OpenAI Compatible API:** Exposes an API over localhost, allowing you to plug it into any standard OpenAI-compatible client.
- **Dynamic Hardware Analysis:** Analyzes your installed RAM and Intel OpenVINO Accelerators (NPU, GPU, CPU) to assess your system.
- **Smart Model Recommendations:** Ranks and scores popular models (like Llama 3, Phi-3, Qwen) based on your specific hardware configuration.
- **One-Click Downloading:** Select a model from the table and download it directly from the Hugging Face Hub inside the TUI.
- **Hardware Telemetry:** Monitor CPU and RAM usage directly from the TUI.
- **Single Executable:** Can be packaged into a single binary for portable usage without the need to configure Python environments manually.

## How to Build & Run
Run the build script to automatically create a virtual environment, install all required dependencies, and package the application using PyInstaller:
```bash
./build.sh
```

Once built, execute the self-contained binary:
```bash
./dist/pistak
```

## Architecture
Pistak is built using **Textual** for the interactive terminal user interface and **OpenVINO GenAI** for highly efficient local LLM inference, fully utilizing Intel's latest hardware such as NPUs and integrated Arc GPUs.
