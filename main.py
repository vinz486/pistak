import os
import sys
import json
import psutil
import subprocess
import threading
import platform
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Header, Footer, Button, Select, Input, Label, RichLog, TabbedContent, TabPane, ProgressBar, RadioSet, RadioButton, Markdown
from textual.reactive import reactive
from textual import work

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

try:
    import openvino as ov
except ImportError:
    ov = None

# PyInstaller / Multiprocessing support wrapper
if "--run-server" in sys.argv:
    import server
    sys.argv.remove("--run-server")
    server.main()
    sys.exit(0)

CONFIG_FILE = "settings.json"
MODELS_DIR = "models"

HELP_MD = """
# Troubleshooting & Guides

## 1. NPU Not Showing Up?
If your Intel NPU (e.g., on Meteor/Lunar Lake) isn't appearing in the hardware list, it's likely a missing driver or permission issue on Linux.

**Step 1: Check Permissions**
Your user must be in the `render` group to access AI accelerators. Run this in your terminal:
```bash
sudo usermod -aG render $USER
```
*(Reboot your PC afterward for it to take effect).*

**Step 2: Install Intel NPU Drivers**
Ubuntu does not pre-install the Intel Level Zero NPU drivers. You can install them by running:
```bash
mkdir -p /tmp/npu_driver
cd /tmp/npu_driver
wget https://github.com/intel/linux-npu-driver/releases/download/v1.33.0/linux-npu-driver-v1.33.0.20260529-26625960453-ubuntu2404.tar.gz
tar -xzf linux-npu-driver-*.tar.gz
sudo apt install -y ./*.deb
```

## 2. Performance Tips
* Models with **INT4** quantization use vastly less RAM and are highly optimized for OpenVINO.
* When running 7B+ models on a **GPU**, ensure you have at least 16GB of system RAM since the iGPU shares system memory.
* **CPU** fallback is universally compatible but will drain battery faster and run slower.
"""

MODEL_CATALOG = [
    {
        "id": "OpenVINO/Llama-3-8B-Instruct-int4-ov",
        "name": "Llama 3 (8B) INT4",
        "params": "8B",
        "ram_gb": 6.5,
        "best_for": ["GPU", "CPU"]
    },
    {
        "id": "OpenVINO/Qwen2.5-7B-Instruct-int4-ov",
        "name": "Qwen 2.5 (7B) INT4",
        "params": "7B",
        "ram_gb": 6.0,
        "best_for": ["GPU", "CPU"]
    },
    {
        "id": "OpenVINO/Mistral-7B-Instruct-v0.2-int4-ov",
        "name": "Mistral v0.2 (7B) INT4",
        "params": "7B",
        "ram_gb": 6.0,
        "best_for": ["GPU", "CPU"]
    },
    {
        "id": "OpenVINO/Phi-3-mini-4k-instruct-int4-ov",
        "name": "Phi-3 Mini (3.8B) INT4",
        "params": "3.8B",
        "ram_gb": 3.0,
        "best_for": ["NPU", "GPU", "CPU"]
    },
    {
        "id": "OpenVINO/Qwen2-1.5B-Instruct-int4-ov",
        "name": "Qwen 2 (1.5B) INT4",
        "params": "1.5B",
        "ram_gb": 1.5,
        "best_for": ["CPU", "NPU"]
    }
]

class ModelRow(Container):
    def __init__(self, model_info, is_downloaded, **kwargs):
        super().__init__(**kwargs)
        self.model_info = model_info
        self.is_downloaded = is_downloaded

    def compose(self) -> ComposeResult:
        with Horizontal(classes="model-row"):
            with Vertical(classes="model-details"):
                yield Label(f"[bold]{self.model_info['name']}[/bold] ({self.model_info['params']})", classes="model-title")
                yield Label(f"🖥️ Min RAM: {self.model_info['ram_gb']}GB  |  ⚡ Best for: {', '.join(self.model_info['best_for'])}")
                yield Label(f"📊 Rating: {self.model_info['stars_str']}")
            
            with Vertical(classes="model-actions"):
                safe_id = self.model_info['id'].replace('/', '___').replace('.', '_dot_')
                if self.is_downloaded:
                    yield Label("[bold green]✅ Downloaded[/bold green]")
                    yield Button("Select Model", id=f"btn_start_{safe_id}", variant="success")
                else:
                    yield Label("[bold blue]☁️ Cloud[/bold blue]")
                    btn = Button("Download", id=f"btn_dl_{safe_id}", variant="primary")
                    if "Incompatible" in self.model_info['stars_str']:
                        btn.disabled = True
                    yield btn


class PistakApp(App):
    CSS = """
    Screen {
        background: $surface;
    }
    
    #sidebar {
        width: 35;
        dock: left;
        padding: 1 2;
        background: $panel;
        border-right: vkey $background;
    }
    
    .section-title {
        text-style: bold;
        padding-top: 1;
        padding-bottom: 1;
        color: $accent;
        border-bottom: solid $accent;
        margin-bottom: 1;
    }
    
    .setting-item {
        margin-top: 1;
        margin-bottom: 1;
    }
    
    #btn_save {
        margin-top: 2;
        width: 100%;
    }
    
    #main-content {
        padding: 1 2;
    }
    
    RichLog {
        background: $boost;
        border: solid $accent;
        height: 1fr;
        margin-top: 1;
    }

    #stat-container {
        align: center middle;
        height: 100%;
    }

    .stat-label {
        text-align: center;
        padding: 1;
        text-style: bold;
    }

    .hw-info {
        padding: 1;
        margin-bottom: 1;
        background: $boost;
        border-left: thick $accent;
    }
    
    .hw-score {
        padding: 1;
        margin-bottom: 1;
        background: $primary-background;
        border: solid $success;
        text-style: bold;
    }

    /* Model List Styling */
    .model-row {
        height: auto;
        padding: 1 2;
        margin-bottom: 1;
        background: $boost;
        border: solid $accent;
    }
    
    .model-details {
        width: 1fr;
    }
    
    .model-title {
        text-style: bold;
        color: $success;
        margin-bottom: 1;
    }

    .model-actions {
        width: 25;
        align: center middle;
    }
    
    .model-actions Button {
        width: 100%;
        margin-top: 1;
    }
    
    /* Radio Button Fix */
    RadioSet {
        border: none;
        background: transparent;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("s", "toggle_server", "Start/Stop Server")
    ]

    # App State
    server_process = None
    settings = {
        "model_path": "",
        "device": "CPU",
        "port": "8000"
    }
    
    server_running = reactive(False)
    cpu_percent = reactive(0.0)
    ram_percent = reactive(0.0)

    def load_settings(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    self.settings.update(json.load(f))
            except Exception:
                pass
        
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)

    def save_settings(self):
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.settings, f, indent=4)

    def get_local_models(self):
        if not os.path.exists(MODELS_DIR):
            return []
        return [(d, os.path.join(MODELS_DIR, d)) for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]

    def get_hardware_info(self):
        ram_gb = psutil.virtual_memory().total / (1024**3)
        ov_devices = ["CPU"]
        if ov:
            try:
                core = ov.Core()
                ov_devices = core.available_devices
            except Exception:
                pass
                
        cpu_name = platform.processor()
        if os.path.exists("/proc/cpuinfo"):
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_name = line.split(":")[1].strip()
                            break
            except Exception:
                pass

        return {
            "ram_gb": ram_gb,
            "ov_devices": ov_devices,
            "ui_devices": sorted(list(set(ov_devices))), # Only show what OpenVINO detects dynamically
            "cpu_name": cpu_name,
            "os_name": f"{platform.system()} {platform.release()}"
        }

    def evaluate_hardware(self, hw):
        score = 0
        ram = hw["ram_gb"]
        devices = hw["ov_devices"]
        
        if ram >= 31: score += 4
        elif ram >= 15: score += 3
        elif ram >= 7: score += 1
        
        if "NPU" in devices: score += 3
        if "GPU" in devices: score += 2
        
        if score >= 8:
            tier = "Super PC AI 🚀 (Top Performance)"
        elif score >= 5:
            tier = "Great PC ⚡ (Excellent for medium models & NPU)"
        elif score >= 3:
            tier = "Decent PC 💻 (Good for lightweight models)"
        else:
            tier = "Potato PC 🐢 (Will struggle with AI)"
            
        return score, tier

    def rate_and_sort_models(self, hw):
        ram = hw["ram_gb"]
        devices = hw["ov_devices"]
        
        for m in MODEL_CATALOG:
            stars = 0
            # Require at least 500MB free RAM beyond model requirements
            if ram > m["ram_gb"] + 0.5:
                if "7B" in m["params"] or "8B" in m["params"]:
                    # Very smart models. 5 stars if we have a GPU and 16GB+ RAM
                    if "GPU" in devices and ram >= 15:
                        stars = 5
                    else:
                        stars = 3 # Can run on CPU but slowly
                elif "3.8B" in m["params"]:
                    # Phi-3: 5 stars if NPU is present, 4 otherwise (very fast)
                    stars = 5 if "NPU" in devices else 4
                elif "1.5B" in m["params"]:
                    # Small models
                    stars = 3
            
            m["stars_num"] = stars
            if stars > 0:
                m["stars_str"] = "⭐" * stars + "☆" * (5 - stars)
            else:
                m["stars_str"] = "❌ Incompatible"
                
        # Sort models: highest stars first, then by intelligence (params) descending
        MODEL_CATALOG.sort(key=lambda x: (x["stars_num"], x["ram_gb"]), reverse=True)

    def compose(self) -> ComposeResult:
        self.load_settings()
        hw = self.get_hardware_info()
        hw_score, hw_tier = self.evaluate_hardware(hw)
        self.rate_and_sort_models(hw)
        
        yield Header()
        
        with Horizontal():
            # Sidebar for settings
            with Vertical(id="sidebar"):
                yield Label("⚙️ Configuration", classes="section-title")
                
                yield Label("Target Device", classes="setting-item")
                # Dynamically populate using RadioButtons based on detected hardware
                with RadioSet(id="radio_device"):
                    for d in hw["ui_devices"]:
                        yield RadioButton(d, value=(d == self.settings.get("device", "CPU")))
                
                yield Label("Local Model", classes="setting-item")
                models = self.get_local_models()
                model_value = self.settings["model_path"] if any(m[1] == self.settings["model_path"] for m in models) else None
                if model_value is None:
                    model_select = Select(models, id="select_model", allow_blank=True)
                else:
                    model_select = Select(models, value=model_value, id="select_model", allow_blank=True)
                if not models:
                    model_select.disabled = True
                yield model_select
                
                yield Label("Server Port", classes="setting-item")
                yield Input(value=self.settings["port"], id="input_port")
                
                yield Button("Save Configuration", id="btn_save", variant="primary")

            # Main content area
            with Container(id="main-content"):
                with TabbedContent(initial="tab-hw"):
                    
                    with TabPane("💻 Hardware", id="tab-hw"):
                        yield Label("Hardware Overview", classes="section-title")
                        
                        yield Label(f"Rating: {hw_tier} (Score: {hw_score}/10)", classes="hw-score")
                        
                        # Compact hardware display
                        yield Label(f"🖥️ [bold]CPU:[/bold] {hw['cpu_name']}  |  🐏 [bold]RAM:[/bold] {hw['ram_gb']:.1f} GB  |  💽 [bold]OS:[/bold] {hw['os_name']}", classes="hw-info")
                        yield Label(f"⚡ [bold]OpenVINO Accelerators:[/bold] {', '.join(hw['ov_devices'])}", classes="hw-info")
                        
                        yield Label("\n[bold]Accelerator Usage Guide:[/bold]")
                        yield Label("• [bold green]NPU[/bold green]: Perfect for models up to 4B parameters. High battery efficiency, ideal for background tasks.")
                        yield Label("• [bold blue]GPU[/bold blue]: Highest performance, excellent for 7B-8B models if you have at least 16GB RAM.")
                        yield Label("• [bold magenta]CPU[/bold magenta]: Fallback option, universal but generally slower.")

                    with TabPane("📥 Models", id="tab-models"):
                        yield Label("Suggested Models Ranked for Your PC", classes="section-title")
                        yield Label("💡 [italic]Click Download to fetch a model. If already downloaded, click Select Model.[/italic]", classes="setting-item")
                        
                        with VerticalScroll(id="models_list_container"):
                            # Populated in on_mount
                            pass
                            
                        yield RichLog(id="log_download")

                    with TabPane("🚀 Server", id="tab-server"):
                        yield Label("OpenAI Compatible Server Control", classes="section-title")
                        with Horizontal(classes="setting-item"):
                            yield Button("Start Server", id="btn_toggle_server", variant="success")
                            yield Label("  Status: Stopped", id="lbl_server_status")
                        yield RichLog(id="log_server", highlight=True, markup=True)
                        
                    with TabPane("📊 Statistics", id="tab-stats"):
                        with Vertical(id="stat-container"):
                            yield Label("System Load Monitoring", classes="section-title")
                            yield Label("CPU Usage:", classes="stat-label")
                            yield ProgressBar(id="pb_cpu", total=100, show_eta=False)
                            yield Label("RAM Usage:", classes="stat-label")
                            yield ProgressBar(id="pb_ram", total=100, show_eta=False)
                            
                    with TabPane("🆘 Help", id="tab-help"):
                        yield Markdown(HELP_MD)

        yield Footer()

    def on_mount(self):
        self.refresh_model_list()
        self.update_timer = self.set_interval(1.0, self.update_stats)

    def refresh_model_list(self):
        try:
            container = self.query_one("#models_list_container")
            # Clear existing children
            for child in container.children:
                child.remove()
                
            downloaded_models = [m[0] for m in self.get_local_models()]
            hw = self.get_hardware_info()
            self.rate_and_sort_models(hw)
            
            for m in MODEL_CATALOG:
                model_dir_name = m["id"].split("/")[-1]
                is_downloaded = model_dir_name in downloaded_models
                container.mount(ModelRow(m, is_downloaded))
        except Exception as e:
            pass

    def update_stats(self):
        self.cpu_percent = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        self.ram_percent = mem.percent
        
        # Update progress bars
        try:
            self.query_one("#pb_cpu", ProgressBar).update(progress=self.cpu_percent)
            self.query_one("#pb_ram", ProgressBar).update(progress=self.ram_percent)
        except Exception:
            pass

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id == "radio_device":
            self.settings["device"] = str(event.pressed.label)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "select_model":
            self.settings["model_path"] = event.value

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "input_port":
            self.settings["port"] = event.value

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_save":
            self.save_settings()
            self.notify("Configuration saved transparently!")
            
        elif event.button.id == "btn_toggle_server":
            self.action_toggle_server()
            
        elif event.button.id and event.button.id.startswith("btn_start_"):
            # Handle selecting a downloaded model
            repo_id = event.button.id.replace("btn_start_", "").replace("___", "/").replace("_dot_", ".")
            model_name = repo_id.split("/")[-1]
            local_path = os.path.join(MODELS_DIR, model_name)
            
            self.settings["model_path"] = local_path
            try:
                model_select = self.query_one("#select_model", Select)
                if local_path in [m[1] for m in self.get_local_models()]:
                    model_select.value = local_path
            except Exception:
                pass
                
            try:
                tabs = self.query_one(TabbedContent)
                tabs.active = "tab-server"
                self.notify(f"Selected {model_name}. Click 'Start Server' to run it!", title="Ready to Run")
            except Exception:
                pass
                
        elif event.button.id and event.button.id.startswith("btn_dl_"):
            # Handle download button
            repo_id = event.button.id.replace("btn_dl_", "").replace("___", "/").replace("_dot_", ".")
            self.download_model(repo_id)

    def action_toggle_server(self) -> None:
        if self.server_running:
            self.stop_server()
        else:
            self.start_server()

    def start_server(self):
        self.save_settings()
        
        if not self.settings.get("model_path"):
            self.notify("Please select a model first!", severity="error")
            return
            
        btn = self.query_one("#btn_toggle_server", Button)
        lbl = self.query_one("#lbl_server_status", Label)
        log = self.query_one("#log_server", RichLog)
        
        port = self.settings.get("port", "8000")
        device = self.settings.get("device", "CPU")
        model = self.settings.get("model_path")
        
        log.write(f"[bold green]Starting Server on port {port}...[/]")
        log.write(f"Model: {model}")
        log.write(f"Accelerator: {device}")
        
        if getattr(sys, 'frozen', False):
            # If running as a PyInstaller executable
            cmd = [
                sys.executable, "--run-server",
                "--model-path", model,
                "--device", device,
                "--port", str(port)
            ]
        else:
            # If running as a standard python script
            cmd = [
                sys.executable, "server.py",
                "--model-path", model,
                "--device", device,
                "--port", str(port)
            ]
        
        try:
            self.server_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            self.server_running = True
            btn.label = "Stop Server"
            btn.variant = "error"
            lbl.update("  Status: [bold green]Running[/]")
            
            # Start a thread to read logs
            threading.Thread(target=self.read_server_logs, daemon=True).start()
            
        except Exception as e:
            log.write(f"[bold red]Server start error:[/] {e}")

    def stop_server(self):
        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None
            
        self.server_running = False
        
        try:
            btn = self.query_one("#btn_toggle_server", Button)
            lbl = self.query_one("#lbl_server_status", Label)
            log = self.query_one("#log_server", RichLog)
            
            btn.label = "Start Server"
            btn.variant = "success"
            lbl.update("  Status: [bold red]Stopped[/]")
            log.write("[bold red]Server stopped.[/]")
        except Exception:
            pass

    def read_server_logs(self):
        try:
            for line in iter(self.server_process.stdout.readline, ''):
                self.call_from_thread(self.write_server_log, line.strip())
        except Exception:
            pass

    def write_server_log(self, text: str):
        try:
            log = self.query_one("#log_server", RichLog)
            log.write(text)
        except Exception:
            pass

    @work(thread=True)
    def download_model(self, repo_id: str):
        if not snapshot_download:
            self.call_from_thread(self.write_download_log, "[bold red]huggingface_hub is not installed.[/]")
            return
            
        self.call_from_thread(self.write_download_log, f"Starting download for: [bold]{repo_id}[/]")
        model_name = repo_id.split("/")[-1]
        target_dir = os.path.join(MODELS_DIR, model_name)
        
        try:
            snapshot_download(repo_id=repo_id, local_dir=target_dir)
            self.call_from_thread(self.write_download_log, f"[bold green]Download completed![/] Saved to {target_dir}")
            
            # Update data table local status immediately
            self.call_from_thread(self.refresh_models)
            self.call_from_thread(self.refresh_model_list)
        except Exception as e:
            self.call_from_thread(self.write_download_log, f"[bold red]Download failed:[/] {e}")

    def refresh_models(self):
        try:
            model_select = self.query_one("#select_model", Select)
            models = self.get_local_models()
            model_select.set_options(models)
            if models:
                model_select.disabled = False
        except Exception:
            pass

    def on_unmount(self):
        self.stop_server()

if __name__ == "__main__":
    app = PistakApp()
    app.run()
