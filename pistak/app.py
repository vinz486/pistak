import os
import sys
import psutil
import subprocess
import threading

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Header, Footer, Button, Select, Input, Label, RichLog, TabbedContent, TabPane, ProgressBar, RadioSet, RadioButton, Markdown, Rule
from textual.reactive import reactive
from textual import work

from pistak.config import ConfigManager
from pistak.hardware import HardwareEvaluator
from pistak.ui import ModelRow

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

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
        border-right: solid $primary-background;
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
        width: 1fr;
    }
    
    #main-content {
        padding: 1 2;
    }
    
    #download_monitor {
        height: auto;
        max-height: 10;
        dock: bottom;
        background: $surface;
        border-top: solid $accent;
        padding: 1;
    }
    
    #log_download {
        height: auto;
        min-height: 2;
        max-height: 6;
    }
    
    RichLog {
        background: $boost;
        border: solid $accent;
        height: 1fr;
        margin-top: 1;
    }

    #stat-container {
        padding: 2 4;
        height: auto;
    }

    .stat-label {
        text-align: center;
        padding: 1 0 0 0;
        text-style: bold;
    }
    
    .stat-value {
        text-align: center;
        color: $success;
        text-style: bold;
        margin-bottom: 1;
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
        padding: 1 1;
        margin-bottom: 1;
        border-bottom: solid $primary-background;
        overflow: hidden;
    }
    
    .model-info-col {
        width: 1fr;
        height: auto;
        overflow: hidden;
    }
    
    .model-btn-col {
        width: 20;
        height: auto;
        align: right middle;
    }
    
    .model-row Button {
        margin: 0;
        min-width: 16;
    }
    
    .model-title {
        text-style: bold;
        color: $success;
        margin-top: 1;
    }
    
    .model-subtitle {
        color: $text-muted;
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

    server_running = reactive(False)
    cpu_percent = reactive(0.0)
    ram_percent = reactive(0.0)
    
    total_tokens_in = 0
    total_tokens_out = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.server_process = None
        self.config_manager = ConfigManager()
        self.hardware_evaluator = HardwareEvaluator()
        # Add a convenience reference to settings
        self.settings = self.config_manager.settings

    def compose(self) -> ComposeResult:
        self.config_manager.load_settings()
        hw = self.hardware_evaluator.get_hardware_info()
        hw_score, hw_tier = self.hardware_evaluator.evaluate_hardware(hw)
        self.hardware_evaluator.rate_and_sort_models(hw)
        
        yield Header()
        
        with Horizontal():
            # Sidebar for settings
            with VerticalScroll(id="sidebar"):
                yield Label("⚙️ Configuration", classes="section-title")
                
                yield Label("Target Device", classes="setting-item")
                # Dynamically populate using RadioButtons based on detected hardware
                with RadioSet(id="radio_device"):
                    for d in hw["ui_devices"]:
                        yield RadioButton(d, value=(d == self.settings.get("device", "CPU")))
                
                yield Label("Local Model", classes="setting-item")
                models = self.config_manager.get_local_models()
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
                
                yield Label("🔑 Hugging Face", classes="section-title")
                yield Label("Token for gated models (like Llama 3). Get it at:\nhuggingface.co/settings/tokens", classes="setting-item")
                hf_token_val = self.settings.get("hf_token", "")
                yield Input(value=hf_token_val, placeholder="hf_...", password=True, id="input_hf_token", classes="setting-item")
                login_btn = Button("Login", id="btn_hf_login", variant="primary")
                if hf_token_val:
                    login_btn.label = "Logged In ✅"
                    login_btn.variant = "success"
                yield login_btn

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
                            
                        with Vertical(id="download_monitor"):
                            yield Label("Download Status: Ready", id="lbl_download_status", classes="section-title")
                            yield ProgressBar(id="pb_download", total=100, show_eta=False)
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
                            
                            yield Label("Active Accelerator:", classes="stat-label")
                            yield Label("Offline", id="lbl_stat_device", classes="stat-value")
                            yield Label("Tokens Processed (Session):", classes="stat-label")
                            yield Label("In: 0  |  Out: 0", id="lbl_stat_tokens", classes="stat-value")
                            yield Label("Generation Speed:", classes="stat-label")
                            yield Label("0.00 TPS", id="lbl_stat_tps", classes="stat-value")
                            
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
                
            downloaded_models = [m[0] for m in self.config_manager.get_local_models()]
            hw = self.hardware_evaluator.get_hardware_info()
            selected_dev = self.settings.get("device", "CPU")
            catalog = self.hardware_evaluator.rate_and_sort_models(hw, selected_device=selected_dev)
            
            for m in catalog:
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
            self.refresh_model_list()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "select_model":
            self.settings["model_path"] = event.value

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "input_port":
            self.settings["port"] = event.value
        elif event.input.id == "input_hf_token":
            self.settings["hf_token"] = event.value

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_save":
            self.config_manager.save_settings()
            self.notify("Configuration saved transparently!")
            
        elif event.button.id == "btn_hf_login":
            token = self.settings.get("hf_token", "").strip()
            if not token:
                self.notify("Please enter a valid token!", severity="error")
                return
            try:
                from huggingface_hub import login
                login(token=token)
                self.config_manager.save_settings()
                self.notify("Successfully logged in to Hugging Face!", title="Login Success", severity="information")
                event.button.label = "Logged In ✅"
                event.button.variant = "success"
            except Exception as e:
                self.notify(f"Login failed: {e}", severity="error")
            
        elif event.button.id == "btn_toggle_server":
            self.action_toggle_server()
            
        elif event.button.id and event.button.id.startswith("btn_start_"):
            # Handle selecting a downloaded model
            repo_id = event.button.id.replace("btn_start_", "").replace("___", "/").replace("_dot_", ".")
            model_name = repo_id.split("/")[-1]
            local_path = os.path.join(self.config_manager.models_dir, model_name)
            
            self.settings["model_path"] = local_path
            try:
                model_select = self.query_one("#select_model", Select)
                if local_path in [m[1] for m in self.config_manager.get_local_models()]:
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
        self.config_manager.save_settings()
        
        if not self.settings.get("model_path"):
            self.notify("Please select a model first!", severity="error")
            return
            
        btn = self.query_one("#btn_toggle_server", Button)
        lbl = self.query_one("#lbl_server_status", Label)
        log = self.query_one("#log_server", RichLog)
        
        port = self.settings.get("port", "1234")
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
            import os
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            
            self.server_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
            )
            self.server_running = True
            btn.label = "Stop Server"
            btn.variant = "error"
            lbl.update("  Status: [bold yellow]Starting (Loading Model...)[/]")
            
            try:
                self.total_tokens_in = 0
                self.total_tokens_out = 0
                self.query_one("#lbl_stat_device", Label).update(f"[bold blue]{device}[/]")
                self.query_one("#lbl_stat_tokens", Label).update("In: 0  |  Out: 0")
                self.query_one("#lbl_stat_tps", Label).update("0.00 TPS")
            except Exception:
                pass
            
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
            
            try:
                self.query_one("#lbl_stat_device", Label).update("Offline")
            except Exception:
                pass
        except Exception:
            pass

    def read_server_logs(self):
        try:
            if self.server_process and self.server_process.stdout:
                for line in iter(self.server_process.stdout.readline, ''):
                    self.call_from_thread(self.write_server_log, line.strip())
        except Exception:
            pass

    def write_server_log(self, text: str):
        try:
            if "[METRICS]" in text:
                import re
                m_in = re.search(r"TOKENS_IN:(\d+)", text)
                m_out = re.search(r"TOKENS_OUT:(\d+)", text)
                m_tps = re.search(r"TPS:([0-9\.]+)", text)
                
                if m_in and m_out:
                    self.total_tokens_in += int(m_in.group(1))
                    self.total_tokens_out += int(m_out.group(1))
                    lbl_tok = self.query_one("#lbl_stat_tokens", Label)
                    lbl_tok.update(f"In: [bold]{self.total_tokens_in}[/bold]  |  Out: [bold]{self.total_tokens_out}[/bold]")
                    
                if m_tps:
                    lbl_tps = self.query_one("#lbl_stat_tps", Label)
                    lbl_tps.update(f"[bold yellow]{m_tps.group(1)}[/bold yellow] TPS")
                    
                return
                
            log = self.query_one("#log_server", RichLog)
            log.write(text)
            
            # Update status to Running when uvicorn is ready
            if "Application startup complete." in text or "Uvicorn running on" in text:
                lbl = self.query_one("#lbl_server_status", Label)
                lbl.update("  Status: [bold green]Ready & Running[/]")
        except Exception:
            pass

    def write_download_log(self, text: str):
        try:
            log = self.query_one("#log_download", RichLog)
            log.write(text)
        except Exception:
            pass

    def update_download_status(self, text: str):
        try:
            lbl = self.query_one("#lbl_download_status", Label)
            lbl.update(f"[bold blue]Downloading:[/] {text}")
            
            import re
            m = re.search(r'(\d+)%\|', text)
            if m:
                pb = self.query_one("#pb_download", ProgressBar)
                pb.update(progress=int(m.group(1)))
        except Exception:
            pass

    @work(thread=True)
    def download_model(self, repo_id: str):
        if not snapshot_download:
            self.call_from_thread(self.write_download_log, "[bold red]huggingface_hub is not installed.[/]")
            return
            
        self.call_from_thread(self.write_download_log, f"Starting download for: [bold]{repo_id}[/]")
        model_name = repo_id.split("/")[-1]
        target_dir = os.path.join(self.config_manager.models_dir, model_name)
        
        # Setup stderr capture for tqdm
        import sys, re
        old_stderr = sys.stderr
        
        class TqdmCapture:
            def __init__(self, app):
                self.app = app
                self.buffer = ""
            def write(self, s):
                clean_s = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', s)
                self.buffer += clean_s
                if '\r' in self.buffer or '\n' in self.buffer:
                    lines = self.buffer.replace('\r', '\n').split('\n')
                    for line in lines[:-1]:
                        text = line.strip()
                        if text:
                            self.app.call_from_thread(self.app.update_download_status, text)
                            if "100%" in text or "Download" in text:
                                self.app.call_from_thread(self.app.write_download_log, text)
                    self.buffer = lines[-1]
                old_stderr.write(s)
            def flush(self):
                old_stderr.flush()
                
        sys.stderr = TqdmCapture(self)
        
        try:
            snapshot_download(repo_id=repo_id, local_dir=target_dir)
            self.call_from_thread(self.write_download_log, f"[bold green]Download completed![/] Saved to {target_dir}")
            try:
                self.call_from_thread(self.query_one("#pb_download", ProgressBar).update, progress=100)
            except Exception:
                pass
            
            # Update data table local status immediately
            self.call_from_thread(self.refresh_models)
            self.call_from_thread(self.refresh_model_list)
        except Exception as e:
            self.call_from_thread(self.write_download_log, f"[bold red]Download failed:[/] {e}")
        finally:
            sys.stderr = old_stderr

    def refresh_models(self):
        try:
            model_select = self.query_one("#select_model", Select)
            models = self.config_manager.get_local_models()
            model_select.set_options(models)
            if models:
                model_select.disabled = False
        except Exception:
            pass

    def on_unmount(self):
        self.stop_server()
