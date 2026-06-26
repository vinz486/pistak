import os
import sys
import json
import psutil
import subprocess
import threading
import platform
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Button, Select, Input, Label, RichLog, TabbedContent, TabPane, ProgressBar, DataTable
from textual.reactive import reactive
from textual import work
from textual.worker import get_current_worker

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

    DataTable {
        height: auto;
        margin-top: 1;
        margin-bottom: 2;
        border: solid $accent;
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
    """

    BINDINGS = [
        ("q", "quit", "Esci"),
        ("s", "toggle_server", "Avvia/Ferma Server")
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
        devices = ["CPU"]
        if ov:
            try:
                core = ov.Core()
                devices = core.available_devices
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
            "ov_devices": devices,
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
            tier = "Super PC AI 🚀 (Prestazioni Top)"
        elif score >= 5:
            tier = "Ottimo PC ⚡ (Eccellente per modelli medi e NPU)"
        elif score >= 3:
            tier = "PC Discreto 💻 (Buono per modelli leggeri)"
        else:
            tier = "Ciofeca 🐢 (Farà fatica con l'AI)"
            
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
                m["stars_str"] = "❌ Non fattibile"
                
        # Sort models: highest stars first, then by intelligence (params) descending
        MODEL_CATALOG.sort(key=lambda x: (x["stars_num"], x["ram_gb"]), reverse=True)

    def compose(self) -> ComposeResult:
        self.load_settings()
        hw = self.get_hardware_info()
        available_devices = hw["ov_devices"]
        hw_score, hw_tier = self.evaluate_hardware(hw)
        self.rate_and_sort_models(hw)
        
        yield Header()
        
        with Horizontal():
            # Sidebar for settings
            with Vertical(id="sidebar"):
                yield Label("⚙️ Configurazione", classes="section-title")
                
                yield Label("Dispositivo di Calcolo", classes="setting-item")
                # Dynamically populate available OpenVINO devices
                device_options = [(d, d) for d in available_devices]
                # Fallback to config if not available right now
                if self.settings["device"] not in available_devices:
                    device_options.append((self.settings["device"], self.settings["device"]))
                
                device_select = Select(
                    device_options,
                    value=self.settings["device"],
                    id="select_device"
                )
                yield device_select
                
                yield Label("Modello Locale", classes="setting-item")
                models = self.get_local_models()
                model_value = self.settings["model_path"] if any(m[1] == self.settings["model_path"] for m in models) else None
                if model_value is None:
                    model_select = Select(models, id="select_model", allow_blank=True)
                else:
                    model_select = Select(models, value=model_value, id="select_model", allow_blank=True)
                if not models:
                    model_select.disabled = True
                yield model_select
                
                yield Label("Porta Server", classes="setting-item")
                yield Input(value=self.settings["port"], id="input_port")
                
                yield Button("Salva Configurazione", id="btn_save", variant="primary")

            # Main content area
            with Container(id="main-content"):
                with TabbedContent(initial="tab-hw"):
                    
                    with TabPane("💻 Hardware", id="tab-hw"):
                        yield Label("Analisi Hardware", classes="section-title")
                        
                        yield Label(f"Valutazione: {hw_tier} (Voto: {hw_score}/10)", classes="hw-score")
                        
                        yield Label(f"[bold]Processore (CPU):[/bold] {hw['cpu_name']}", classes="hw-info")
                        yield Label(f"[bold]Memoria RAM Totale:[/bold] {hw['ram_gb']:.1f} GB", classes="hw-info")
                        yield Label(f"[bold]Sistema Operativo:[/bold] {hw['os_name']}", classes="hw-info")
                        yield Label(f"[bold]Acceleratori OpenVINO:[/bold] {', '.join(available_devices)}", classes="hw-info")
                        
                        yield Label("\n[bold]Guida all'uso degli acceleratori:[/bold]")
                        yield Label("• [bold green]NPU[/bold green]: Perfetta per modelli fino a 4B parametri. Bassissimo consumo di batteria, ideale in background.")
                        yield Label("• [bold blue]GPU[/bold blue]: Altissime prestazioni, ottima per modelli 7B-8B se hai almeno 16GB di RAM.")
                        yield Label("• [bold magenta]CPU[/bold magenta]: Soluzione di ripiego, universale ma tendenzialmente più lenta.")

                    with TabPane("🚀 Server", id="tab-server"):
                        yield Label("Controllo Server OpenAI Compatibile", classes="section-title")
                        with Horizontal(classes="setting-item"):
                            yield Button("Avvia Server", id="btn_toggle_server", variant="success")
                            yield Label("  Stato: Fermo", id="lbl_server_status")
                        yield RichLog(id="log_server", highlight=True, markup=True)
                        
                    with TabPane("📥 Modelli", id="tab-models"):
                        yield Label("Modelli Suggeriti ed Ordinati per il tuo PC", classes="section-title")
                        
                        yield DataTable(id="models_table")
                        
                        yield Horizontal(
                            Input(placeholder="Seleziona un modello dalla tabella o scrivi il Repo ID...", id="input_hf_repo"),
                            Button("Scarica", id="btn_download", variant="primary")
                        )
                        yield RichLog(id="log_download")
                        
                    with TabPane("📊 Statistiche", id="tab-stats"):
                        with Vertical(id="stat-container"):
                            yield Label("Monitoraggio del Carico di Sistema", classes="section-title")
                            yield Label("Utilizzo CPU:", classes="stat-label")
                            yield ProgressBar(id="pb_cpu", total=100, show_eta=False)
                            yield Label("Utilizzo RAM:", classes="stat-label")
                            yield ProgressBar(id="pb_ram", total=100, show_eta=False)

        yield Footer()

    def on_mount(self):
        # Populate the dynamic hardware table
        hw = self.get_hardware_info()
        table = self.query_one("#models_table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Modello", "Intelligenza", "Min RAM", "Voto (Hardware Attuale)")
        
        for m in MODEL_CATALOG:
            table.add_row(
                m["name"], 
                m["params"], 
                f"{m['ram_gb']} GB", 
                m["stars_str"],
                key=m["id"]
            )

        self.update_timer = self.set_interval(1.0, self.update_stats)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        repo_id = event.row_key.value
        if repo_id:
            inp = self.query_one("#input_hf_repo", Input)
            inp.value = repo_id
            self.notify(f"Selezionato {repo_id}. Clicca Scarica quando sei pronto.")

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

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "select_device":
            self.settings["device"] = event.value
        elif event.select.id == "select_model":
            self.settings["model_path"] = event.value

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "input_port":
            self.settings["port"] = event.value

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_save":
            self.save_settings()
            self.notify("Configurazione salvata in background!")
            
        elif event.button.id == "btn_toggle_server":
            self.action_toggle_server()
            
        elif event.button.id == "btn_download":
            repo_id = self.query_one("#input_hf_repo", Input).value
            if repo_id:
                self.download_model(repo_id)

    def action_toggle_server(self) -> None:
        if self.server_running:
            self.stop_server()
        else:
            self.start_server()

    def start_server(self):
        self.save_settings()
        
        if not self.settings.get("model_path"):
            self.notify("Per favore seleziona un modello prima di avviare!", severity="error")
            return
            
        btn = self.query_one("#btn_toggle_server", Button)
        lbl = self.query_one("#lbl_server_status", Label)
        log = self.query_one("#log_server", RichLog)
        
        port = self.settings.get("port", "8000")
        device = self.settings.get("device", "CPU")
        model = self.settings.get("model_path")
        
        log.write(f"[bold green]Avviando il Server sulla porta {port}...[/]")
        log.write(f"Modello: {model}")
        log.write(f"Acceleratore: {device}")
        
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
            btn.label = "Ferma Server"
            btn.variant = "error"
            lbl.update("  Stato: [bold green]In Esecuzione[/]")
            
            # Start a thread to read logs
            threading.Thread(target=self.read_server_logs, daemon=True).start()
            
        except Exception as e:
            log.write(f"[bold red]Errore avvio server:[/] {e}")

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
            
            btn.label = "Avvia Server"
            btn.variant = "success"
            lbl.update("  Stato: [bold red]Fermo[/]")
            log.write("[bold red]Server arrestato.[/]")
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
            self.call_from_thread(self.write_download_log, "[bold red]huggingface_hub non è installato.[/]")
            return
            
        self.call_from_thread(self.write_download_log, f"Inizio download di: [bold]{repo_id}[/]")
        model_name = repo_id.split("/")[-1]
        target_dir = os.path.join(MODELS_DIR, model_name)
        
        try:
            snapshot_download(repo_id=repo_id, local_dir=target_dir)
            self.call_from_thread(self.write_download_log, f"[bold green]Download completato![/] Salvato in {target_dir}")
            self.call_from_thread(self.refresh_models)
        except Exception as e:
            self.call_from_thread(self.write_download_log, f"[bold red]Download fallito:[/] {e}")

    def write_download_log(self, text: str):
        try:
            log = self.query_one("#log_download", RichLog)
            log.write(text)
        except Exception:
            pass

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
