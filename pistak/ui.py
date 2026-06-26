from textual.containers import Vertical, Horizontal
from textual.widgets import Label, Button
from textual.app import ComposeResult

class ModelRow(Horizontal):
    def __init__(self, model_info, is_downloaded, **kwargs):
        kwargs["classes"] = kwargs.get("classes", "") + " model-row"
        super().__init__(**kwargs)
        self.model_info = model_info
        self.is_downloaded = is_downloaded

    def compose(self) -> ComposeResult:
        with Vertical(classes="model-info-col"):
            yield Label(f"[bold]{self.model_info['name']}[/bold] ({self.model_info['params']})", classes="model-title")
            yield Label(f"🖥️ RAM: {self.model_info['ram_gb']}GB  |  ⚡ Best: {', '.join(self.model_info['best_for'])}  |  📊 {self.model_info['stars_str']}", classes="model-subtitle")
        
        with Vertical(classes="model-btn-col"):
            safe_id = self.model_info['id'].replace('/', '___').replace('.', '_dot_')
            if self.is_downloaded:
                yield Button("✅ Select", id=f"btn_start_{safe_id}", variant="success")
            else:
                btn = Button("☁️ Download", id=f"btn_dl_{safe_id}", variant="primary")
                if "Incompatible" in self.model_info['stars_str']:
                    btn.disabled = True
                yield btn
