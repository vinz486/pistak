from textual.containers import Vertical
from textual.widgets import Label, Button, Rule
from textual.app import ComposeResult

class ModelRow(Vertical):
    DEFAULT_CSS = """
    ModelRow {
        height: auto;
        padding: 1 2;
    }
    ModelRow Button {
        margin-top: 1;
        margin-bottom: 1;
    }
    """
    def __init__(self, model_info, is_downloaded, **kwargs):
        super().__init__(**kwargs)
        self.styles.height = "auto"
        self.model_info = model_info
        self.is_downloaded = is_downloaded

    def compose(self) -> ComposeResult:
        yield Label(f"[bold]{self.model_info['name']}[/bold] ({self.model_info['params']})", classes="model-title")
        yield Label(f"🖥️ Min RAM: {self.model_info['ram_gb']}GB  |  ⚡ Best for: {', '.join(self.model_info['best_for'])}")
        yield Label(f"📊 Rating: {self.model_info['stars_str']}")
        
        safe_id = self.model_info['id'].replace('/', '___').replace('.', '_dot_')
        if self.is_downloaded:
            yield Button("✅ Downloaded - Select Model", id=f"btn_start_{safe_id}", variant="success")
        else:
            btn = Button("☁️ Download", id=f"btn_dl_{safe_id}", variant="primary")
            if "Incompatible" in self.model_info['stars_str']:
                btn.disabled = True
            yield btn
        
        yield Rule()
