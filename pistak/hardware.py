import os
import platform
import psutil

try:
    import openvino as ov
except ImportError:
    ov = None

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

class HardwareEvaluator:
    def __init__(self, catalog=None):
        self.catalog = catalog if catalog is not None else [dict(m) for m in MODEL_CATALOG]

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

    def rate_and_sort_models(self, hw, selected_device=None):
        ram = hw["ram_gb"]
        devices = hw["ov_devices"]
        
        for m in self.catalog:
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
                    
            sort_score = stars
            if stars > 0 and selected_device and selected_device in m["best_for"]:
                sort_score += 2  # Boost to top if it perfectly matches the selected device
            
            m["stars_num"] = stars
            if stars > 0:
                m["stars_str"] = "⭐" * stars + "☆" * (5 - stars)
            else:
                m["stars_str"] = "❌ Incompatible"
                
        # Sort models: highest sort_score first, then by intelligence (params) descending
        self.catalog.sort(key=lambda x: (x.get("sort_score", x["stars_num"]), x["ram_gb"]), reverse=True)
        return self.catalog
