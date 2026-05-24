## 🔧 Project Setup Instructions

This project uses **Poetry** for dependency and environment management and runs inference with **llama-cpp-python** (CUDA-enabled). Follow the steps below to set up the project properly.

---

### 1. 📦 Install Dependencies

Use [Poetry](https://python-poetry.org/) to install project dependencies:

```bash
poetry install
```

Ensure that you install the **CUDA-specific version** of `llama-cpp-python` for optimal performance.

---

### 2. ⚙️ Configuration

#### Model Configuration

* Located at: `config/constants.py`

#### Generation Framework Settings

* Located at: `config/setting.py`

#### Memory Limits (for Model Caching)

Set the following parameters in `config/setting.py`:

* `VRAM_LIMIT_GB`: Limit for GPU memory usage
* `SYSTEM_RAM_LIMIT_GB`: Limit for system RAM usage

These settings optimize model caching for faster loading and unloading.

---

### 3. 🛠️ Initialize Required Scripts

Before running the main file, run the following setup scripts:

```bash
python scripts/setup_nltk.py
python scripts/init_db.py
```

---

Once all steps are complete, the environment will be ready for inference and further development.

