---

### ⚠️ Review-Only Notice

This repository is provided **solely for peer review purposes**.

- The code is **not licensed for commercial or public use**.
- Redistribution, reuse, or publication of the code or data is **not permitted**.
- For questions, contact the author at [seyedeh.ebrahimi@tuni.fi](mailto:seyedeh.ebrahimi@tuni.fi).

If you're a reviewer, thank you for evaluating this work!

---


# 1. Clone the repository
```bash
git clone https://github.com/seyedeh-mona-ebrahimi/Voices-Between-Lines.git
```

# 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

# 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

# 4. Install system dependencies
```bash
sudo apt-get install voikko-fi python3-libvoikko
```

# 5. Run the pipeline
```bash
python run_labeling_pipeline.py
```
