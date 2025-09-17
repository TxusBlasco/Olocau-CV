# Olocau-CV 
Olocau-CV is an AI-powered app designed by Txus Blasco (https://github.com/TxusBlasco/Olocau-CV) for screening CVs with simple AI technology.

## Generate fake CVs
### 1) Install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

### 2) Generate 28 PDFs
python tools/gen_fake_cvs.py

### (optional) choose amount / seed / folder
python tools/gen_fake_cvs.py --n 30 --seed 7 --outdir data/samples/fake_cvs
