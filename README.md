# Simple RAG to extract information, including chart data, from PDF; query the file using natural language

If you have a copy of the `test_rag.ipynb`, then in your current  working directory `<CWD>`:
```bash
curl -L -o  <CWD>/load_utils.py  
curl -L -o  <CWD>/chart_to_structured.py chart_to_structured.py
curl -L -o  <CWD>/demo.py demo.py    

Then, run the following commands:

```bash
git clone https://github.com/opendatalab/MinerU.git
cp demo.py MinerU/demo/
cd MinerU/demo/
rm -rf output
rm pdfs/*
cp ../test_rag_ph.pdf pdfs/


## Setup Instructions

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows