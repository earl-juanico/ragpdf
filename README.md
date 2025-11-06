# Simple RAG to extract information, including chart data, from PDF; query the file using natural language

	* If you have a copy of the `test_rag.ipynb`, then in your current  working directory:
        ```
			curl … load_utils.py chart_to_structured.py demo.py in your CWD
    

	* ```git clone https://github.com/opendatalab/MinerU.git 
	* ```cp demo.py MinerU/demo/
	* cd MinerU/demo/
	* rm -rf output 
	* rm pdfs/*
	* cp test_rag_ph.pdf pdfs/

## Setup Instructions

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows