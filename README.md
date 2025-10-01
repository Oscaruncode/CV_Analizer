# Sentiment and cv analizer API

## 🚀 Getting Started

### 1. Clone the repository

Open your terminal (CMD, PowerShell, or bash) and run:

```bash
git clone https://github.com/Oscaruncode/sentimentAnalysis.git
cd sentimentAnalysis

2. Create a virtual environment (optional but recommended)

On Windows:

python -m venv venv
venv\Scripts\activate


On Linux/Mac:

python3 -m venv venv
source venv/bin/activate

3. Install dependencies

The requirements.txt file contains all the necessary libraries. Install them with:

pip install -r requirements.txt

4. Run the API

Start the FastAPI server using Uvicorn:

uvicorn main:app --reload --port 8000


The API will be available at: http://localhost:8000