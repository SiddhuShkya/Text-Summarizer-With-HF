<h1 style="text-align: center; color: #888888; font-style: italic;">
Text Summarization with HuggingFace 🤖📝
</h1>

## Project Description 📖
This project implements an end-to-end Text Summarization pipeline using state-of-the-art Natural Language Processing (NLP) models. It covers everything from project initialization and data ingestion to model fine-tuning, modular development, and API deployment. The system is designed to provide concise summaries of long dialogues or articles.

## Demo (Web App Interface) 💻
Below is the screenshot of the FastAPI-based web application (Swagger UI) used for training and prediction:

<img src="screenshots/fastapi-text-summarizer-webapp.png"
    alt="FastAPI Web App"
    style="border:1px solid white; padding:1px; background:#fff; width: 100%;" />

## HuggingFace Model Used 🤗
The project utilizes the **[google/flan-t5-small](https://huggingface.co/google/flan-t5-small)** model. 
- **Type:** Encoder-Decoder (Seq2Seq) 🔄
- **Framework:** PyTorch 🔥
- **Capability:** Efficient and lightweight for text-to-text tasks like summarization. ⚡

## Dataset Used 📊
The model is fine-tuned using the **[samsum](https://huggingface.co/datasets/knkarthick/samsum)** dataset.
- **Description:** A collection of ~16k messenger-like conversations with corresponding summaries. 💬
- **Structure:** Contains `id`, `dialogue`, and `summary` fields. 📋

## Tech Stack 🛠️
- **Language:** Python 3.10 🐍
- **NLP Framework:** HuggingFace Transformers, Datasets 🤗
- **Backend Framework:** FastAPI, Uvicorn ⚡
- **Deep Learning Library:** PyTorch 🔥
- **Environment Management:** Conda 📦
- **Logging & Utilities:** Python Logging, ConfigBox, Ensure 📑
- **Packaging:** Setup.py 📂

## Project Structure 📂
```text
.
├── app.py                      # FastAPI application
├── main.py                     # Main execution pipeline
├── config/
│   └── config.yaml             # Project configuration
├── params.yaml                 # Model hyperparameters
├── artifacts/                  # Created during execution (Data, Model, Metrics)
├── research/                   # Jupyter notebooks for experimentation
├── screenshots/                # Project screenshots
├── src/
│   └── textSummarizer/
│       ├── components/         # Modular project components
│       ├── config/             # Configuration management
│       ├── constants/          # Constant variables
│       ├── entity/             # Data classes
│       ├── logging/            # Logging setup
│       ├── pipeline/           # Training and Prediction pipelines
│       └── utils/              # Common utility functions
├── requirements.txt
├── Dockerfile
└── setup.py
```

## How to Run? 🚀
1. **Clone the repository:** 📥

   ```sh
   git clone https://github.com/SiddhuShkya/Text-Summarizer-With-HF.git
   cd Text-Summarizer-With-HF
   ```
2. **Create and activate environment:** 🛠️

   ```sh
   conda create -p venv python=3.10 -y
   conda activate venv/
   ```
3. **Install dependencies:** 📦

   ```sh
   pip install -r requirements.txt
   ```
4. **Run the application:** ▶️

   ```sh
   python app.py
   ```
5. **Access the API:** 🌐
   
   ```text
   http://localhost:8000/docs
   ```

## Docker Deployment 🐳

### 1. Using Docker Directly 📦

**Build the Docker image:**

```sh
docker build -t text-s .
```

**Run the Docker container:**
> [!IMPORTANT]
> The application inside the container runs on port **8000**. You must map it to your desired host port.

```sh
docker run -p 8000:8000 text-s
```
*(To use a different host port, e.g., 8080, use `-p 8080:8000`)*

### 2. Using Docker Compose 🛠️

**Start the application:**
```sh
docker-compose up
```

**Stop the application:**
```sh
docker-compose down
```

> [!TIP]
> If you get a "Connection Refused" error, ensure you are using the correct port mapping (e.g., `8000:8000`). If your app listens on `0.0.0.0:8000` inside the container, mapping `-p 8080:8080` will fail because nothing is listening on 8080 inside the container.