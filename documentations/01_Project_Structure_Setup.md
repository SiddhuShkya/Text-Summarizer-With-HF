## Text Summarization : An NLP Project With Huggingface 

In this project, we are going to implement a text summarization using the various open-source models that are provided by huggingface. We will also be fine tuning the models for summarization

---

### 1. Project Structure & GitHub Repo Setup

In this section, we are going start our project implementation. We are going to set up our project github repository and create an initial project structure suitable for the project.

1.1 Firstly, let's create an github repositorty with the project name 'Text-Summarizer-With-HF'

<img src="../screenshots/text-summarizer-repo.png"
    alt="Image Caption"
    style="border:1px solid white; padding:1px; background:#fff; width: 3000px;" />

> Also clone the newly created project repo into our local machine, from withing the colned repo open up your vs-code

```sh
siddhu@ubuntu:~/Desktop$ git clone git@github.com:SiddhuShkya/Text-Summarizer-With-HF.git
Cloning into 'Text-Summarizer-With-HF'...
remote: Enumerating objects: 4, done.
remote: Counting objects: 100% (4/4), done.
remote: Compressing objects: 100% (3/3), done.
remote: Total 4 (delta 0), reused 0 (delta 0), pack-reused 0 (from 0)
Receiving objects: 100% (4/4), 12.75 KiB | 12.75 MiB/s, done.
siddhu@ubuntu:~/Desktop$ cd Text-Summarizer-With-HF/
siddhu@ubuntu:~/Desktop/Text-Summarizer-With-HF$ code .
```

<img src="../screenshots/text-summarizer-vscode.png"
    alt="Image Caption"
    style="border:1px solid white; padding:1px; background:#fff; width: 3000px;" />

1.2 Create and activate an environment for this project using the below commands

```sh
siddhu@ubuntu:~/Desktop/Text-Summarizer-With-HF$ conda create -p venv python==3.10
siddhu@ubuntu:~/Desktop/Text-Summarizer-With-HF$ conda activate venv/
```

1.3 Create 2 new files requirements.txt and template.py, and copy paste the below dependences and python script to them respectively.

> Copy paste the below to the requirements.txt file

```text
## requirements.txt

ipykernel
fastapi
transformers>=4.26.0,<5.0.0
transformers[sentencepiece]
datasets
sacrebleu
rouge_score
py7zr
pandas
nltk
tqdm
PyYAML
matplotlib
torch
evaluate
boto3
mypy-boto3-s3
accelerate>=1.1.0
python-box==6.0.2
ensure==1.0.2
uvicorn==0.18.3
Jinja2==3.1.2
notebook
jupyter>=1.0.0
ipywidgets>=8.0.0
```

> Install the dependencies to your conda environment, using the requirements.txt file

```sh
(/home/siddhu/Desktop/Text-Summarizer-With-HF/venv) siddhu@ubuntu:~/Desktop/Text-Summarizer-With-HF$ pip install -r requirements.txt 
```

> While the installion is being done, lets also complete our template.py, copy paste the below script to template.py

```python
## template.py

import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")

project_name = "textSummarizer"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/logging/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "params.yaml",
    "config/config.yaml",
    ".gitignore",
    "app.py",
    "main.py",
    "requirements.txt",
    "Dockerfile",
    "setup.py",
    "research/research-notebook.ipynb",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}, skipping creation.")
```

> This file helps to create all required directories and empty files for our project in a consistent structure, instead of having to do it manually. This is especially useful in:

- MLOps projects
- Production-grade ML pipelines
- Team environments

1.4 Initiate our project structure using the template.py

```sh
(/home/siddhu/Desktop/Text-Summarizer-With-HF/venv) siddhu@ubuntu:~/Desktop/Text-Summarizer-With-HF$ python template.py 
```

> After, running the above command our project structure should look something like this:

```sh
.
├── app.py
├── config
│   └── config.yaml
├── Dockerfile
├── .gitignore
├── .git
├── .github
│   └── workflows
├── LICENSE
├── main.py
├── params.yaml
├── README.md
├── requirements.txt
├── research
│   └── research-notebook.ipynb
├── setup.py
├── src
│   └── textSummarizer
├── template.py
└── venv
```

1.5 Add the below two folders to .gitignore file as we dont have to track them 

```text
/venv
/artifacts
/logs

# Ignore Jupyter Notebook checkpoints
.ipynb_checkpoints/
# Ignore Python cache files
__pycache__/
*.pyc
```

1.6 Add, commit and push our changes to the main branch of our github repository

```sh
(/home/siddhu/Desktop/Text-Summarizer-With-HF/venv) siddhu@ubuntu:~/Desktop/Text-Summarizer-With-HF$ git add .
(/home/siddhu/Desktop/Text-Summarizer-With-HF/venv) siddhu@ubuntu:~/Desktop/Text-Summarizer-With-HF$ git commit -m 'Initial Project Structure'
(/home/siddhu/Desktop/Text-Summarizer-With-HF/venv) siddhu@ubuntu:~/Desktop/Text-Summarizer-With-HF$ git push origin main
```

> Verify the commit from the github repository page by reloading it

<img src="../screenshots/text-summarizer-git-commit.png"
    alt="Image Caption"
    style="border:1px solid white; padding:1px; background:#fff; width: 3000px;" />

> The Project Structure setup for this project has been completed.

