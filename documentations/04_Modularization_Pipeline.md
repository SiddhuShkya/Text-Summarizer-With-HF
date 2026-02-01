
### 4. Modularizing the Project & Building the End-to-End Pipeline

In this section, we will convert our previously created notebook (which was written as a linear, step-by-step script) into a modular, reusable, and clean code structure. This transformation will make our project scalable, maintainable, and production-ready.

We will also define our workflow so that we can automate the entire end-to-end project creation—from data ingestion to deployment. The workflow includes:

- `Config.yaml` => Store file paths, directories, and basic configuration settings.
- `Params.yaml` => Store hyperparameters and model-specific settings such as test size, random seed, model parameters, etc.
- `Config Entity` => Create structured data models (using dataclasses) to represent configuration settings.
- `Configuration Manager` => A centralized class responsible for reading configuration files and providing configuration objects to the components.
- `Components` => 
    - Refactor the core modules to follow modular design:
        - Data Ingestion
        - Data Transformation
        - Model Trainer
- `Pipelines` => 
Build the Training Pipeline and Prediction Pipeline to automate the complete workflow.
- `Frontend – APIs` => 
    - Create APIs for:
        - Training
        - Batch Prediction
    - This will enable the project to be deployed and used as a real application.

---

4.1 Let's first write our constants and yaml files (config & params). copy paste the below to their respective files

> constants

```python
## src/textSummarizer/constants/__init__.py

from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")
```

> config.yaml

```yaml
## config/config.yaml

artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  source_url: "https://github.com/SiddhuShkya/Text-Summarizer-With-HF/raw/main/data/summarizer-data.zip"
  local_data_file: artifacts/data_ingestion/summarizer-data.zip
  unzip_dir: artifacts/data_ingestion

data_transformation:
  root_dir: artifacts/data_transformation
  data_path: artifacts/data_ingestion/summarizer-data
  tokenizer_name: "google/flan-t5-small"

model_trainer:
  root_dir: artifacts/model_trainer
  data_path: artifacts/data_transformation/summarizer_dataset
  model_ckpt: "google/flan-t5-small"

model_evaluation:
  root_dir: artifacts/model_evaluation
  data_path: artifacts/data_transformation/summarizer_dataset
  model_path: artifacts/model_trainer/t5_model
  tokenizer_path: artifacts/model_trainer/t5_tokenizer
  metric_file_name: artifacts/model_evaluation/metrics.csv
```

> params.yaml

```yaml
TrainerArguments:
  num_train_epochs: 1
  warmup_steps: 500
  per_device_train_batch_size: 1
  weight_decay: 0.01
  logging_steps: 10
  eval_strategy: "steps"
  eval_steps: 500
  save_steps: 100000
  gradient_accumulation_steps: 16
```

4.2 Now let's implement our data ingestion

For better understanding, we will be implementing the data ingestion process using a jupyter notebook file, and then we will try to convert it into a python script file (.py)

> Go ahead and create a new notebook file (data-ingestion.ipynb) inside your project's research folder.

```text
.
├── app.py
├── config
├── data
├── Dockerfile
├── .git
├── .github
├── .gitignore
├── LICENSE
├── logs
├── main.py
├── params.yaml
├── README.md
├── requirements.txt
├── research
│   ├── data-ingestion.ipynb  <----------------------- ## Your new notebook file
│   ├── .ipynb_checkpoints
│   ├── pegasus-finetuned
│   ├── pegasus-model
│   ├── pegasus-tokenizer
│   ├── research-notebook.ipynb
│   ├── summarizer-data
│   ├── summarizer-data.zip
│   ├── text-summarizer.ipynb
│   └── text-summarizer.ipynb - Colab.pdf
├── setup.py
├── src
├── template.py
└── venv
```

> Copy paste the below code cell by cell

- Check your current working directory

```python
## Cell 1

%pwd
```
```text
'/home/siddhu/Desktop/Text-Summarizer-With-HF/research'
```

- Move your notebook file to our parent project directory.

```python
## Cell 2

import os

os.chdir("../")
%pwd
```
```text
'/home/siddhu/Desktop/Text-Summarizer-With-HF'
```

- Import all the necessary dependencies

```python
## Cell 3

import zipfile
import urllib.request as request
from dataclasses import dataclass
from src.textSummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.textSummarizer.utils.common import read_yaml, create_directories
from src.textSummarizer.logging import logger
```

- Lets create a dataclass to read our yaml file and store every fields. This  will be used as input for our data ingestion module to create folders and store data automatically

```python
## Cell 4

@dataclass
class DataIngestionConfig:
    root_dir: str
    source_url: str
    local_data_file: str
    unzip_dir: str
```

- Lets create another class for defining our configuration manager

```python
## Cell 5

class ConfigurationManager:
    
    def __init__(self, config_path=CONFIG_FILE_PATH, params_path=PARAMS_FILE_PATH):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        create_directories([self.config.artifacts_root])
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,  
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )
        return data_ingestion_config
```

- Define the components

```python
## Cell 6

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    def download_data(self):
        if not os.path.exists(self.config.local_data_file):
            file_name, headers = request.urlretrieve(
                url=self.config.source_url, 
                filename=self.config.local_data_file
            )
            logger.info("File downloaded successfully!")
        else:
            logger.info(f"File already exists at {self.config.local_data_file}. Skipping download.")
        
    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"File extracted successfully at {unzip_path}")
```

- You can use the below code to test if everything is working fine or not.

```python
## Cell 7 (Optional)

config = ConfigurationManager()
data_ingestion_config = config.get_data_ingestion_config()
data_ingestion = DataIngestion(config=data_ingestion_config)
data_ingestion.download_data()
data_ingestion.extract_zip_file()
```
```text
[2026-01-31 09:12:41,454: INFO: common: YAML file 'config/config.yaml' read successfully.]
[2026-01-31 09:12:41,487: INFO: common: YAML file 'params.yaml' read successfully.]
[2026-01-31 09:12:41,488: INFO: common: Directory 'artifacts' created successfully or already exists.]
[2026-01-31 09:12:41,489: INFO: common: Directory 'artifacts/data_ingestion' created successfully or already exists.]
[2026-01-31 09:12:43,498: INFO: 2552295685: File downloaded successfully!]
[2026-01-31 09:12:43,554: INFO: 2552295685: File extracted successfully at artifacts/data_ingestion]
```

> Since, the above code is running fine, lets modularize it by copy pasting the code blocks to their respective files

- Update entity

```python
## src/textSummarizer/entity/__init__.py

from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: str
    source_url: str
    local_data_file: str
    unzip_dir: str
```

- Update config

```python
## src/textSummarizer/config/configuration.py

from src.textSummarizer.constants import *
from src.textSummarizer.entity import DataIngestionConfig
from src.textSummarizer.utils.common import read_yaml, create_directories

class ConfigurationManager:
    def __init__(self, config_path=CONFIG_FILE_PATH, params_path=PARAMS_FILE_PATH):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )
        return data_ingestion_config
```

- Update Components, also create a new python file (data_ingetsion.py) inside the src/textSummarizer/components for this step

```python
## src/textSummarizer/components/data_ingestion.py

import os
import zipfile
import urllib.request as request
from src.textSummarizer.logging import logger
from src.textSummarizer.entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self):
        if not os.path.exists(self.config.local_data_file):
            file_name, headers = request.urlretrieve(
                url=self.config.source_url, filename=self.config.local_data_file
            )
            logger.info("File downloaded successfully!")
        else:
            logger.info(
                f"File already exists at {self.config.local_data_file}. Skipping download."
            )

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"File extracted successfully at {unzip_path}")

```

- Create our first stage for our pipeline (stage1_data_ingestion.py)

```python
## src/textSummarizer/pipeline/stage1_data_ingestion.py

from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.components.data_ingestion import DataIngestion
from src.textSummarizer.logging import logger


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_ingestion(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_data()
            data_ingestion.extract_zip_file()
        except Exception as e:
            logger.exception(e)
            raise e
```

> Now, lets test if everyting is working fine or not

- Update main.py. Copy paste the below code to main.py

```python
## main.py

from src.textSummarizer.logging import logger
from src.textSummarizer.pipeline.stage1_data_ingestion import (
    DataIngestionTrainingPipeline,
)

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.initiate_data_ingestion()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(f"Error in stage {STAGE_NAME}: {e}")
    raise e
```

- Delete the artifacts folder if it already exists in your project folder

```text
.
├── app.py
├── artifacts   <---------------- # Delete this, if exists
├── config
├── data
├── Dockerfile
├── .git
├── .github
├── .gitignore
├── LICENSE
├── logs
├── main.py
├── params.yaml
├── README.md
├── requirements.txt
├── research
├── setup.py
├── src
├── template.py
└── venv
```

- Run main.py

```sh
(/home/siddhu/Desktop/Text-Summarizer-With-HF/venv) siddhu@ubuntu:~/Desktop/Text-Summarizer-With-HF$ python main.py 
config/config.yaml
params.yaml
[2026-01-27 09:24:39,651: INFO: main: >>>>>> Stage Data Ingestion Stage started <<<<<<]
[2026-01-27 09:24:39,651: INFO: common: YAML file 'config/config.yaml' read successfully.]
[2026-01-27 09:24:39,652: INFO: common: YAML file 'params.yaml' read successfully.]
[2026-01-27 09:24:39,652: INFO: common: Directory 'artifacts' created successfully or already exists.]
[2026-01-27 09:24:39,652: INFO: common: Directory 'artifacts/data_ingestion' created successfully or already exists.]
[2026-01-27 09:24:44,034: INFO: data_ingestion: File downloaded successfully!]
[2026-01-27 09:24:44,082: INFO: data_ingestion: File extracted successfully at artifacts/data_ingestion]
[2026-01-27 09:24:44,082: INFO: main: >>>>>> Stage Data Ingestion Stage completed <<<<<<]
```

*If you see your output similar to the above, then everything is working fine till now*

> Commit and push the changes to github

```sh
git add .
git commit -m 'Data Ingestion Modularization Completed'
git push origin main
```

4.3 Now lets implement our data transformation

In this we are going proceed ahead and implement the data transformation which is nothing but feature engineering.

Similar to what we did for data ingestion, for better understanding, we will be implementing the data transformation process using a jupyter notebook file first, and then we will try to convert it into a python script file (.py)

> Create the new notebook file (data-transformation.ipynb) inside your research folder

```text
.
├── app.py
├── config
├── data
├── Dockerfile
├── .git
├── .github
├── .gitignore
├── LICENSE
├── logs
├── main.py
├── params.yaml
├── README.md
├── requirements.txt
├── research
│   ├── data-transformation.ipynb  <----------------------- ## Your new notebook file
│   ├── data-ingestion.ipynb  
│   ├── .ipynb_checkpoints
│   ├── pegasus-finetuned
│   ├── pegasus-model
│   ├── pegasus-tokenizer
│   ├── research-notebook.ipynb
│   ├── summarizer-data
│   ├── summarizer-data.zip
│   ├── text-summarizer.ipynb
│   └── text-summarizer.ipynb - Colab.pdf
├── setup.py
├── src
├── template.py
└── venv
```

> Copy paste the below code cell by cell:

- Update the present working directory to your parent folder

```python
## Cell 1

import os

os.chdir('../')
%pwd
```
```text
'/home/siddhu/Desktop/Text-Summarizer-With-HF'
```

- Import necessary dependencies

```python
## Cell 2

import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from src.textSummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.textSummarizer.utils.common import read_yaml, create_directories
```

- Create dataclass to store data transformation fields

```python
## Cell 3

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: Path
```

- Create our configuration manager

```python
## Cell 4

class ConfigurationManager:
    def __init__(self, config_path=CONFIG_FILE_PATH, params_path=PARAMS_FILE_PATH):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        create_directories([self.config.artifacts_root])

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])
        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            tokenizer_name=config.tokenizer_name
        )
        return data_transformation_config
```

- Implement Our Data Transformation component

```python
## Cell 5

class DataTransformation:
    
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name
        )

    def convert_examples_to_features(self, example_batch):
        model_inputs = self.tokenizer(
            example_batch["dialogue"],  # Tokenize input dialogue
            text_target=example_batch["summary"],  # Tokenize target summary for seq2seq
            max_length=1024,  # Max length for input tokens
            truncation=True,  # Truncate if longer than max_length
        )
        labels = self.tokenizer(
            text_target=example_batch["summary"],
            max_length=128,
            truncation=True,  # Tokenize summary as labels
        )
        model_inputs["labels"] = labels["input_ids"]  # Add tokenized labels to model inputs
        return model_inputs

    def convert(self):
        dataset_splits = {}

        for split in ["train", "validation", "test"]:
            csv_path = os.path.join(
                self.config.data_path, f"{split}.csv"
            )

            df = pd.read_csv(csv_path)
            ds = Dataset.from_pandas(df)

            ds = ds.map(
                self.convert_examples_to_features,
                batched=True
            )

            dataset_splits[split] = ds

        summarizer_dataset = DatasetDict(dataset_splits)

        summarizer_dataset.save_to_disk(
            os.path.join(self.config.root_dir, "summarizer_dataset")
        )
```

- You can use the below code to test if everything is working fine or not.

```python 
## Cell 6

config = ConfigurationManager()
data_transformation_config = config.get_data_transformation_config()
data_transformation = DataTransformation(config=data_transformation_config)
data_transformation.convert()
```
```text
[2026-02-01 09:59:42,404: INFO: common: YAML file 'config/config.yaml' read successfully.]
[2026-02-01 09:59:42,406: INFO: common: YAML file 'params.yaml' read successfully.]
[2026-02-01 09:59:42,407: INFO: common: Directory 'artifacts' created successfully or already exists.]
[2026-02-01 09:59:42,408: INFO: common: Directory 'artifacts/data_transformation' created successfully or already exists.]
Map: 100%|██████████| 14731/14731 [00:12<00:00, 1189.45 examples/s]
Map: 100%|██████████|   818/818   [00:01<00:00, 812.67 examples/s]
Map: 100%|██████████|   819/819   [00:01<00:00, 804.31 examples/s]
Saving the dataset (0/1 shards): 100%|██████████| 14731/14731 [00:03<00:00, 4321.18 examples/s]
Saving the dataset (0/1 shards): 100%|██████████|   818/818   [00:00<00:00, 5120.44 examples/s]
Saving the dataset (0/1 shards): 100%|██████████|   819/819   [00:00<00:00, 4987.62 examples/s]
```
> Since, all the above code is running fine, lets modularize it by copy pasting the code blocks to their respective files

- Update entity

```python
## src/textSummarizer/entity/__init__.py

from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: str
    source_url: str
    local_data_file: str
    unzip_dir: str

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: Path
```

- Update config

```python
## src/textSummarizer/config/configuration.py

from src.textSummarizer.constants import *
from src.textSummarizer.entity import (
    DataIngestionConfig,
    DataTransformationConfig
)
from src.textSummarizer.utils.common import read_yaml, create_directories

class ConfigurationManager:
    def __init__(self, config_path=CONFIG_FILE_PATH, params_path=PARAMS_FILE_PATH):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )
        return data_ingestion_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])
        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            tokenizer_name=config.tokenizer_name,
        )
        return data_transformation_config
```

- Update components, also create a new python file (data_transformation.py) inside the src/textSummarizer/components for this step

```python
## src/textSummarizer/components/data_transformation.py

import os
import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from src.textSummarizer.logging import logger
from src.textSummarizer.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        logger.info(f"Initialized DataTransformation with tokenizer: {self.config.tokenizer_name}")

    def convert_examples_to_features(self, example_batch):
        model_inputs = self.tokenizer(
            example_batch["dialogue"],  # Tokenize input dialogue
            text_target=example_batch["summary"],  # Tokenize target summary for seq2seq
            max_length=1024,  # Max length for input tokens
            truncation=True,  # Truncate if longer than max_length
        )
        labels = self.tokenizer(
            text_target=example_batch["summary"],
            max_length=128,
            truncation=True,  # Tokenize summary as labels
        )
        model_inputs["labels"] = labels[
            "input_ids"
        ]  # Add tokenized labels to model inputs
        return model_inputs

    def convert(self):
        dataset_splits = {}

        for split in ["train", "validation", "test"]:
            csv_path = os.path.join(self.config.data_path, f"{split}.csv")

            df = pd.read_csv(csv_path)
            ds = Dataset.from_pandas(df)

            ds = ds.map(self.convert_examples_to_features, batched=True)

            dataset_splits[split] = ds

        summarizer_dataset = DatasetDict(dataset_splits)
        logger.info("Saving transformed dataset to disk...")
        summarizer_dataset.save_to_disk(
            os.path.join(self.config.root_dir, "summarizer_dataset")
        )
        logger.info(f"Dataset saved successfully at {os.path.join(self.config.root_dir, 'summarizer_dataset')}")
```

- Create our second stage for our pipeline (stage2_data_transformation.py)

```python
## src/textSummarizer/pipeline/stage2_data_transformation.py

from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.components.data_transformation import DataTransformation
from src.textSummarizer.logging import logger


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            data_transformation.convert()
        except Exception as e:
            logger.exception(e)
            raise e
```

> Now, lets test if everyting is working fine or not

- Update main.py. Copy paste the below code to main.py

```python
## main.py

from src.textSummarizer.logging import logger
from src.textSummarizer.pipeline.stage1_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from src.textSummarizer.pipeline.stage2_data_transformation import (
    DataTransformationTrainingPipeline,
)

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.initiate_data_ingestion()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(f"Error in stage {STAGE_NAME}: {e}")
    raise e

STAGE_NAME = "Data Transformation Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.initiate_data_transformation()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(f"Error in stage {STAGE_NAME}: {e}")
    raise e
```

- Delete the artifacts folder if it already exists in your project folder

```text
.
├── app.py
├── artifacts   <---------------- # Delete this, if exists
├── config
├── data
├── Dockerfile
├── .git
├── .github
├── .gitignore
├── LICENSE
├── logs
├── main.py
├── params.yaml
├── README.md
├── requirements.txt
├── research
├── setup.py
├── src
├── template.py
└── venv
```

- Run main.py

```sh
(/home/siddhu/Desktop/Text-Summarizer-With-HF/venv) siddhu@ubuntu:~/Desktop/Text-Summarizer-With-HF$ python main.py 
[2026-02-01 10:07:11,408: INFO: main: >>>>>> Stage Data Ingestion Stage started <<<<<<]
[2026-02-01 10:07:11,410: INFO: common: YAML file 'config/config.yaml' read successfully.]
[2026-02-01 10:07:11,411: INFO: common: YAML file 'params.yaml' read successfully.]
[2026-02-01 10:07:11,411: INFO: common: Directory 'artifacts' created successfully or already exists.]
[2026-02-01 10:07:11,411: INFO: common: Directory 'artifacts/data_ingestion' created successfully or already exists.]
[2026-02-01 10:07:13,163: INFO: data_ingestion: File downloaded successfully!]
[2026-02-01 10:07:13,215: INFO: data_ingestion: File extracted successfully at artifacts/data_ingestion]
[2026-02-01 10:07:13,215: INFO: main: >>>>>> Stage Data Ingestion Stage completed <<<<<<]
[2026-02-01 10:07:13,215: INFO: main: >>>>>> Stage Data Transformation Stage started <<<<<<]
[2026-02-01 10:07:13,217: INFO: common: YAML file 'config/config.yaml' read successfully.]
[2026-02-01 10:07:13,218: INFO: common: YAML file 'params.yaml' read successfully.]
[2026-02-01 10:07:13,218: INFO: common: Directory 'artifacts' created successfully or already exists.]
[2026-02-01 10:07:13,218: INFO: common: Directory 'artifacts/data_transformation' created successfully or already exists.]
[2026-02-01 10:07:14,321: INFO: data_transformation: Initialized DataTransformation with tokenizer: google/flan-t5-small]
Map: 100%|███████████████| 14731/14731 [00:01<00:00, 7857.65 examples/s]
Map: 100%|███████████████████| 818/818 [00:00<00:00, 9459.31 examples/s]
Map: 100%|███████████████████| 819/819 [00:00<00:00, 9207.48 examples/s]
[2026-02-01 10:07:16,571: INFO: data_transformation: Saving transformed dataset to disk...]
Saving the dataset (1/1 shards): 100%|█| 14731/14731 [00:00<00:00, 57063
Saving the dataset (1/1 shards): 100%|█| 818/818 [00:00<00:00, 215497.81
Saving the dataset (1/1 shards): 100%|█| 819/819 [00:00<00:00, 206587.38
[2026-02-01 10:07:16,609: INFO: data_transformation: Dataset saved successfully at artifacts/data_transformation/summarizer_dataset]
[2026-02-01 10:07:16,634: INFO: main: >>>>>> Stage Data Transformation Stage completed <<<<<<]
```

*If you see your output similar to the above, then everything is working fine till now*

> Commit and push the changes to github

```sh
git add .
git commit -m 'Data Transformation Modularization Completed'
git push origin main
```

4.4 Now, lets implement our model trainer module

In this section we are going to proceed ahead and implement our model trainer for this project.

Similar to what we did for data ingestion and transformation, for better understanding, we will be implementing our model trainer process using a jupyter notebook file first, and then we will try to convert it into a python script file (.py)

> Create the new notebook file (model-trainer.ipynb) inside your research folder

```text
.
├── app.py
├── config
├── data
├── Dockerfile
├── .git
├── .github
├── .gitignore
├── LICENSE
├── logs
├── main.py
├── params.yaml
├── README.md
├── requirements.txt
├── research
│   ├── model-trainer.ipynb  <----------------------- ## Your new notebook file
│   ├── data-transformation.ipynb  
│   ├── data-ingestion.ipynb  
│   ├── .ipynb_checkpoints
│   ├── pegasus-finetuned
│   ├── pegasus-model
│   ├── pegasus-tokenizer
│   ├── research-notebook.ipynb
│   ├── summarizer-data
│   ├── summarizer-data.zip
│   ├── text-summarizer.ipynb
│   └── text-summarizer.ipynb - Colab.pdf
├── setup.py
├── src
├── template.py
└── venv
```

> Copy paste the below codes cell by cell

- Update the present working directory to your parent folder

```python
## Cell 1

import os

os.chdir('../')
%pwd
```
```text
'/home/siddhu/Desktop/Text-Summarizer-With-HF'
```

- Import necessary dependencies

```python
## Cell 2

import torch
from datasets import load_from_disk
from dataclasses import dataclass
from pathlib import Path
from src.textSummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.textSummarizer.utils.common import read_yaml, create_directories
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq
```

- Create dataclass to store fields of our model trainer config

```python
## Cell 3

@dataclass
class ModelTrainerConfig:
    # From config.yaml
    root_dir: Path
    data_path: Path
    model_ckpt: Path
    # From params.yaml
    num_train_epochs: int
    warmup_steps: int
    per_device_train_batch_size: int
    weight_decay: float
    logging_steps: int
    eval_strategy: str
    eval_steps: int
    save_steps: float
    gradient_accumulation_steps: int
```

- Create our configuration manager

```python
## Cell 4

class ConfigurationManager:

    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])
        
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainerArguments
        create_directories([config.root_dir])
        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            model_ckpt=config.model_ckpt,
            num_train_epochs=params.num_train_epochs,
            warmup_steps=params.warmup_steps,
            per_device_train_batch_size=params.per_device_train_batch_size,
            weight_decay=params.weight_decay,
            logging_steps=params.logging_steps,
            eval_strategy=params.eval_strategy,
            eval_steps=params.eval_steps,
            save_steps=params.save_steps,
            gradient_accumulation_steps=params.gradient_accumulation_steps
        )
        return model_trainer_config
```

- Create our model trainer component

```python
## Cell 5

class ModelTrainer:
    
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def train(self):
        # Load dataset
        dataset = load_from_disk(self.config.data_path)
        
        # Load model and tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)

        # Data collator
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.eval_strategy,
            warmup_steps=self.config.warmup_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset['test'],
            eval_dataset=dataset['validation'],
            data_collator=data_collator,
            processing_class=tokenizer,
        )
        
        # Start training
        trainer.train()

        # Save the model and tokenizer
        model.save_pretrained(os.path.join(self.config.root_dir, "t5_model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "t5_tokenizer"))
```

- You can use the below code to test if everything is working fine or not.

```python 
## Cell 6

config = ConfigurationManager()
model_trainer_config = config.get_model_trainer_config()
model_trainer = ModelTrainer(config=model_trainer_config)
model_trainer.train()
```
**Output case 1:**
```text
[2026-01-31 09:53:00,066: INFO: common: YAML file 'config/config.yaml' read successfully.]
[2026-01-31 09:53:00,068: INFO: common: YAML file 'params.yaml' read successfully.]
[2026-01-31 09:53:00,069: INFO: common: Directory 'artifacts' created successfully or already exists.]
[2026-01-31 09:53:00,070: INFO: common: Directory 'artifacts/model_trainer' created successfully or already exists.]
```
<div>
    <progress value='52' max='52' style='width:300px; height:20px; vertical-align: middle;'></progress>
    [52/52 00:49, Epoch 1/1]
</div>
<br>

**Output Case 2:**
```text
OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 3.63 GiB of which 51.94 MiB is free. 
```

**📌 Important Note:**  

You may encounter an **OutOfMemoryError** while training on a GPU with limited memory (like GTX 1650 with ~3.6 GB VRAM).  

- Sometiimes, large models require more memory than our GPU has to offer.  
- This can happen when loading the model or during the forward/backward pass.  


-> *If you encounter such issues, what you can do is train your model from jupyter notebook in the google colab and then download the entire folder that are basically created while saving the model and directly save it to your project folder*


> Now lets modularize our model trainer, by copy pasting the code blocks to their respective files

- Update entity

```python
## src/textSummarizer/entity/__init__.py

from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: str
    source_url: str
    local_data_file: str
    unzip_dir: str

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: Path

@dataclass
class ModelTrainerConfig:
    # From config.yaml
    root_dir: Path
    data_path: Path
    model_ckpt: Path
    # From params.yaml
    num_train_epochs: int
    warmup_steps: int
    per_device_train_batch_size: int
    weight_decay: float
    logging_steps: int
    eval_strategy: str
    eval_steps: int
    save_steps: float
    gradient_accumulation_steps: int
```

- Update config

```python
## src/textSummarizer/config/configuration.py

from src.textSummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from pathlib import Path
from src.textSummarizer.entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)
from src.textSummarizer.utils.common import read_yaml, create_directories


class ConfigurationManager:
    def __init__(self, config_path=CONFIG_FILE_PATH, params_path=PARAMS_FILE_PATH):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )
        return data_ingestion_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])
        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            tokenizer_name=config.tokenizer_name,
        )
        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainerArguments
        create_directories([config.root_dir])
        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            model_ckpt=config.model_ckpt,
            num_train_epochs=params.num_train_epochs,
            warmup_steps=params.warmup_steps,
            per_device_train_batch_size=params.per_device_train_batch_size,
            weight_decay=params.weight_decay,
            logging_steps=params.logging_steps,
            eval_strategy=params.eval_strategy,
            eval_steps=params.eval_steps,
            save_steps=params.save_steps,
            gradient_accumulation_steps=params.gradient_accumulation_steps,
        )
        return model_trainer_config
```

- Update components, also create a new python file (model_trainer.py) inside the src/textSummarizer/components for this step

```python
## src/textSummarizer/components/model_trainer.py

import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq
from datasets import load_from_disk
from src.textSummarizer.logging import logger
from src.textSummarizer.entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initialized ModelTrainer on device: {self.device}")

    def train(self):
        # Load dataset
        dataset = load_from_disk(self.config.data_path)
        # Load model and tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(
            self.device
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        # Data collator
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.eval_strategy,
            warmup_steps=self.config.warmup_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )
        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["test"],
            eval_dataset=dataset["validation"],
            data_collator=data_collator,
            processing_class=tokenizer,
        )
        # Start training
        logger.info("Starting model training...")
        trainer.train()
        logger.info("Model training completed successfully.")
        # Save the model and tokenizer
        logger.info("Saving model and tokenizer...")
        model.save_pretrained(os.path.join(self.config.root_dir, "t5_model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "t5_tokenizer"))
        logger.info(f"Model and tokenizer saved at {self.config.root_dir}")
```

- Create our third stage for our pipeline (stage3_model_trainer.py)

```python
## src/textSummarizer/pipeline/stage3_model_trainer.py

from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.components.model_trainer import ModelTrainer
from src.textSummarizer.logging import logger

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_trainer(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.train()
        except Exception as e:
            logger.exception(e)
            raise e
```

> Now, lets test if everyting is working fine or not

- Update main.py. Copy paste the below code to main.py

```python
## main.py

from src.textSummarizer.logging import logger
from src.textSummarizer.pipeline.stage1_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from src.textSummarizer.pipeline.stage2_data_transformation import (
    DataTransformationTrainingPipeline,
)
from src.textSummarizer.pipeline.stage3_model_trainer import (
    ModelTrainerTrainingPipeline,
)

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.initiate_data_ingestion()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(f"Error in stage {STAGE_NAME}: {e}")
    raise e

STAGE_NAME = "Data Transformation Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.initiate_data_transformation()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(f"Error in stage {STAGE_NAME}: {e}")
    raise e

STAGE_NAME = "Model Trainer Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    model_trainer = ModelTrainerTrainingPipeline()
    model_trainer.initiate_model_trainer()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(f"Error in stage {STAGE_NAME}: {e}")
    raise e
```

- Delete the artifacts folder if it already exists in your project folder

```text
.
├── app.py
├── artifacts   <---------------- # Delete this, if exists
├── config
├── data
├── Dockerfile
├── .git
├── .github
├── .gitignore
├── LICENSE
├── logs
├── main.py
├── params.yaml
├── README.md
├── requirements.txt
├── research
├── setup.py
├── src
├── template.py
└── venv
```

- Run main.py

```sh
(/home/siddhu/Desktop/Text-Summarizer-With-HF/venv) siddhu@ubuntu:~/Desktop/Text-Summarizer-With-HF$ python main.py
[2026-02-01 10:23:34,320: INFO: main: >>>>>> Stage Data Ingestion Stage started <<<<<<]
[2026-02-01 10:23:34,322: INFO: common: YAML file 'config/config.yaml' read successfully.]
[2026-02-01 10:23:34,324: INFO: common: YAML file 'params.yaml' read successfully.]
[2026-02-01 10:23:34,324: INFO: common: Directory 'artifacts' created successfully or already exists.]
[2026-02-01 10:23:34,324: INFO: common: Directory 'artifacts/data_ingestion' created successfully or already exists.]
[2026-02-01 10:23:35,872: INFO: data_ingestion: File downloaded successfully!]
[2026-02-01 10:23:35,923: INFO: data_ingestion: File extracted successfully at artifacts/data_ingestion]
[2026-02-01 10:23:35,923: INFO: main: >>>>>> Stage Data Ingestion Stage completed <<<<<<]
[2026-02-01 10:23:35,923: INFO: main: >>>>>> Stage Data Transformation Stage started <<<<<<]
[2026-02-01 10:23:35,925: INFO: common: YAML file 'config/config.yaml' read successfully.]
[2026-02-01 10:23:35,926: INFO: common: YAML file 'params.yaml' read successfully.]
[2026-02-01 10:23:35,926: INFO: common: Directory 'artifacts' created successfully or already exists.]
[2026-02-01 10:23:35,926: INFO: common: Directory 'artifacts/data_transformation' created successfully or already exists.]
[2026-02-01 10:23:37,081: INFO: data_transformation: Initialized DataTransformation with tokenizer: google/flan-t5-small]
Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 14731/14731 [00:01<00:00, 9096.39 examples/s]
Map: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 818/818 [00:00<00:00, 10506.28 examples/s]
Map: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 819/819 [00:00<00:00, 10007.04 examples/s]
[2026-02-01 10:23:39,051: INFO: data_transformation: Saving transformed dataset to disk...]
Saving the dataset (1/1 shards): 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 14731/14731 [00:00<00:00, 535502.62 examples/s]
Saving the dataset (1/1 shards): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 818/818 [00:00<00:00, 203098.36 examples/s]
Saving the dataset (1/1 shards): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 819/819 [00:00<00:00, 202019.23 examples/s]
[2026-02-01 10:23:39,091: INFO: data_transformation: Dataset saved successfully at artifacts/data_transformation/summarizer_dataset]
[2026-02-01 10:23:39,113: INFO: main: >>>>>> Stage Data Transformation Stage completed <<<<<<]
[2026-02-01 10:23:39,113: INFO: main: >>>>>> Stage Model Trainer Stage started <<<<<<]
[2026-02-01 10:23:39,115: INFO: common: YAML file 'config/config.yaml' read successfully.]
[2026-02-01 10:23:39,115: INFO: common: YAML file 'params.yaml' read successfully.]
[2026-02-01 10:23:39,115: INFO: common: Directory 'artifacts' created successfully or already exists.]
[2026-02-01 10:23:39,116: INFO: common: Directory 'artifacts/model_trainer' created successfully or already exists.]
[2026-02-01 10:23:39,240: INFO: model_trainer: Initialized ModelTrainer on device: cuda]
[2026-02-01 10:23:42,465: INFO: model_trainer: Starting model training...]
{'loss': 2.0079, 'grad_norm': 2.6245455741882324, 'learning_rate': 9e-07, 'epoch': 0.2}                                                                                               
{'loss': 2.0806, 'grad_norm': 3.358659267425537, 'learning_rate': 1.9e-06, 'epoch': 0.39}                                                                                             
{'loss': 2.0031, 'grad_norm': 2.7413535118103027, 'learning_rate': 2.9e-06, 'epoch': 0.59}                                                                                            
{'loss': 1.9963, 'grad_norm': 2.9518542289733887, 'learning_rate': 3.9e-06, 'epoch': 0.78}                                                                                            
{'loss': 1.8982, 'grad_norm': 2.6860415935516357, 'learning_rate': 4.9000000000000005e-06, 'epoch': 0.98}                                                                             
{'train_runtime': 48.6178, 'train_samples_per_second': 16.846, 'train_steps_per_second': 1.07, 'train_loss': 2.004348736542922, 'epoch': 1.0}                                         
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 52/52 [00:48<00:00,  1.07it/s]
[2026-02-01 10:24:31,212: INFO: model_trainer: Model training completed successfully.]
[2026-02-01 10:24:31,212: INFO: model_trainer: Saving model and tokenizer...]
[2026-02-01 10:24:31,740: INFO: model_trainer: Model and tokenizer saved at artifacts/model_trainer]
[2026-02-01 10:24:31,757: INFO: main: >>>>>> Stage Model Trainer Stage completed <<<<<<]
```

*You may aslo again encounter OutOfMemory Error when you run main.py*

> Commit and push the changes to github

```sh
git add .
git commit -m 'Data Trainer Modularization Completed'
git push origin main
```

4.5 Finally, lets implement our model evaluation module

In this section we are going to proceed ahead and implement our final module which is model trainer.

> Create a new notebook file (model-evaluation.ipynb) under the research folder.

```text
.
├── app.py
├── config
├── data
├── Dockerfile
├── .git
├── .github
├── .gitignore
├── LICENSE
├── logs
├── main.py
├── params.yaml
├── README.md
├── requirements.txt
├── research
│   ├── model-evaluation.ipynb  <----------------------- ## Your new notebook file
│   ├── model-trainer.ipynb  
│   ├── data-transformation.ipynb  
│   ├── data-ingestion.ipynb  
│   ├── .ipynb_checkpoints
│   ├── pegasus-finetuned
│   ├── pegasus-model
│   ├── pegasus-tokenizer
│   ├── research-notebook.ipynb
│   ├── summarizer-data
│   ├── summarizer-data.zip
│   ├── text-summarizer.ipynb
│   └── text-summarizer.ipynb - Colab.pdf
├── setup.py
├── src
├── template.py
└── venv
```

> Copy paste the below code cell by cell to model-evaluation.ipynb notebook file

- Update the present working directory to your parent folder

```python
## Cell 1

import os

os.chdir('../')
%pwd
```
```text
'/home/siddhu/Desktop/Text-Summarizer-With-HF'
```

- Import necessary dependencies

```python
## Cell 2

import torch
import evaluate
from dataclasses import dataclass
from pathlib import Path
from src.textSummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH 
from src.textSummarizer.utils.common import read_yaml, create_directories
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
import pandas as pd
```

- Create dataclass to store fields of our model evaluation config

```python
## Cell 3

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_name: Path
```
- Create our configuration manager

```python
## Cell 4

class ConfigurationManager:
    def __init__(
        self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([config.root_dir])
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_path=config.model_path,
            tokenizer_path=config.tokenizer_path,
            metric_file_name=config.metric_file_name,
        )
        return model_evaluation_config
```

- Create our model evaluation component

```python
## Cell 5

class ModelEvaluation:
    
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rouge_metric = evaluate.load("rouge")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        self.dataset = load_from_disk(self.config.data_path)

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]
            
    def calculate_metric_on_test_ds(
        self,
        dataset,
        metric,
        model,
        tokenizer,
        device,
        batch_size=1,
        column_text="article",
        column_summary="highlights",
    ):
        model.eval()  # Set model to evaluation mode

        # Split dataset into batches
        article_batches = list(
            self.generate_batch_sized_chunks(dataset[column_text], batch_size)
        )
        target_batches = list(
            self.generate_batch_sized_chunks(dataset[column_summary], batch_size)
        )

        with torch.no_grad():  # Disable gradient calculation
            for article_batch, target_batch in zip(article_batches, target_batches):
                inputs = tokenizer(
                    article_batch,
                    max_length=256,  # Max input length
                    truncation=True,  # Truncate if too long
                    padding="max_length",  # Pad to max length
                    return_tensors="pt",  # Return PyTorch tensors
                )

                summaries = model.generate(
                    input_ids=inputs["input_ids"].to(device),  # Move inputs to device
                    attention_mask=inputs["attention_mask"].to(device),
                    max_new_tokens=128,  # Max tokens for output
                    num_beams=1,  # Beam search width
                    do_sample=False,  # Deterministic generation
                    use_cache=True,  # Use past key values for speed
                )

                decoded_summaries = tokenizer.batch_decode(
                    summaries,
                    skip_special_tokens=True,  # Decode outputs to text
                )

                metric.add_batch(
                    predictions=decoded_summaries,  # Add generated summaries
                    references=target_batch,  # Add reference summaries
                )

        return metric.compute()  # Compute final metric (e.g., ROUGE, BLEU)
    
    def evaluate(self):
        dataset = load_from_disk(self.config.data_path)
        rouge_metric = self.rouge_metric
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        score = self.calculate_metric_on_test_ds(
            dataset=dataset["validation"],
            metric=rouge_metric,
            device=self.device,
            model=self.model,
            tokenizer=self.tokenizer,
            batch_size=2,
            column_text="dialogue",
            column_summary="summary"
        )
        rouge_dict = {name: score[name] for name in rouge_names}
        df = pd.DataFrame(rouge_dict, index=["flan-t5-small-finetuned"])
        df.to_csv(self.config.metric_file_name, index=True)
```

- You can use the below code to test if everything is working fine or not.

```python 
## Cell 6

config = ConfigurationManager()
model_evaluation_config = config.get_model_evaluation_config()
model_evaluator = ModelEvaluation(config=model_evaluation_config)
model_evaluator.evaluate()
```
```text
[2026-02-01 11:52:26,946: INFO: common: YAML file 'config/config.yaml' read successfully.]
[2026-02-01 11:52:26,950: INFO: common: YAML file 'params.yaml' read successfully.]
[2026-02-01 11:52:26,951: INFO: common: Directory 'artifacts' created successfully or already exists.]
[2026-02-01 11:52:26,953: INFO: common: Directory 'artifacts/model_evaluation' created successfully or already exists.]
[2026-02-01 11:54:36,209: INFO: rouge_scorer: Using default tokenizer.]
```

> Since, the above code is running fine, lets modularize it by copy pasting the code blocks to their respective files

- Update entity

```python
## src/textSummarizer/entity/__init__.py

from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: str
    source_url: str
    local_data_file: str
    unzip_dir: str

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: Path

@dataclass
class ModelTrainerConfig:
    # From config.yaml
    root_dir: Path
    data_path: Path
    model_ckpt: Path
    # From params.yaml
    num_train_epochs: int
    warmup_steps: int
    per_device_train_batch_size: int
    weight_decay: float
    logging_steps: int
    eval_strategy: str
    eval_steps: int
    save_steps: float
    gradient_accumulation_steps: int

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_name: Path
```

- Update config

```python
## src/textSummarizer/config/configuration.py

from pathlib import Path
from src.textSummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.textSummarizer.utils.common import read_yaml, create_directories
from src.textSummarizer.entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)

class ConfigurationManager:

    def __init__(self, config_path=CONFIG_FILE_PATH, params_path=PARAMS_FILE_PATH):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )
        return data_ingestion_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])
        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            tokenizer_name=config.tokenizer_name,
        )
        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainerArguments
        create_directories([config.root_dir])
        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            model_ckpt=config.model_ckpt,
            num_train_epochs=params.num_train_epochs,
            warmup_steps=params.warmup_steps,
            per_device_train_batch_size=params.per_device_train_batch_size,
            weight_decay=params.weight_decay,
            logging_steps=params.logging_steps,
            eval_strategy=params.eval_strategy,
            eval_steps=params.eval_steps,
            save_steps=params.save_steps,
            gradient_accumulation_steps=params.gradient_accumulation_steps,
        )
        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([config.root_dir])
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_path=config.model_path,
            tokenizer_path=config.tokenizer_path,
            metric_file_name=config.metric_file_name,
        )
        return model_evaluation_config
```

- Update components, also create a new python file (model_evaluation.py) inside the src/textSummarizer/components for this step

```python
## src/textSummarizer/components/model_evaluation.py

import torch
import evaluate
import pandas as pd
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.textSummarizer.logging import logger
from src.textSummarizer.entity import ModelEvaluationConfig

class ModelEvaluation:

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initialized ModelEvaluation on device: {self.device}")
        self.rouge_metric = evaluate.load("rouge")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(
            self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        self.dataset = load_from_disk(self.config.data_path)
        logger.info(f"Loaded model from {self.config.model_path} and dataset from {self.config.data_path}")

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metric_on_test_ds(
        self,
        dataset,
        metric,
        model,
        tokenizer,
        device,
        batch_size=1,
        column_text="article",
        column_summary="highlights",
    ):
        model.eval()  # Set model to evaluation mode

        # Split dataset into batches
        article_batches = list(
            self.generate_batch_sized_chunks(dataset[column_text], batch_size)
        )
        target_batches = list(
            self.generate_batch_sized_chunks(dataset[column_summary], batch_size)
        )

        with torch.no_grad():  # Disable gradient calculation
            for article_batch, target_batch in zip(article_batches, target_batches):
                inputs = tokenizer(
                    article_batch,
                    max_length=256,  # Max input length
                    truncation=True,  # Truncate if too long
                    padding="max_length",  # Pad to max length
                    return_tensors="pt",  # Return PyTorch tensors
                )

                summaries = model.generate(
                    input_ids=inputs["input_ids"].to(device),  # Move inputs to device
                    attention_mask=inputs["attention_mask"].to(device),
                    max_new_tokens=128,  # Max tokens for output
                    num_beams=1,  # Beam search width
                    do_sample=False,  # Deterministic generation
                    use_cache=True,  # Use past key values for speed
                )

                decoded_summaries = tokenizer.batch_decode(
                    summaries,
                    skip_special_tokens=True,  # Decode outputs to text
                )

                metric.add_batch(
                    predictions=decoded_summaries,  # Add generated summaries
                    references=target_batch,  # Add reference summaries
                )

        return metric.compute()  # Compute final metric (e.g., ROUGE, BLEU)

    def evaluate(self):
        dataset = load_from_disk(self.config.data_path)
        rouge_metric = self.rouge_metric
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        score = self.calculate_metric_on_test_ds(
            dataset=dataset["validation"],
            metric=rouge_metric,
            device=self.device,
            model=self.model,
            tokenizer=self.tokenizer,
            batch_size=2,
            column_text="dialogue",
            column_summary="summary",
        )
        rouge_dict = {name: score[name] for name in rouge_names}
        df = pd.DataFrame(rouge_dict, index=["flan-t5-small-finetuned"])
        logger.info("Saving evaluation metrics to CSV...")
        df.to_csv(self.config.metric_file_name, index=True)
        logger.info(f"Evaluation metrics saved successfully at {self.config.metric_file_name}")
```

- Create our final stage for our pipeline (stage4_model_evaluation.py)

```python
## src/textSummarizer/pipeline/stage4_model_evaluation.py

from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.components.model_evaluation import ModelEvaluation
from src.textSummarizer.logging import logger

class ModelEvaluationTrainingPipeline:

    def __init__(self):
        pass

    def initiate_model_evaluation(self):
        try:
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            model_evaluation.evaluate()
        except Exception as e:
            logger.exception(e)
            raise e
```

> Now, lets test if everyting is working fine or not

- Update main.py. Copy paste the below code to main.py

```python
## main.py

from src.textSummarizer.logging import logger
from src.textSummarizer.pipeline.stage1_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from src.textSummarizer.pipeline.stage2_data_transformation import (
    DataTransformationTrainingPipeline,
)
from src.textSummarizer.pipeline.stage3_model_trainer import (
    ModelTrainerTrainingPipeline,
)
from src.textSummarizer.pipeline.stage4_model_evaluation import (
    ModelEvaluationTrainingPipeline,
)


STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.initiate_data_ingestion()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(f"Error in stage {STAGE_NAME}: {e}")
    raise e

STAGE_NAME = "Data Transformation Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.initiate_data_transformation()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(f"Error in stage {STAGE_NAME}: {e}")
    raise e

STAGE_NAME = "Model Trainer Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    model_trainer = ModelTrainerTrainingPipeline()
    model_trainer.initiate_model_trainer()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(f"Error in stage {STAGE_NAME}: {e}")
    raise e

STAGE_NAME = "Model Evaluation Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    model_evaluation = ModelEvaluationTrainingPipeline()
    model_evaluation.initiate_model_evaluation()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(f"Error in stage {STAGE_NAME}: {e}")
    raise e
```

- Delete the artifacts folder if it already exists in your project folder

```text
.
├── app.py
├── artifacts   <---------------- # Delete this, if exists
├── config
├── data
├── Dockerfile
├── .git
├── .github
├── .gitignore
├── LICENSE
├── logs
├── main.py
├── params.yaml
├── README.md
├── requirements.txt
├── research
├── setup.py
├── src
├── template.py
└── venv
```

- Run main.py

```sh
(/home/siddhu/Desktop/Text-Summarizer-With-HF/venv) siddhu@ubuntu:~/Desktop/Text-Summarizer-With-HF$ python main.py
[2026-02-01 11:59:30,408: INFO: main: >>>>>> Stage Data Ingestion Stage started <<<<<<]
[2026-02-01 11:59:30,412: INFO: common: YAML file 'config/config.yaml' read successfully.]
[2026-02-01 11:59:30,413: INFO: common: YAML file 'params.yaml' read successfully.]
[2026-02-01 11:59:30,414: INFO: common: Directory 'artifacts' created successfully or already exists.]
[2026-02-01 11:59:30,414: INFO: common: Directory 'artifacts/data_ingestion' created successfully or already exists.]
[2026-02-01 11:59:36,142: INFO: data_ingestion: File downloaded successfully!]
[2026-02-01 11:59:36,238: INFO: data_ingestion: File extracted successfully at artifacts/data_ingestion]
[2026-02-01 11:59:36,238: INFO: main: >>>>>> Stage Data Ingestion Stage completed <<<<<<]
[2026-02-01 11:59:36,238: INFO: main: >>>>>> Stage Data Transformation Stage started <<<<<<]
[2026-02-01 11:59:36,242: INFO: common: YAML file 'config/config.yaml' read successfully.]
[2026-02-01 11:59:36,243: INFO: common: YAML file 'params.yaml' read successfully.]
[2026-02-01 11:59:36,244: INFO: common: Directory 'artifacts' created successfully or already exists.]
[2026-02-01 11:59:36,244: INFO: common: Directory 'artifacts/data_transformation' created successfully or already exists.]
[2026-02-01 11:59:37,741: INFO: data_transformation: Initialized DataTransformation with tokenizer: google/flan-t5-small]
Map: 100%|██████████████████████████████████████████████████████| 14731/14731 [00:02<00:00, 5190.94 examples/s]
Map: 100%|██████████████████████████████████████████████████████████| 818/818 [00:00<00:00, 4855.11 examples/s]
Map: 100%|██████████████████████████████████████████████████████████| 819/819 [00:00<00:00, 5728.22 examples/s]
[2026-02-01 11:59:41,268: INFO: data_transformation: Saving transformed dataset to disk...]
Saving the dataset (1/1 shards): 100%|████████████████████████| 14731/14731 [00:00<00:00, 276054.72 examples/s]
Saving the dataset (1/1 shards): 100%|█████████████████████████████| 818/818 [00:00<00:00, 99286.40 examples/s]
Saving the dataset (1/1 shards): 100%|█████████████████████████████| 819/819 [00:00<00:00, 86368.52 examples/s]
[2026-02-01 11:59:41,347: INFO: data_transformation: Dataset saved successfully at artifacts/data_transformation/summarizer_dataset]
[2026-02-01 11:59:41,377: INFO: main: >>>>>> Stage Data Transformation Stage completed <<<<<<]
[2026-02-01 11:59:41,378: INFO: main: >>>>>> Stage Model Trainer Stage started <<<<<<]
[2026-02-01 11:59:41,381: INFO: common: YAML file 'config/config.yaml' read successfully.]
[2026-02-01 11:59:41,383: INFO: common: YAML file 'params.yaml' read successfully.]
[2026-02-01 11:59:41,383: INFO: common: Directory 'artifacts' created successfully or already exists.]
[2026-02-01 11:59:41,384: INFO: common: Directory 'artifacts/model_trainer' created successfully or already exists.]
[2026-02-01 11:59:41,423: INFO: model_trainer: Initialized ModelTrainer on device: cuda]
[2026-02-01 11:59:46,905: INFO: model_trainer: Starting model training...]
{'loss': 2.0079, 'grad_norm': 2.6245455741882324, 'learning_rate': 9e-07, 'epoch': 0.2}                        
{'loss': 2.0806, 'grad_norm': 3.358659267425537, 'learning_rate': 1.9e-06, 'epoch': 0.39}                      
{'loss': 2.0031, 'grad_norm': 2.7413535118103027, 'learning_rate': 2.9e-06, 'epoch': 0.59}                     
{'loss': 1.9963, 'grad_norm': 2.9518542289733887, 'learning_rate': 3.9e-06, 'epoch': 0.78}                     
{'loss': 1.8982, 'grad_norm': 2.6860415935516357, 'learning_rate': 4.9000000000000005e-06, 'epoch': 0.98}      
{'train_runtime': 56.4117, 'train_samples_per_second': 14.518, 'train_steps_per_second': 0.922, 'train_loss': 2.004348736542922, 'epoch': 1.0}
100%|██████████████████████████████████████████████████████████████████████████| 52/52 [00:56<00:00,  1.08s/it]
[2026-02-01 12:00:43,643: INFO: model_trainer: Model training completed successfully.]
[2026-02-01 12:00:43,643: INFO: model_trainer: Saving model and tokenizer...]
[2026-02-01 12:00:44,206: INFO: model_trainer: Model and tokenizer saved at artifacts/model_trainer]
[2026-02-01 12:00:44,223: INFO: main: >>>>>> Stage Model Trainer Stage completed <<<<<<]
[2026-02-01 12:00:44,223: INFO: main: >>>>>> Stage Model Evaluation Stage started <<<<<<]
[2026-02-01 12:00:44,225: INFO: common: YAML file 'config/config.yaml' read successfully.]
[2026-02-01 12:00:44,226: INFO: common: YAML file 'params.yaml' read successfully.]
[2026-02-01 12:00:44,226: INFO: common: Directory 'artifacts' created successfully or already exists.]
[2026-02-01 12:00:44,228: INFO: common: Directory 'artifacts/model_evaluation' created successfully or already exists.]
[2026-02-01 12:00:44,228: INFO: model_evaluation: Initialized ModelEvaluation on device: cuda]
[2026-02-01 12:00:47,166: INFO: model_evaluation: Loaded model from artifacts/model_trainer/t5_model and dataset from artifacts/data_transformation/summarizer_dataset]
[2026-02-01 12:02:38,302: INFO: rouge_scorer: Using default tokenizer.]
[2026-02-01 12:02:38,714: INFO: model_evaluation: Saving evaluation metrics to CSV...]
[2026-02-01 12:02:38,715: INFO: model_evaluation: Evaluation metrics saved successfully at artifacts/model_evaluation/metrics.csv]
[2026-02-01 12:02:38,715: INFO: main: >>>>>> Stage Model Evaluation Stage completed <<<<<<]
```
*Again, You can encounter OOM (OutOfMemory) Error.*

> Commit and push the changes to github

```sh
git add .
git commit -m 'Data Evaluation Modularization Completed'
git push origin main
```

---
