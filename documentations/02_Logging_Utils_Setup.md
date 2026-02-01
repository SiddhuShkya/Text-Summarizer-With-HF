
### 2. Logging & Utils Common Functionalities

In this section we are going to implement the common utility functions and logging setup required across the project. These components will help us:

- Track training progress
- Save model checkpoints
- Log evaluation metrics
- Handle reusable helper functions
- Improve reproducibility

2.1 Let's begin by implementing our basic logging functionalites. 

> For logging we will be using the file shown below 

```text
├── src
│   └── textSummarizer
│       ├── components
│       ├── config
│       ├── constants
│       ├── entity
│       ├── __init__.py
│       ├── logging
│       │   └── __init__.py <------------ # This file 
│       ├── pipeline
│       └── utils
```

> Copy paste the below python script to the above mentioned file

```python
## src/logging/__init__.py

import os
import sys
import logging

log_dir = "logs"
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_file_path = os.path.join(log_dir, "text_summarizer.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("summarizer_logger")
```

2.2 Verify if the logging has been implemented or not

> Copy paste the below python code to main.py

```python
## main.py

from src.textSummarizer.logging import logger

logger.info("Starting the text summarization process.")
logger.info("Text summarization process completed successfully.")
```
> Run the python main.py file, you should see output similar to the below one

```sh
siddhu@ubuntu:~/Desktop/Text-Summarizer-With-HF$ python main.py 
[2026-01-23 12:34:00,557: INFO: main: Starting the text summarization process.]
[2026-01-23 12:34:00,557: INFO: main: Text summarization process completed successfully.]
```

> In our case, its working fine

2.3 Now, let's go ahead and implement some of the common functionalities we will be using for the project

> For defining our common utilities/functions we will be using the file shown below 

```text
├── src
│   └── textSummarizer
│       ├── components
│       ├── config
│       ├── constants
│       ├── entity
│       └── utils
│       │    └── __init__.py  
│       │    └── common.py   <------------ # This file 
│       ├── __init__.py
│       ├── logging
│       ├── pipeline
```

> Copy paste the below python script to the above mentioned file

```python
## src/utils/common.py

import os
from box.exceptions import BoxValueError
import yaml
from src.textSummarizer.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns its contents as a ConfigBox object.

    Args:
        path_to_yaml (Path): The path to the YAML file.

    Returns:
        ConfigBox: The contents of the YAML file as a ConfigBox object.
    """
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file '{path_to_yaml}' read successfully.")
            return ConfigBox(content)
    except BoxValueError as box_error:
        logger.error(
            f"BoxValueError while converting YAML content to ConfigBox: {box_error}"
        )
        raise box_error
    except Exception as e:
        logger.error(f"Error reading YAML file '{path_to_yaml}': {e}")
        raise e

@ensure_annotations
def create_directories(path_to_directories: list[Path]) -> None:
    """Creates directories if they do not exist.

    Args:
        path_to_directories (list[Path]): A list of directory paths to create.
    """
    for path in path_to_directories:
        try:
            os.makedirs(path, exist_ok=True)
            logger.info(f"Directory '{path}' created successfully or already exists.")
        except Exception as e:
            logger.error(f"Error creating directory '{path}': {e}")
            raise e
```

📌 **Important Notes**

- Why use ConfigBox ?

> Without ConfigBox:

```python
dict_info = {"name": "Siddhartha", "age": 22, "city": "Bhaktapur"}
print(dict_info["name"])
print(dict_info.name)
```
```sh
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
Cell In[6], line 3
      1 dict_info = {"name": "Siddhartha", "age": 22, "city": "Bhaktapur"}
      2 print(dict_info["name"])
----> 3 print(dict_info.name)

AttributeError: 'dict' object has no attribute 'name'
```

> With ConfigBox:

```python
from box import ConfigBox

dict_info = ConfigBox({"name": "Siddhartha", "age": 22, "city": "Bhaktapur"})
print(dict_info['name'])
print(dict_info.name)
```

```sh
Siddhartha
Siddhartha
```

- Why @ensure_annotations?

> Without @ensure_annotations:

```python
def multiply(a :int, b :int) -> int:
    return a * b

result = multiply(3, "4")
print(result)
```
```sh
444
```

> With @ensure_annotations:

```python
from ensure import ensure_annotations

@ensure_annotations
def multiply(a: int, b: int) -> int:
    return a * b

result = multiply(3, "4")
print(result)
```
```sh
EnsureError: Argument b of type <class 'str'> to <function multiply at 0x7ddb4806a950> does not match annotation type <class 'int'>
```

Throws an Error, making sure that only the correct values are passed and returned.

---
