Python_package_in_conda
License Python Version Conda Version

name: Python Model Tests
import pandas as pd
import numpy as np
import lightgbm as lgb
import pytest

def test_model_prediction():
    # 1. Setup Mock Data
    feature_names = ['feat1', 'feat2']
    X = np.array([[10, 10], [1, 1], [10, 11]]) # Row 2 is distinct
    y = np.array([0, 1, 0])
    X_df = pd.DataFrame(X, columns=feature_names)

    # 2. Initialize with "Tiny Data" settings
    test_params = {
        "objective": "binary",
        "min_data_in_leaf": 1,
        "min_child_samples": 1,
        "n_estimators": 10,
        "deterministic": True,
        "verbose": -1
    }
    
    model = lgb.LGBMClassifier(**test_params)
    model.fit(X_df, y)

    # 3. Debug/Predict logic
    probs = model.predict_proba(X_df)[:, 1]
    print(f"\nClass 1 Probabilities: {probs}")

    # Use a threshold or direct predict
    predictions = (probs >= 0.5).astype(int)
    expected_predictions = np.array([0, 1, 0])

    # 4. Enhanced Assertion for Copilot Debugging
    assert np.array_equal(predictions, expected_predictions), (
        f"Prediction mismatch! Expected {expected_predictions} but got {predictions}. "
        f"Probabilities were: {probs}"
    )



conda install -c conda-forge pytest
pip install -e .

name: my_env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytest          # <--- Add it here
  - pip
  - pip:
    - -e .          # Installs your local package in editable mode
conda env update -f environment.yml


on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout Code
      uses: actions/checkout@v4

    - name: Set up Python 3.10
      uses: actions/setup-python@v5
      with:
        python-version: "3.10"
        # This automatically caches your pip dependencies
        cache: 'pip' 

    - name: Install Dependencies
      run: |
        python -m pip install --upgrade pip
        # It is best practice to use a requirements file for caching
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
        pip install pytest pandas lightgbm joblib

    - name: Run Fast Unit Tests (Mocks)
      run: pytest -m fast

    - name: Run Integration Tests (Real Model)
      run: pytest -m integration


Features
🚀 Feature 1: Brief description
📊 Feature 2: Brief description
🔧 Feature 3: Brief description
⚡ Feature 4: Brief description
Table of Contents
Installation
Quick Start
Usage
Configuration
Documentation
Contributing
License
Contact
Installation
Prerequisites
Python 3.8 or higher
Conda (Miniconda or Anaconda)
Using Conda (Recommended)
Option 1: Install from Conda-Forge
conda install -c conda-forge package-name

jobs:
  name: oasis
channels:
  - conda-forge
  - defaults
dependencies:
  - python>=3.8
  - rdflib=6.1.1
  - pip
  - pip:
    - -e .


      # mv environment.yml OASIS/
git add OASIS/environment.yml
git commit -m "Place environment.yml in OASIS folder"

# mv environment.yml OASIS/
git add OASIS/environment.yml
git commit -m "Place environment.yml in OASIS folder"
git push

# Install dependencies
  run: conda env update --file ../environment.yml --name base
  working-directory: SunShineHead/OASIS/

import pandas as pd
# Define data as a dictionary
data = {
   'Name': ['Alice', 'Bob', 'Charlie'],
   'Age': [25, 30, 35],
   'Gender': ['Female', 'Male', 'Male']
}
# Create DataFrame
df = pd.DataFrame(data)
# Print DataFrame
print(df)

-  The Ontology for Agents, Systems and Integration of Services: recent advancements of OASIS. Giampaolo Bella, Domenico Cantone, Marianna Nicolosi-Asmundo, Daniele Francesco Santamaria. Proceedings of WOA 2022- 23nd Workshop From Objects to Agents, 1–2, September 2022, Genova, Italy, CEUR Workshop Proceedings, ISSN 1613-0073, Vol. 3261, pp.176--193.
-  Blockchains through ontologies: the case study of the Ethereum ERC721 standard in OASIS. Giampaolo Bella, Domenico Cantone, Cristiano Longo, Marianna Nicolosi-Asmundo, Daniele Francesco Santamaria. In D. Camacho et al. (eds.), Intelligent Distributed Computing XIV, Studies in Computational Intelligence 1026, Chapter 23,  pp. 249-259.
-  Semantic Representation as a Key Enabler for Blockchain-Based Commerce. Giampaolo Bella, Domenico Cantone, Cristiano Longo, Marianna Nicolosi-Asmundo and Daniele Francesco Santamaria. In: K. Tserpes et al. (Eds.): GECON 2021, Lecture Note in Computer Science, Vol. 13072, pp. 191–198, Springer, 2021.
-  Ontological Smart Contracts in OASIS: Ontology forAgents, Systems, and Integration of Services. Domenico Cantone, Carmelo Fabio Longo, Marianna Nicolosi-Asmundo, Daniele Francesco Santamaria, Corrado Santoro. In D. Camacho et al. (eds.), Intelligent Distributed Computing XIV, Studies in Computational Intelligence 1026, Chapter 22, pp. 237-247.
-  Towards an Ontology-Based Framework for a Behavior-Oriented Integration of the IoT. Domenico Cantone, Carmelo Fabio Longo, Marianna Nicolosi-Asmundo, Daniele Francesco Santamaria, Corrado Santoro. Proceedings of the 20th Workshop From Objects to Agents, 26-28 June, 2019, Parma, Italy, CEUR Workshop Proceedings, ISSN 1613-0073, Vol. 2404, pp. 119--126.
- Giampaolo Bella, Gianpietro Castiglione, Daniele Francesco Santamaria. A Behaviouristic Approach to Representing Processes and Procedures in the OASIS 2 Ontology (2023) CEUR Workshop Proceedings 3637.
-  Giampaolo Bella, Domenico Cantone, Carmelo Fabio Longo, Marianna Nicolosi Asmundo, Daniele Francesco Santamaria. The ontology for agents, systems and integration of services: OASIS version 2.
Intelligenza Artificiale 2023, 17(1), pp. 51–62.
-  Giampaolo Bella, Domenico Cantone, Gianpietro Castiglione, Marianna Nicolosi Asmundo, Daniele Francesco Santamaria. A behaviouristic semantic approach to blockchain-based e-commerce. Semantic Web
(2024), 15 (5), pp. 1863 - 1914. DOI: 10.3233/SW-243543.
-  Giamapolo Bella, Domenico Cantone, Marianna Nicolosi Asmundo, Daniele Francesco Santamaria. Towards a semantic blockchain: A behaviouristic approach to modelling Ethereum. (2024) Applied Ontology, 19 (2), pp. 143 - 180, DOI: 10.3233/AO-230010.

## Licensing information
Copyright (C) 2021.  Giampaolo Bella, Domenico Cantone, Marianna Nicolosi Asmundo, Daniele Francesco Santamaria. This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version. This program is distributed in the hope that it will be useful,  but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.  You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Python BehaviorManager module
This program permits the creation of swallacehuang-debug agents.

## Requirements </br>
   - Python interpreter version 3.7 or greater.
   - RDFLib version 6.1.1.

## Generating new agents and agent behaviors </br>

In order to generate new OASIS behaviors you should

A)  Create three RDFLib ontology objects, one for the ontology hosting the agent behaviors, one for the ontology hosting the agent templates, one for the ontology hosting data.

Create a BehaviorManager object by typing: </br>
      
      b = BehaviorManager(ontology, namespace, ontologyURL, ontologyTemplate, namespaceTemplate, templateURL)
      
   where:  </br>
   - "ontology" is the ontology containing the agent behavior.
   - "namespace" is namespace of "ontology". You can use "None" if "xml:base" is already defined in the ontology.
   - "ontologyURL" is the URL of the ontology.
   - "ontologyTemplate" is the namespace of the ontology containing the behavior template.
   - "namespaceTemplate" is namespace of "ontologyTemplate". You can use "None" if "xml:base" is already defined in the ontology.
   - "templateURL" is the URL of the ontology containing the behavior template.
   
from oasis_manager import BehaviorManager

B) from oasis_manager import BehaviorManager
    namespaceTemplate="http://example.org/template#",
    templateURL="http://url-to-template.owl"
)

      
      b.createAgentTemplate(agentTemplateName)
      
   where:  </br>   
   - "ontologyTemplateName" is the name of the agent template name. </br>
   
   Then, create a new agent template behavior by typing: </br>


      from oasis_manager import BehaviorManager

b = BehaviorManager(
    ontology=my_ontology_obj, 
    namespace="http://example.org/oasis#", 
    ontologyURL="http://url-to-source.owl",
    ontologyTemplate=template_obj,
    namespaceTemplate="http://example.org/template#",
    templateURL="http://url-to-template.owl"
)

                                     [MyTemplateTaskOperator, action], 
                                     [MyTemplateOperatorArgument, actionArgument],
                                     [
                                        [MyTemplateTaskObject, taskObjectProperty, objectTemplate]
                                     ], 
                                     [ 
                                        [MyTemplateInput1, taskInputProperty, input1]
                                     ], 
                                     [ 
                                        ["MyTemplateOutput1", taskOutputProperty, output1]
                                     ])

        
   where:
   - "MyTemplateBehavior" is the entity name of the behavior template. 
   - "MyTemplateGoal" is the entity name of the goal template.
   - "MyTemplateTask" is the entity name of the task template.
   - "MyTemplateTaskOperator" and "action" are, respectively, the entity name of the task operator  and the operator action as defined in OASIS-ABox.
   - "MyTemplateOperatorArgument" and "actionArgument" are, respectively, the entity name of the operator argument and the operator argument as defined in OASIS-ABox.
   - A list of elements of the form:
     - [MyTemplateTaskObject, taskObjectProperty, objectTemplate] </br>
       where: </br>
         - "MyTemplateTaskObject" is the entity name of the task object.
         - "taskObjectProperty" is the either "refersAsNewTo" or "refersExactlyTo".
         - "objectTemplate" is the element associated to the task object.
   - A list of elements of the form:
     - [MyTemplateInput1, taskInputProperty, input1] </br>
       where: </br> 
        - "MyTemplateInput1" is the entity name of the input.
        - "taskInputProperty" is the either "refersAsNewTo" or "refersExactlyTo".
        - "input" is the element associated to the input element.   
   - A list of elements of the form:
     - [MyTemplateOutput1, taskOutputProperty, output1] </br>
       where: </br> 
       - "MyTemplateOutput1" is the entity name of the output.
       - "taskOutputProperty" is either "refersAsNewTo" or "refersExactlyTo".
       - "output" is the element associated with the output element.  
     
 - Connect the behavior with the related template
 
name: myenv
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.10
  - numpy
  - pandas
  - pip
  - pip:
    - requests
conda activate oasis
       b.connectAgentTemplateToBehavior(MyAgentBehaviorTemplate, MyTemplateBehavior)
       
   where: </br>
   - "MyAgentBehaviorTemplate" is the the behavior template created as described above.
   - "MyTemplateBehavior" is the behavior created as above.

C) Create a new agent and a new behavior eventually related with a behavior template.
   
   Create a new agent by typing:
              
      b.createAgent("MyAgent")
  
   where:
   - "MyAgent" is the entity name of the agent.
   
   Create a new agent behavior and eventually connect it to its template by typing
   
      b.createAgentBehavior(OasisBehavior, OasisGoal, MyAgentTask,
                            [OasisTaskOperator, action],
                            [OasisOperatorArgument, actionArgument],
                         [
                            [MyAgentTaskObject, taskObjectProperty, agentobject1]
                         ],
                         [
                             [MyAgentInput1, taskInputProperty, agentinput1]
                         ],
                         [
                             [MyAgentOutput1, taskInputProperty, agentoutput1]
                         ],
                         [
                           MyTemplateTask,
                          [
                              [MyAgentTaskObject, MyTemplateTaskObject]
                          ],
                          [
                              [MyAgentInput1, MyTemplateInput1]
                          ],
                          [
                              [MyAgentOutput1, MyTemplateOutput1]
                          ]
                         ])

   where:
   - "MyAgentBehavior" is the entity name of the behavior. 
   - "MyAgentGoal" is the entity name of the goal.
   - "MyAgentTask" is the entity name of the task.
   - "MyAgentTaskOperator" and "action" are, respectively, the entity name of the task operator  and the operator action as defined in OASIS-ABox.
   - "MyAgentOperatorArgument" and "actionArgument" are, respectively, the entity name of the operator argument and the operator argument as defined in OASIS-ABox.
   - A list of elements of the form:
        - [MyAgentTaskObject, taskObjectProperty, agentobject1] </br>
           where: </br>
           - "MyAgentTaskObject" is the entity name of the task object.
           - "taskObjectProperty" is the either "refersAsNewTo" or "refersExactlyTo".
           - "agentobject1" is the element associated to the task object.
   - A list of elements of the form:
        - [MyAgentInput1, taskInputProperty, agentinput1] </br>
          where: </br> 
          - "MyAgentInput1" is the entity name of the input.
          - "taskInputProperty" is the either "refersAsNewTo" or "refersExactlyTo".
          - "agentinput1" is the element associated to the input element.   
   - A list of elements of the form:
        - [MyAgentOutput1, taskOutputProperty, agentoutput1]</br>
          where: </br> 
          - "MyAgentOutput1" is the entity name of the output.
          - "taskOutputProperty" is the either "refersAsNewTo" or "refersExactlyTo".
          - "agentoutput1" is the element associated to the output element. 
   - Eventually a list of elements mapping from the agent to the template:
       - "MyTemplateTask" is the task object of the behavior template.
       - A list of elements of the form:
            - ["MyAgentTaskObject", "MyTemplateTaskObject"] </br>
              where:</br>
                 -  "MyAgentTaskObject", "MyTemplateTaskObject" represent the entity name of the agent task object  and the entity of the task object template, respectively.
       - A list of elements of the form:  
            - ["MyAgentInput1", "MyTemplateInput1"] </br>
              where:</br>
                -  "MyAgentInput1", "MyTemplateInput1" represent the entity name of the agent input and the agent input template, respectively.
       - A list of elements of the form:  
           - ["MyAgentOutput1", "MyTemplateOutput1"] </br>
           where:</br>
               -  "MyAgentOutput1", "MyTemplateOutput1" represent the entity name of the agent output and the agent output template, respectively.
  - Connect the created behavior to its agent by typing:
     
        b.connectAgentToBehavior("MyAgent", "MyAgentBehavior")
    
    where: </br>
    - "MyAgent" and "MyAgentBehavior" are, respectively, the agent and the agent behavior.
    
       
 D) Generate a new action and connect it to the related behavior by typing
 
       b.createAgentAction(MyAgent, planExecution, executionGoal, executionTask,
                         [executionOperator, action],
                         [executionArgument, argument],
                         [
                             [executionObject, taskObjectProperty, executionobject1]
                         ],
                         [
                             [executionInput1, inputProp, executioninput1]
                         ],
                         [
                             [executionOutput1, outputProp, executionOutput1]
                         ],
                         [
                           MyAgentTask,
                          [
                              [executionObject, MyAgentTaskObject]
                          ],
                          [
                              [executionInput1, MyAgentInput1]
                          ],
                          [
                              [executionOutput1, MyAgentOutput1]
                          ]
                         ])
                         
  where:</br>
  - "MyAgent" is the entity name of the agent responsible for the execution of the action.
  - "planExecution" is the entity name of the plan execution.
  - "executionGoal" is the entity name of the goal execution.
  - "executionTask" is the entity name of the task execution.
  - A list of element of the form:
      - [executionOperator, action] </br>
        where:</br>
         - "executionOperator" is the name of the task operator.
         - "action" is name of the action as defined in OASIS-ABox.
      - [executionArgument, argument] </br>
        where:</br>
        - "executionArgument" is the name of the task argument.
        - "argument" is the name of the argument as defined in OASIS-ABox.
      - A list of element of the form:  
        - [executionObject, taskObjectProperty, executionobject1] </br>
          where: </br>
          - "executionObject" is the entity name of the task execution object.
          - "taskObjectProperty" is  either "refersAsNewTo" or "refersExactlyTo".
          - "executionobject1" is the element associated with the task execution object.     
      - A list of elements of the form:
        - [executionInput1, inputProp, executioninput1] </br>
          where: </br>
             - "executionInput1" is the entity name of task input.
             - "inputProp" is either "refersAsNewTo" or "refersExactlyTo".
             - "executioninput1" is the element associated with the task input.
       - A list of elements of the form:
         - [executionOutput1, outputProp, executionOutput1] </br>
           where: </br>
             - "executionOutput1" is the entity name of task output.
             - "outputProp" is either "refersAsNewTo" or "refersExactlyTo".
             - "executionOutput1" is the element associated with the task output.
       - A list of elements mapping from the agent action to the agent behavior:
          - "MyAgentTask" is the task  of the agent behavior.
       - A list of elements of the form:
          - [executionObject, MyAgentTaskObject] </br>
            where: </br>
              - "executionObject", "MyAgentTaskObject" represent the entity name of the  task execution  and the entity name of the task object of the agent behavior, respectively.
       - A list of elements of the form:  
         - [executionInput1, MyAgentInput1] </br>
           where:</br>
              - "executionInput1", "MyAgentInput1" represent the entity name of the action input and the agent behavior input , respectively.
        - A list of elements of the form:  
          - [executionOutput1, MyAgentOutput1] </br>
            where:</br>
              -  "executionOutput1", "MyAgentOutput1" represent the entity name of the action output and the agent behavior output, respectively.

jobs:
  build:
    runs-on: ubuntu-latest


# Create a new conda environment
conda create -n myenv python=3.9%

# Create Environment
  run: conda env create --file environment.yml
run: conda env update --file environment.yml --name base --verbose

# mv environment.yml OASIS/
git add OASIS/environment.yml
git commit -m "Place environment.yml in OASIS folder"

# mv environment.yml OASIS/
git add OASIS/environment.yml
git commit -m "Place environment.yml in OASIS folder"
git push

- name: Path Debugger
  run: |- name: Update Conda
  run: conda env update --file .Swallacehuang-Debug/OASIS/environment.yml --name base

    pwd
    find . -maxdepth 3 -name "environment.yml"


# Install dependencies
  run: conda env update --file ../environment.yml --name base
  working-directory: .Represitories/OASIS

# Activate the environment
conda activate myenv

pytest --cov=OASIS --cov-fail-under=80
[pytest]
addopts = --cov=OASIS --cov-report=term-missing --cov-fail-under=80
markers =
    fast: simple unit tests using mocks
    integration: heavy tests loading the real model

    - name: Run Tests with Coverage Gate
      run: |
        # This will fail the build if coverage is < 80%
        pytest --cov=OASIS --cov-fail-under=80




# Install the package
conda install -c conda-forge package-name

- name: Setup Conda
  uses: conda-incubator/setup-miniconda@v3
  with:
    activate-environment: oasis-env
    environment-file: .Swallacehuang-Debug/OASIS/environment.yml # Point directly to the file
    auto-activate-base: false
    conda-solver: libmamba


my_conda_project/
├── conda-recipe/
│   └── meta.yaml          # Conda build instructions
├── src/
│   └── my_package/
│       ├── __init__.py
│       └── calculator.py  # Logic to test
├── tests/
│   ├── conftest.py        # Shared fixtures
│   └── test_calculator.py # Test cases
├── pyproject.toml         # Build system & pytest config
└── environment.yml        # Development environment

def add(a: float, b: float) -> float:
    return a + b

def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b
import pytest
from my_package.calculator import add, divide

@pytest.mark.parametrize("a, b, expected", [
    (1, 2, 3),
    (-1, 1, 0),
    (0.1, 0.2, 0.3),  # Note: floating point logic might need pytest.approx
])
def test_add(a, b, expected):
    assert add(a, b) == pytest.approx(expected)

def test_divide_error():
    with pytest.raises(ValueError, match="Cannot divide by zero."):
        divide(10, 0)

conda env create -f environment.yml
conda activate my_env

pytest --verbose




# Clone the repository
git clone https://github.com/sunshinehead/oasis projectname.Python_package_in_conda
cd project-name

# Create environment from git add environment.yml
git commit -m "Add environment.yml"
git push

# Setup Environment
  uses: conda-incubator/setup-miniconda@v3
  with:
    activate-environment: oasis  # Don't use 'base'
    environment-file: environment.yml
    conda-solver: libmamba      # Faster and gives better error messages

# name: base
dependencies:
  - python=3.10
  - numpy
  - pandas
  - scikit-learn


# Activate the environment.pyml
conda activate project-env

pip install package-name

# Clone the repository
git clone https://github.com/username/project-name.git
cd python package in conda

# Create conda environment
conda env git mv Environment.yml environment.yml
git commit -m "Fix filename case"
git push

# Install in editable mode
pip install -e .

# import python_package_in_conda

name: Python Model Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout Code
      uses: actions/checkout@v4

    - name: Set up Python 3.10
      uses: actions/setup-python@v5
      with:
        python-version: "3.10"
        # This automatically caches your pip dependencies
        cache: 'pip' 

    - name: Install Dependencies
      run: |
        python -m pip install --upgrade pip
        # It is best practice to use a requirements file for caching
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
        pip install pytest pandas lightgbm joblib

    - name: Run Fast Unit Tests (Mocks)
      run: pytest -m fast

    - name: Run Integration Tests (Real Model)
      run: pytest -m integration


# OASIS/
 ├── .github/
 ├── environment.yml   ← must be here if using current command
OASIS/
 ├── .github/
 ├── environment.yml   ← must be here if using current command
OASIS/
 ├── AO2022-addendum/
 │    └── environment.yml
- name: Update conda environment
  run: conda env update --file AO2022-addendum/environment.yml --name base
- uses: actions/checkout@v4
- name: Debug workspace
  run: |
    pwd
    ls -la
    find . -name "environment.yml"
- run: conda env create -f environment.yml -n oasis-env
- run: conda activate oasis-eng
# Update conda environment
  run: conda env update --file AO2022-addendum/environment.yml --name base
# actions/checkout@v4
# Debug workspace
  run: |
    pwd
    ls -la
    find . -name "environment.yml"
- run: conda env create -f environment.yml -n oasis-env
- run: conda activate oasis-env

# Example usage
result = package_name.main_function(parameter="value")
print(result)

from package_name import advanced_module

# More complex usage
config = {
    'option1': True,
    'option2': 'custom_value',
    'option3': 42
}

processor = advanced_module.Processor(**config)
results = processor.run()

steps:
  - name: Checkout Repository
    uses: actions/checkout@v4  # This brings your code into the runner

  - name: Update Conda Environment
    run: conda env update --file environment.yml --name base
- name: List files for debugging
  run: ls -R


# Basic command
package-name --help

# Example command
package-name process --input data.csv --output results.csv

settings:
  parameter1: value1
  parameter2: value2
  
options:
  debug: true
  verbose: false

from package_name import load_config

config = load_config('config.yaml')

# Checkout code
  uses: actions/checkout@v4

jobs:
  build:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: .Swallacehuang-debug/Oasis # Change this to your subfolder name


jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

# base
dependencies:
  - python=3.10
  - pip
  - numpy
  - pandas
  - scikit-learn

      # Setup Conda
        uses: conda-incubator/setup-miniconda@v3
        with:
          auto-update-conda: true

      # Install dependencies
  run: conda env update --file ../environment.yml --name base
  working-directory: .Swallacehuang-Debug/OASIS


# project-env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy>=1.21.0
  - pandas>=1.3.0
  - scipy>=1.7.0
  - matplotlib>=3.4.0
  - pytest>=6.2.0
  - pip
  - pip:
    - some-pip-only-package>=1.0.0

conda install -c conda-forge pytest
pip install -e .

name: my_env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytest          # <--- Add it here
  - pip
  - pip:
    - -e .          # Installs your local package in editable mode
conda env update -f environment.yml


project-name/
├── README.md
├── LICENSE
├── setup.py
├── environment.yml
├── requirements.txt
├── .gitignore
├── docs/
│   ├── index.md
│   └── examples.md
├── src/
│   └── package_name/
│       ├── __init__.py
│       ├── core.py
│       └── utils.py
├── tests/
│   ├── __init__.py
│   └── test_core.py
└── examples/
    └── example_notebook.ipynb

# Activate environment
conda activate project-env

# Run tests with pytest
pytest tests/

my_conda_project/
├── conda-recipe/
│   └── meta.yaml          # Conda build instructions
├── src/
│   └── my_package/
│       ├── __init__.py
│       └── calculator.py  # Logic to test
├── tests/
│   ├── conftest.py        # Shared fixtures
│   └── test_calculator.py # Test cases
├── pyproject.toml         # Build system & pytest config
└── environment.yml        # Development environment

def add(a: float, b: float) -> float:
    return a + b

def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b
import pytest
from my_package.calculator import add, divide

@pytest.mark.parametrize("a, b, expected", [
    (1, 2, 3),
    (-1, 1, 0),
    (0.1, 0.2, 0.3),  # Note: floating point logic might need pytest.approx
])
def test_add(a, b, expected):
    assert add(a, b) == pytest.approx(expected)

def test_divide_error():
    with pytest.raises(ValueError, match="Cannot divide by zero."):
        divide(10, 0)

conda env create -f environment.yml
conda activate my_env

pytest --verbose



# Run with coverage
pytest --cov=package_name tests/


This README includes:
- **Badges** for quick status visibility
- **Clear installation instructions** for Conda
- **Multiple installation options** (conda-forge, environment.yml, pip)
- **Quick start and usage examples**
- **Configuration guidelines**
- **Testing instructions**
- **Contributing guidelines**
- **Project structure**
- **Support and contact information**

- OASIS-MAN\Python\test\Test-BehaviorManager.py