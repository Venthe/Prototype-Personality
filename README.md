## Directory structure

├──applications           # High-level applications
├───build_tools           # Scripts, Docker, CI/CD, etc.
│   └───bazel             # Bazel-specific configurations
├───docs                  # "documentation" shortened to follow convention
├───libraries             # Shared libraries and utilities
│   ├───audio_processing  # Example shared library name
│   ├───ml_utils          # Example shared library name
│   └───common            # Common utilities shared across libraries
├───resources
│   ├───models            # Pre-trained models or checkpoints
│   └───training_data     # Datasets for training


## Neuro capabilities

u18:
* Autonomous discord DM's, call invitations
* TTS speed manipulation

Future:
* Toggle control
* Call kicking
* Calling other constructs
* singing
* tweeting
* Soundboard
* Voice changer

## Installation

pip install toml-cli
pip install pyenv-win --target $env:USERPROFILE\\.pyenv
[System.Environment]::SetEnvironmentVariable('PYENV',$env:USERPROFILE + "\.pyenv\pyenv-win\","User")
[System.Environment]::SetEnvironmentVariable('PYENV_ROOT',$env:USERPROFILE + "\.pyenv\pyenv-win\","User")
[System.Environment]::SetEnvironmentVariable('PYENV_HOME',$env:USERPROFILE + "\.pyenv\pyenv-win\","User")
[System.Environment]::SetEnvironmentVariable('path', $env:USERPROFILE + "\.pyenv\pyenv-win\bin;" + $env:USERPROFILE + "\.pyenv\pyenv-win\shims;" + [System.Environment]::GetEnvironmentVariable('path', "User"),"User")

Per project:

pyenv install 3.9.8
pyenv local 3.9.8  # Activate Python 3.9 for the current project
poetry install
pyenv install --skip-existing
pyenv local
poetry install