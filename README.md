## What is Explainable-RL?
Explainable-RL is a Python package that provides a framework for explainable reinforcement learning (XRL), specifically for 
pricing and business decisions. It allows users to upload any pricing dataset and train a tabular RL agent to learn the optimal
pricing policy. It has been created with speed and memory requirements in mind, and is able to train agents on large, multidimensional 
datasets quickly. The package also provides a suite of explainability tools to help users understand the agent's decision-making process.

Full documentation can be found [here](https://explainable-rl.readthedocs.io/en/latest/). 

A full demo can be found in the onboarding Jupyter notebook, found in the project's GitHub.

## Installation

Installation is done by running:

    pip install explainable-rl


## Quick Guide on using the Sphinx Documentation
The documentation of the explainable-RL package first release has already been 
created and resided in ``explainable-RL/documentation/_build/html/<page_name>.html``.
It was built using the Sphinx documentation generator. Here, ``<page_name>`` refers to
the name of doc pages (e.g. index, library, src, ...). For an exhaustive list of names,
navigate to the ``_build/html`` folder and observe files with the ``.html`` extension.

Note:
* The documentation only tracks .py files in packages. To quickly convert a standard
directory into a package, an empty ``__init__.py`` file can be added to the directory.
* Class and method docstrings should be written in Google format both to be in accordance
with this codebase and to be supported by Sphinx.

The following points describe cases when the codebase has changed and the docs need
updating.

### A. The code has changed within existing .py files

https://user-images.githubusercontent.com/72270231/234063465-edac7d65-1c44-4175-9c6a-943421813f1a.mov

If the project structure has not changed (no new files, and files have not moved), then
the following procedure should be followed to update and access the docs:
1. In Terminal: ``cd path/to/documentation``. If left unchanged this should amount to 
running the following command from the root: ``cd documentation``.
2. Once in the documentation directory, run the following in Terminal: ``make html``.
3. Then, in or out of the terminal, go to ``documentation/_build/html/<page_name>.html``,
to find the html pages which make up the documentation. Alternatively, simply run the
following in Terminal: ``open documentation/_build/html/index.html`` to land on the
docs index page.

### B. New files have been written or the project structure has changed

https://user-images.githubusercontent.com/72270231/234068283-9435d4a3-64a4-471f-90e6-6589cbc46098.mov

If the project structure has changed (new files have been created and/or files have been)
moved, then the following procedure should be followed to update and access the docs:
1. In the documentation folder, delete all the `.rst` files (except the ``index.rst`` file). No other files or directories
   (e.g. conf.py, make.bat, Makefile, _build) should be deleted. This would require making
   the documentation from scratch.
2. In Terminal, go to the root of the repository (e.g. ``cd ../..`` if you are two-levels
deep in the project structure).
3. From the project root, run the following command: ``sphinx-apidoc -o path/to/documentation .``.
With this command, sphinx builds the api docs in the documentation folder and documents everything in
the repository root (hence the final ``.`` in the command).
4. Then, follow the steps in part A. above to make and access the docs.

### FAQs
* To ignore particular files/packages, go to the documentation folder (where all
the ``.rst`` files are located) and delete the ``.rst`` files corresponding to those
'unwanted' files/folders. Moreover, navigate to any of the ``.rst`` files that are
parents of the undesired files and remove them as needed from the 'toctree' section.
* The main points of configuration (e.g. the doc style used or the name of the library) for the docs are located
in the ``conf.py`` file.
* For further information, see the official Sphinx documentation: https://www.sphinx-doc.org/en/master/.
