# explainable-RL

- main: merge into this branch after checkpoints with DataSparq
- dev: merge into this branch after task branch has been approved by peers

Create one branch per Jira task.

## Quick Guide on using the Sphinx Documentation
The documentation of the explainable-RL package first release has already been 
created and resided in ``explainable-RL/documentation/_build/html/<page_name>.html``.
It was built using the Sphinx documentation generator.

Note:
* The documentation only tracks .py files in packages. To quickly convert a standard
directory into a package, an empty ``__init__.py`` file can be added to the directory.
* Class and method docstrings should be written in Google format both to be in accordance
with this codebase and to be supported by Sphinx.

The following points describe cases when the codebase has changed and the docs need
updating.

### A. The code has changed within existing .py files
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
If the project structure has changed (new files have been created and/or files have been)
moved, then the following procedure should be followed to update and access the docs:
1. In the documentation folder, delete all the `.rst` files. No other files or directories
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