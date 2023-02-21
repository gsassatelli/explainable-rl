# explainable-RL

- main: merge into this branch after checkpoints with DataSparq
- dev: merge into this branch after task branch has been approved by peers

Create one branch per Jira task.    

## Docs
Used sphinx package. See https://www.youtube.com/watch?v=BWIrhgCAae0&t=18s for a quick video guide on getting started with sphinx.

Note that when a new module (e.g. foundation) is created, an init file (even empty) must be added to the module directory to let sphinx know to document the module.

## Performance analysis
For memory usage: 
- Note that you need to decorate all the functions being called with the "@profile" decorator for full memory analysis
- To view the memory usage without saving to the results, run: python3 -m memory_profiler performance/full_flow.py
- To save the memory data to a data file, run: mprof run performance/full_flow.py
- To plot the memory usage file just created with the standard title, run: mprof plot
- To plot the memory usage file just created with a custom title, run: mprof plot -t "DESIRED_TITLE"

For time complexity:
- Used the in-built Pycharm profiler (and line-by-line profiler) with line_profiler_pycharm package and @profile decorator
