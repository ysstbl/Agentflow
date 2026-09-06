"""Compatibility entrypoint for the Agentflow application.

The canonical workflow is defined in graph.py. App entrypoints should not
recreate the graph or duplicate the state and agent logic here.
"""

from main import main


if __name__ == "__main__":
    main()
