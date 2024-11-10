# PiDoom

Doom-style raycasting engine using pygame and numpy

This script renders a 3D-like environment from a 2D image. It uses the
raycasting technique to create the 3D illusion.

The game map is a 2D image where each pixel represents a wall (white pixel)
or empty space (black pixel). The game map is stored in the `GameMap` class.

The `Renderer` is responsible for rendering the 3D environment. It uses the
raycasting technique to render the walls and the floor.

The game loop is handled by the `Game` class. It handles user input and
updates the player position accordingly.

The game map image path is passed as an argument to the script. The image
should be in the same directory as the script.

Example usage:
    python script.py map.png

