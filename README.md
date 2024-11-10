Certainly, here’s an expanded version that includes more detail on map requirements, setup, and configuration options:

---

# PiRenderer

## Overview
PiRenderer is a Doom-like 3D engine written in Python, designed to recreate the look and feel of retro 3D games like *Doom* using raycasting for fast, immersive graphics. It utilizes Pygame for rendering and numpy for efficient calculations, producing smooth, responsive movement and collision detection, basic shading, and an interactive minimap.

## Features
- **Raycasting Engine**: 3D environment rendered via fast raycasting techniques.
- **Dynamic Lighting and Shading**: Distance-based shading effects for realistic depth.
- **Minimap Overlay**: Displays the player’s position, environment, and active rays.
- **Configurable Performance Options**: Adjustable rendering resolution for performance scaling.
- **Efficient Collision Detection**: Smooth player movement with accurate collision handling.

## Prerequisites
- Python 3.x
- Pygame
- numpy
- Pillow (for map loading)

Install dependencies with:
```bash
pip install pygame numpy pillow
```

## Usage

### Launching PiRenderer
Run PiRenderer from the command line with:
```bash
python doom_engine.py <map_image_path>
```
where `<map_image_path>` is the path to a map image file that defines the game environment.

### Map Requirements
The map is defined using an image where specific colors indicate different elements:
- **Walls**: Red pixels (R=255, G=0) represent walls. Wall color intensity (blue channel) affects shading.
- **Player Start Position**: Green pixel (R=0, G=255, B=0) marks the player’s initial location.
- **Empty Spaces**: Black pixels (R=0, G=0, B=0) represent walkable areas.

### Controls
- **WASD or Arrow Keys**: Move forward/backward and strafe
- **Left/Right Arrows or A/D**: Rotate view
- **Shift**: Sprint
- **F11**: Toggle rendering resolution for improved performance
- **Escape**: Exit the game

### Customization Options
You can adjust key parameters directly in the code to fine-tune performance and visuals:
- `WINDOW_WIDTH` and `WINDOW_HEIGHT`: Sets the display resolution.
- `RENDER_SCALE`: Adjusts internal render scale for performance scaling.
- `FOV` and `NUM_RAYS`: Control the field of view and ray density for different levels of detail.
- `PLAYER_SPEED` and `PLAYER_SPRINT_SPEED`: Adjusts player movement speeds.
  
### Example Command
```bash
python3 main.py map.png
```

## Notes
- PiRenderer supports variable frame rates and displays FPS in the window title.
- The minimap can be configured to show map details and raycasts for enhanced spatial awareness.

## License
PiRenderer is open-source software.