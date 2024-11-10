# PiRenderer

## Overview
PiRenderer is a Doom-like 3D engine written in Python, designed to recreate the look and feel of retro 3D games like *Doom* using raycasting for fast, immersive graphics. It utilizes Pygame for rendering and numpy for efficient calculations, producing smooth, responsive movement and collision detection, basic shading, and an interactive minimap with support for interactive text boxes.

## Features
- **Raycasting Engine**: 3D environment rendered via fast raycasting techniques.
- **Dynamic Lighting and Shading**: Distance-based shading effects for realistic depth.
- **Minimap Overlay**: Displays the player's position, environment, and active rays.
- **Interactive Text Boxes**: Support for triggerable text messages when approaching marked locations.
- **Configurable Performance Options**: Adjustable rendering resolution for performance scaling.
- **Efficient Collision Detection**: Smooth player movement with accurate collision handling.

## Prerequisites
- Python 3.x
- Pygame
- numpy
- Pillow (for map loading)
- csv (included in Python standard library)

Install dependencies with:
```bash
pip install pygame numpy pillow
```

## Usage

### Launching PiRenderer
Run PiRenderer from the command line with:
```bash
python main.py <map_image_path>
```
where `<map_image_path>` is the path to a map image file that defines the game environment.

### Map Requirements
The map is defined using an image where specific colors indicate different elements:
- **Walls**: Red pixels (R=255, G=0) represent walls. Wall color intensity (blue channel) affects shading.
- **Player Start Position**: Green pixel (R=0, G=255, B=0) marks the player's initial location.
- **Empty Spaces**: Black pixels (R=0, G=0, B=0) represent walkable areas.
- **Text Box Triggers**: Purple pixels (R=240, G=0) mark interactive text box locations. The blue channel (B=xx) defines the text box ID.

### Text Box Setup
1. Create a CSV file with the same name as your map file (e.g., `map.png` → `map.csv`)
2. Format the CSV as follows:
```csv
id,message
1,Welcome to the first room! This is an example message.
2,You found a secret area! Good job exploring.
3,Watch out for the enemies ahead!
```
3. In your map image, use colors with format #F000xx where xx is the ID from your CSV:
   - #F00001 (RGB: 240, 0, 1) for text box ID 1
   - #F00002 (RGB: 240, 0, 2) for text box ID 2
   - etc.

### Controls
- **WASD or Arrow Keys**: Move forward/backward and strafe
- **Left/Right Arrows or A/D**: Rotate view
- **Shift**: Sprint
- **F**: Interact with nearby text boxes
- **F11**: Toggle rendering resolution for improved performance
- **Escape**: Close active text box or exit game
- **ESC**: Close text box (when active) or exit game

### Customization Options
You can adjust key parameters directly in the code to fine-tune performance and visuals:
- `WINDOW_WIDTH` and `WINDOW_HEIGHT`: Sets the display resolution
- `RENDER_SCALE`: Adjusts internal render scale for performance scaling
- `FOV` and `NUM_RAYS`: Control the field of view and ray density
- `PLAYER_SPEED` and `PLAYER_SPRINT_SPEED`: Adjusts player movement speeds
- `text_box_range`: Distance within which text boxes can be triggered

### Example Command
```bash
python3 main.py map.png
```

### Example Map Structure
To create an interactive map with text boxes:
1. Create your map image (e.g., `level1.png`) using the color codes above
2. Create a corresponding CSV file (`level1.csv`) with your messages
3. Place both files in the same directory
4. Launch the game with the map file path

## Notes
- PiRenderer supports variable frame rates and displays FPS in the window title
- The minimap shows map details, raycasts, and text box locations
- Text boxes automatically show an interaction prompt when in range
- Messages support word wrapping and multi-line display

## License
PiRenderer is open-source software.