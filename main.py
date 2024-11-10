"""
Doom-style raycasting engine using pygame and numpy

This script renders a 3D-like environment from a 2D image. It uses the
raycasting technique to create the 3D illusion.

The game map is a 2D image where each pixel represents a wall (white pixel)
or empty space (black pixel). The game map is stored in the GameMap class.

The renderer is responsible for rendering the 3D environment. It uses the
raycasting technique to render the walls and the floor.

The game loop is handled by the Game class. It handles user input and
updates the player position accordingly.

The game map image path is passed as an argument to the script. The image
should be in the same directory as the script.

Example usage:
    python script.py map.png

"""

import pygame
import sys
import colorsys
import numpy as np
from PIL import Image
import math
from numba import njit
from array import array
from dataclasses import dataclass
from typing import Tuple, List


"""By coderXYZ7 """

player_angle_global = 0
fov_global = 0

@dataclass
class RenderSettings:
    """Settings for the renderer configuration"""
    SCREEN_WIDTH: int = 1500
    SCREEN_HEIGHT: int = 900
    FOV: float = math.pi / 3  # 60 degrees FOV
    RAY_SCALE: int = 5  # Higher = fewer rays but faster
    MAX_DEPTH: float = 50.0
    PLAYER_SPEED: float = 0.3
    ROTATION_SPEED: float = 0.05
    
    def __post_init__(self):
        self.HALF_WIDTH = self.SCREEN_WIDTH // 2
        self.HALF_HEIGHT = self.SCREEN_HEIGHT // 2
        self.NUM_RAYS = self.SCREEN_WIDTH // self.RAY_SCALE
        self.HALF_FOV = self.FOV / 2
        self.DELTA_ANGLE = self.FOV / self.NUM_RAYS
        self.DISTANCE = self.HALF_WIDTH / math.tan(self.HALF_FOV)
    
    fov_global = FOV

@dataclass
class Colors:
    """Color definitions"""
    SKY: Tuple[int, int, int] = (50, 100, 200)
    FLOOR: Tuple[int, int, int] = (80, 80, 80)
    WALL: Tuple[int, int, int] = (200, 150, 100)

@njit(cache=True)
def ray_cast_single(player_x: float, player_y: float, angle: float, 
                    world_map: np.ndarray, max_depth: float) -> tuple:
    """Cast a single ray and return the distance to wall and wall orientation"""
    # Normalize angle to stay within [0, 2π]
    angle = angle % (2 * math.pi)
    
    sin_a = math.sin(angle)
    cos_a = math.cos(angle)
    
    # Vertical intersections
    x_vert, y_vert = player_x, player_y
    depth_v = 0
    dx = 1 if cos_a > 0 else -1
    
    # First vertical intersection point
    # Determine the first vertical intersection point and the step direction
    if cos_a > 0:
        # Ray is facing right, calculate the first intersection to the right
        x_vert = math.ceil(player_x)
        dx_vert = x_vert - player_x
    else:
        # Ray is facing left, calculate the first intersection to the left
        x_vert = math.floor(player_x)
        dx_vert = x_vert - player_x
    
    # Calculate y coordinate of first intersection
    y_vert = player_y + dx_vert * sin_a / cos_a if cos_a != 0 else player_y
    
    # Steps between vertical lines
    delta_depth = abs(1 / cos_a) if cos_a != 0 else float('inf')
    dy = delta_depth * sin_a
    
    # Vertical ray casting loop
    wall_orientation_v = 'E' if cos_a > 0 else 'W'
    for _ in range(int(max_depth)):
        map_x, map_y = int(x_vert), int(y_vert)
        
        # Check boundaries
        if not (0 <= map_x < world_map.shape[1] and 0 <= map_y < world_map.shape[0]):
            depth_v = float('inf')
            break
            
        # Check for wall hit
        if world_map[map_y, map_x]:
            depth_v = math.sqrt((x_vert - player_x)**2 + (y_vert - player_y)**2)
            break
            
        x_vert += dx
        y_vert += dy
        depth_v += delta_depth
    
    # Horizontal intersections
    x_horz, y_horz = player_x, player_y
    depth_h = 0
    dy = 1 if sin_a > 0 else -1
    
    # First horizontal intersection point
    if sin_a > 0:
        y_horz = math.ceil(player_y)
        dy_horz = y_horz - player_y
    else:
        y_horz = math.floor(player_y)
        dy_horz = y_horz - player_y
    
    # Calculate x coordinate of first intersection
    x_horz = player_x + dy_horz * cos_a / sin_a if sin_a != 0 else player_x
    
    # Steps between horizontal lines
    delta_depth = abs(1 / sin_a) if sin_a != 0 else float('inf')
    dx = delta_depth * cos_a
    
    # Horizontal ray casting loop
    wall_orientation_h = 'S' if sin_a > 0 else 'N'
    for _ in range(int(max_depth)):
        map_x, map_y = int(x_horz), int(y_horz)
        
        # Check boundaries
        if not (0 <= map_x < world_map.shape[1] and 0 <= map_y < world_map.shape[0]):
            depth_h = float('inf')
            break
            
        # Check for wall hit
        if world_map[map_y, map_x]:
            depth_h = math.sqrt((x_horz - player_x)**2 + (y_horz - player_y)**2)
            break
            
        x_horz += dx
        y_horz += dy
        depth_h += delta_depth
    
    # Return the shortest distance and corresponding orientation
    if depth_v < depth_h:
        return (depth_v, wall_orientation_v)
    else:
        return (depth_h, wall_orientation_h)

@njit(cache=True)
def ray_cast_all(player_x: float, player_y: float, angle: float, 
                 world_map: np.ndarray, settings: Tuple) -> np.ndarray:
    """Cast all rays for the view"""
    num_rays, max_depth, fov = settings
    rays = np.zeros(num_rays, dtype=np.float32)
    
    start_angle = angle - fov / 2
    angle_step = fov / (num_rays - 1)  # Distribute rays evenly across FOV
    
    for i in range(num_rays):
        ray_angle = start_angle + (i * angle_step)
        depth, orientation = ray_cast_single(player_x, player_y, ray_angle, world_map, max_depth)
        # Fix fisheye effect
        rays[i] = depth * math.cos(ray_angle - angle)
    
    return rays

class GameMap:
    """Handles map loading and collision detection"""
    def __init__(self, map_path: str):
        self.load_map(map_path)
    
    def load_map(self, map_path: str) -> Tuple[np.ndarray, List[float]]:
        """Load map from image file"""
        with Image.open(map_path) as img:
            img = img.convert('RGB')
            self.width, self.height = img.size
            
            # Convert image to numpy array
            img_data = np.array(img)
            
            # Blue channel for walls and red channel for hue
            self.wall_map = img_data[:, :, 2] == 255
            self.wall_hues = np.where(self.wall_map, img_data[:, :, 0], 0)
            
            # Find player start (green pixel)
            player_positions = np.where(img_data[:, :, 1] == 255)
            if len(player_positions[0]) == 0:
                raise ValueError("No player start position (green pixel) found!")
            
            self.player_start = [float(player_positions[1][0]), 
                               float(player_positions[0][0])]
    
    def check_collision(self, x: float, y: float, radius: float = 0.2) -> bool:
        """Check if position collides with walls"""
        check_points = [
            (x + dx, y + dy)
            for dx in [-radius, 0, radius]
            for dy in [-radius, 0, radius]
        ]
        
        for px, py in check_points:
            map_x, map_y = int(px), int(py)
            if not (0 <= map_x < self.width and 0 <= map_y < self.height):
                return True
            if self.wall_map[map_y, map_x]:
                return True
        return False

class Renderer:
    def __init__(self, settings: RenderSettings, colors: Colors, game_map: GameMap, player_pos: Tuple[float, float]):
        self.settings = settings
        self.colors = colors
        self.game_map = game_map
        self.player_pos = player_pos
        self.player_angle = player_angle_global
        self.current_rays = None  # Store current rays data

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode(
            (settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT)
        )
        self.buffer = pygame.Surface(
            (settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT)
        )
        pygame.display.set_caption('Doom-style Renderer')

        # Precalculate shaded colors
        self.shaded_colors = self._precalculate_colors()

        # Create minimap
        self.minimap_size = 200  # Size of the minimap
        self.minimap_surface = pygame.Surface((self.minimap_size, self.minimap_size))

    def get_shaded_color(self, depth: float, hue: int) -> Tuple[int, int, int]:
        """Get precalculated shaded color with specific hue"""
        depth = min(int(self.settings.MAX_DEPTH - 1), int(depth - (depth % 5)))
        base_color = self.shaded_colors[depth]['wall']
        
        # Normalize hue value to range of color wheel
        hue_normalized = (hue / 360) % 1
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue_normalized, 0.75, 0.75)
        
        # Apply shading to RGB
        shaded_rgb = (
            max(0, min(int(base_color[0] * rgb[0]), 255)),
            max(0, min(int(base_color[1] * rgb[1]), 255)),
            max(0, min(int(base_color[2] * rgb[2]), 255))
        )
        
        return shaded_rgb

    def _precalculate_colors(self) -> dict:
        """Precalculate color shadings for wall, floor, and sky based on depth"""
        shaded = {}
        for depth in range(0, int(self.settings.MAX_DEPTH), 1):
            factor = 1.0 / (1 + 0.002 * depth ** 2)
            shaded[depth] = {
                'wall': tuple(max(0, min(int(c * factor), 255)) for c in self.colors.WALL),
                'floor': tuple(max(0, min(int(c * factor), 255)) for c in self.colors.FLOOR),
                'sky': tuple(max(0, min(int(c * factor), 255)) for c in self.colors.SKY)
            }
        return shaded

    def render_static_background(self):
        """Render the static sky and floor"""
        self.buffer.fill(self.colors.SKY)
        # Draw the floor
        floor_height_start = self.settings.HALF_HEIGHT
        for y in range(floor_height_start, self.settings.SCREEN_HEIGHT):
            pygame.draw.line(
                self.buffer,
                self.colors.FLOOR,
                (0, y),
                (self.settings.SCREEN_WIDTH, y)
            )
    
    def render_minimap(self):
        """Render the minimap showing player position and rays"""
        if self.current_rays is None:
            return
            
        # Clear minimap surface
        self.minimap_surface.fill((0, 0, 0))  # Black background for minimap
        
        # Draw the game map on the minimap
        scale = self.minimap_size / max(self.game_map.width, self.game_map.height)
        for y in range(self.game_map.height):
            for x in range(self.game_map.width):
                if self.game_map.wall_map[y, x]:
                    pygame.draw.rect(
                        self.minimap_surface,
                        (255, 255, 255),  # White walls
                        (x * scale, y * scale, scale, scale)
                    )

        # Draw player position
        player_x_minimap = self.player_pos[0] * scale
        player_y_minimap = self.player_pos[1] * scale
        pygame.draw.circle(
            self.minimap_surface,
            (255, 0, 0),  # Red color for player
            (int(player_x_minimap), int(player_y_minimap)),
            5  # Radius of the player marker
        )

        # Draw rays on the minimap
        for ray in range(self.settings.NUM_RAYS):
            angle = self.player_angle - self.settings.HALF_FOV + (ray * self.settings.DELTA_ANGLE)
            ray_length = max(1, self.current_rays[ray])  # Get ray length from stored rays
            ray_end_x = player_x_minimap + ray_length * scale * math.cos(angle)
            ray_end_y = player_y_minimap + ray_length * scale * math.sin(angle)

            pygame.draw.line(
                self.minimap_surface,
                (0, 255, 0),  # Green color for rays
                (int(player_x_minimap), int(player_y_minimap)),
                (int(ray_end_x), int(ray_end_y)),
                1  # Line width
            )

        # Blit minimap to the main screen
        self.screen.blit(self.minimap_surface, (self.settings.SCREEN_WIDTH - self.minimap_size - 10, 10))

    def render_frame(self, rays, player_angle, player_pos):
        """Render a complete frame"""
        self.current_rays = rays  # Store the current rays data
        self.player_angle = player_angle  # Update current player angle
        self.player_pos = player_pos  # Update current player position
        
        # Clear the buffer for the new frame
        self.render_static_background()  # Draw the sky and floor first
        
        for ray in range(self.settings.NUM_RAYS):
            depth = max(1.0, rays[ray])
            wall_height = min(
                int(self.settings.DISTANCE / depth),
                self.settings.SCREEN_HEIGHT * 2
            )
            wall_top = max(0, self.settings.HALF_HEIGHT - wall_height // 2)
            wall_bottom = min(
                self.settings.SCREEN_HEIGHT,
                self.settings.HALF_HEIGHT + wall_height // 2
            )
            
            # Get hue for the current wall slice
            wall_x = int(player_pos[0] + depth * math.cos(player_angle))
            wall_y = int(player_pos[1] + depth * math.sin(player_angle))
            
            wall_x = max(0, min(wall_x, self.game_map.width - 1))
            wall_y = max(0, min(wall_y, self.game_map.height - 1))
            
            hue = self.game_map.wall_hues[wall_y, wall_x]
            
            # Here you might want to use the orientation to adjust the hue or color
            # For example, you could define colors based on orientation
            # orientation = orientations[ray]  # This assumes you stored orientations
            
            # Get a shaded color based on the depth and hue values
            wall_color = self.get_shaded_color(depth, hue)
            
            pygame.draw.rect(
                self.buffer,
                wall_color,
                (ray * self.settings.RAY_SCALE, wall_top,
                self.settings.RAY_SCALE, wall_bottom - wall_top)
            )
        
        self.screen.blit(self.buffer, (0, 0))
        self.render_minimap()  # Render the minimap
        pygame.display.flip()

class Game:
    """Main game class"""
    def __init__(self, map_path: str):
        self.settings = RenderSettings()
        self.colors = Colors()
        
        # Initialize components
        self.game_map = GameMap(map_path)
        self.player_pos = self.game_map.player_start
        
        # Player state
        self.player_angle = - fov_global/2
        
        # Initialize renderer after player_pos is initialized
        self.renderer = Renderer(self.settings, self.colors, self.game_map, self.player_pos)
        
        # Performance monitoring
        self.clock = pygame.time.Clock()
    
    def handle_input(self) -> bool:
        """Handle player input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        
        keys = pygame.key.get_pressed()
        
        # Rotation
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.player_angle -= self.settings.ROTATION_SPEED
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.player_angle += self.settings.ROTATION_SPEED
        
        # Movement
        if keys[pygame.K_UP] or keys[pygame.K_w] or keys[pygame.K_DOWN] or keys[pygame.K_s]:
            sin_a = math.sin((self.player_angle)) #+ self.settings.HALF_FOV/2)
            cos_a = math.cos((self.player_angle)) # + self.settings.HALF_FOV/2)
            
            forward = 1 if keys[pygame.K_UP] or keys[pygame.K_w] else -1
            
            next_x = self.player_pos[0] + forward * cos_a * self.settings.PLAYER_SPEED
            next_y = self.player_pos[1] + forward * sin_a * self.settings.PLAYER_SPEED
            
            if not self.game_map.check_collision(next_x, next_y):
                self.player_pos[0] = next_x
                self.player_pos[1] = next_y
        
        global player_angle_global
        player_angle_global = self.player_angle
        
        return True
    
    def run(self):
        """Main game loop"""
        running = True
        while running:
            # Handle input
            running = self.handle_input()
            
            # Ray casting
            ray_settings = (
                self.settings.NUM_RAYS,
                self.settings.MAX_DEPTH,
                self.settings.FOV
            )
            rays = ray_cast_all(
                self.player_pos[0],
                self.player_pos[1],
                self.player_angle,
                self.game_map.wall_map,
                ray_settings
            )
            
            # Render frame
            self.renderer.render_frame(rays, self.player_angle, self.player_pos)
            
            # Cap framerate and show FPS
            self.clock.tick(60)
            fps = self.clock.get_fps()
            pygame.display.set_caption(f'Doom-style Renderer - FPS: {fps:.1f}')
        
        pygame.quit()

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_map_image>")
        sys.exit(1)
    
    game = Game(sys.argv[1])
    game.run()

if __name__ == "__main__":
    main()