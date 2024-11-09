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
    sin_a = math.sin(angle)
    cos_a = math.cos(angle)
    
    # Vertical intersections
    depth_v = float('inf')
    wall_orientation_v = None
    
    if cos_a > 0:  # Looking right
        x = int(player_x) + 1  # Start from the next column
        dx = 1
        wall_orientation_v = 'E'  # East-facing wall
    elif cos_a < 0:  # Looking left
        x = int(player_x)  # Start from the current column
        dx = -1
        wall_orientation_v = 'W'  # West-facing wall
    else:
        # If cos_a is zero, we are looking straight up or down
        return (max_depth, None)  # No vertical wall hit

    # Vertical raycasting loop
    for _ in range(int(max_depth)):
        depth_v = (x - player_x) / cos_a if cos_a != 0 else max_depth
        y = player_y + depth_v * sin_a
        
        # Check the cell on the vertical line
        map_x = x  # No adjustment needed for vertical walls
        map_y = int(y)
        
        if not (0 <= map_x < world_map.shape[1] and 0 <= map_y < world_map.shape[0]):
            depth_v = max_depth
            break
            
        if world_map[map_y, map_x]:  # Wall hit
            break
            
        x += dx
    
    # Horizontal intersections
    depth_h = float('inf')
    wall_orientation_h = None
    
    print(angle)
    if sin_a > 0:  # Looking down
        y = int(player_y) + 1  # Start from the next row
        dy = 1
        wall_orientation_h = 'S'  # South-facing wall
    elif sin_a < 0:  # Looking up
        y = int(player_y)  # Start from the current row
        dy = -1
        wall_orientation_h = 'N'  # North-facing wall
    else:
        # If sin_a is zero, we are looking straight left or right
        return (max_depth, None)  # No horizontal wall hit

    # Horizontal raycasting loop
    for _ in range(int(max_depth)):
        depth_h = (y - player_y) / sin_a if sin_a != 0 else max_depth
        x = player_x + depth_h * cos_a
        
        # Check the cell on the horizontal line
        map_x = int(x)
        map_y = y  # No adjustment needed for horizontal walls
        
        if not (0 <= map_x < world_map.shape[1] and 0 <= map_y < world_map.shape[0]):
            depth_h = max_depth
            break
            
        if world_map[map_y, map_x]:  # Wall hit
            break
            
        y += dy
    
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
    angle_step = fov / num_rays
    
    for i in range(num_rays):
        ray_angle = start_angle + i * angle_step
        depth, orientation = ray_cast_single(player_x, player_y, ray_angle, world_map, max_depth)
        rays[i] = depth * math.cos(angle - ray_angle)  # Fix fisheye effect
        
        # Store orientation if needed for rendering (optional)
        # You might want to create a separate array to store orientations
        # For example: orientations[i] = orientation
    
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
        self.player_angle = 0
        
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
            sin_a = math.sin(self.player_angle)
            cos_a = math.cos(self.player_angle)
            
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