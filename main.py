import pygame
import numpy as np
import sys
from PIL import Image
import math
import csv

debug = False

class DoomEngine:
    def __init__(self, map_path):
        pygame.init()
        
        # Window constants
        self.WINDOW_WIDTH = 1800
        self.WINDOW_HEIGHT = 900
        
        # Render resolution (can be adjusted for performance)
        self.RENDER_SCALE = 0.4  # Render at x resolution
        self.RENDER_WIDTH = int(self.WINDOW_WIDTH * self.RENDER_SCALE)
        self.RENDER_HEIGHT = int(self.WINDOW_HEIGHT * self.RENDER_SCALE)
        
        # Create render surface
        self.render_surface = pygame.Surface((self.RENDER_WIDTH, self.RENDER_HEIGHT))
        
        # Game constants
        self.FOV = math.pi/3  # 60 degrees in radians
        self.HALF_FOV = self.FOV / 2
        self.NUM_RAYS = self.RENDER_WIDTH  # Number of rays based on render width
        self.MAX_DEPTH = 800
        self.WALL_HEIGHT = int(2000 * self.RENDER_SCALE)  # Scale wall height with render resolution
        self.PLAYER_SPEED = 0.07
        self.PLAYER_SPRINT_SPEED = 0.2
        self.ROTATION_SPEED = 0.03
        
        # Minimap settings
        self.MINIMAP_SIZE = 200
        self.minimap_surface = pygame.Surface((self.MINIMAP_SIZE, self.MINIMAP_SIZE))
        
        # Precalculate colors and shading
        self.DARKNESS = 3
        self.COLORS = [(i, 50, 40) for i in range(256)]  # Using tuples instead of pygame.Color
        self.FLOOR_COLOR = (50, 50, 50)
        self.CEILING_COLOR = (100, 100, 100)
        self.SHADING_TABLE = self.create_shading_table()

        # Add new text box related properties
        self.text_boxes = {}  # Dictionary to store text box locations and messages
        self.active_text_box = None  # Currently displayed text box
        self.text_box_range = 2.0  # Distance within which player can interact
        self.font = pygame.font.Font(None, 36)  # Font for rendering text
        
        # Load text box data from CSV
        self.load_text_boxes(map_path)
        
        # Initialize screen
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("Doom-like Engine")
        
        # Load and process map
        self.load_map(map_path)
        
        # Player properties
        self.player_pos = np.array(self.find_player_start(), dtype=float)
        self.player_angle = 0
        self.current_rays = None
        
        # Precalculate ray angles
        self.setup_rays()

    def load_text_boxes(self, map_path):
        """Load text box data from a CSV file with the same name as the map."""
        csv_path = map_path.rsplit('.', 1)[0] + '.csv'
        try:
            with open(csv_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header row
                for row in reader:
                    if len(row) >= 2:
                        id_value = int(row[0])  # xx value from #F000xx
                        message = row[1]
                        self.text_boxes[id_value] = message
        except FileNotFoundError:
            print(f"Warning: Text box data file {csv_path} not found.")
        
    def setup_rays(self):
        """Precalculate ray angles and their sine/cosine values."""
        self.ray_angles = np.linspace(-self.HALF_FOV, self.HALF_FOV, self.NUM_RAYS)
        self.ray_cos = np.cos(self.ray_angles)
        self.ray_sin = np.sin(self.ray_angles)

    def create_shading_table(self):
        """Precalculate shading values for different distances and colors."""
        shading_table = {}
        for color in self.COLORS:  # Now color is already a tuple
            distance_shades = []
            for distance in range(self.MAX_DEPTH):
                shade = max(0, min(255, 255 - int(distance * 2)))
                shaded_color = (
                    color[0] * shade // 255,
                    color[1] * shade // 255,
                    color[2] * shade // 255
                )
                distance_shades.append(shaded_color)
            shading_table[color] = distance_shades
        return shading_table

    def load_map(self, map_path):
        """Modified load_map to handle text box markers."""
        img = Image.open(map_path)
        self.map_width, self.map_height = img.size
        self.map_data = []
        self.text_box_locations = {}  # Store text box locations
        
        for y in range(self.map_height):
            row = []
            for x in range(self.map_width):
                pixel = img.getpixel((x, y))
                if isinstance(pixel, int):
                    pixel = (pixel, pixel, pixel)
                
                # Check for text box marker (#F000xx)
                if pixel[0] == 240 and pixel[1] == 0:  # #F000xx
                    text_box_id = pixel[2]
                    self.text_box_locations[(x, y)] = text_box_id
                    row.append(0)  # Treat as empty space for collision
                elif pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:  # Empty space
                    row.append(0)
                elif pixel[0] == 255 and pixel[1] == 0:  # Wall
                    row.append(pixel[2] + 1)
                else:
                    row.append(0)
            self.map_data.append(row)
        
        self.map_array = np.array(self.map_data)

    def check_text_box_interaction(self):
        """Check if player is near any text box and handle interaction."""
        closest_distance = float('inf')
        closest_text_box = None

        for (x, y), text_id in self.text_box_locations.items():
            distance = math.sqrt(
                (self.player_pos[0] - x) ** 2 +
                (self.player_pos[1] - y) ** 2
            )
            
            if distance < self.text_box_range and distance < closest_distance:
                closest_distance = distance
                if text_id in self.text_boxes:
                    closest_text_box = text_id

        return closest_text_box

    def render_text_box(self):
        """Render the active text box on screen."""
        if self.active_text_box is not None and self.active_text_box in self.text_boxes:
            message = self.text_boxes[self.active_text_box]
            
            # Create text box background
            padding = 20
            margin = 50
            max_width = self.WINDOW_WIDTH - (margin * 2)
            
            # Render text with word wrap
            words = message.split()
            lines = []
            current_line = []
            
            for word in words:
                test_line = ' '.join(current_line + [word])
                test_surface = self.font.render(test_line, True, (255, 255, 255))
                
                if test_surface.get_width() <= max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        lines.append(word)
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Calculate text box dimensions
            line_height = self.font.get_linesize()
            box_height = (line_height * len(lines)) + (padding * 2)
            box_width = min(max_width + padding * 2, self.WINDOW_WIDTH - margin * 2)
            
            # Draw text box
            box_surface = pygame.Surface((box_width, box_height))
            box_surface.fill((0, 0, 0))
            pygame.draw.rect(box_surface, (255, 255, 255), box_surface.get_rect(), 2)
            
            # Render text lines
            for i, line in enumerate(lines):
                text_surface = self.font.render(line, True, (255, 255, 255))
                box_surface.blit(
                    text_surface,
                    (padding, padding + i * line_height)
                )
            
            # Position text box at bottom of screen
            self.screen.blit(
                box_surface,
                (margin, self.WINDOW_HEIGHT - box_height - margin)
            )
        
    def find_player_start(self):
        """Find the player's starting position (green pixel)."""
        img = Image.open(sys.argv[1])
        for y in range(self.map_height):
            for x in range(self.map_width):
                pixel = img.getpixel((x, y))
                if isinstance(pixel, int):
                    pixel = (pixel, pixel, pixel)
                if pixel[1] == 255 and pixel[0] == 0 and pixel[2] == 0:
                    return [x, y]
        return [1, 1]

    def check_collision(self, x, y, radius=0.2):
        """Check if position collides with walls."""
        check_points = [
            (x + dx, y + dy)
            for dx in [-radius, 0, radius]
            for dy in [-radius, 0, radius]
        ]
        
        for px, py in check_points:
            map_x, map_y = int(px), int(py)
            if not (0 <= map_x < self.map_width and 0 <= map_y < self.map_height):
                return True
            if self.map_array[map_y, map_x]:
                return True
        return False

    def cast_rays(self):
        """Cast all rays at once using vectorized operations."""
        angles = self.player_angle + self.ray_angles
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        
        distances = np.full(self.NUM_RAYS, self.MAX_DEPTH, dtype=float)
        wall_colors = np.zeros(self.NUM_RAYS, dtype=int)
        
        # Ray starting position
        ray_pos = np.zeros((self.NUM_RAYS, 2))
        ray_pos[:, 0] = self.player_pos[0]
        ray_pos[:, 1] = self.player_pos[1]
        
        # Ray direction
        ray_dir = np.stack((cos_a, sin_a), axis=-1)
        
        # DDA Algorithm vectorized
        step = 0.1
        for depth in range(self.MAX_DEPTH):
            # Update ray positions
            ray_pos = self.player_pos + ray_dir * (depth * step)
            map_pos = ray_pos.astype(int)
            
            # Check valid positions
            valid_pos = (map_pos[:, 0] >= 0) & (map_pos[:, 0] < self.map_width) & \
                       (map_pos[:, 1] >= 0) & (map_pos[:, 1] < self.map_height)
            
            if not np.any(valid_pos):
                break
            
            # Check for wall hits
            hit_mask = valid_pos & (distances == self.MAX_DEPTH)
            if not np.any(hit_mask):
                break
                
            # Get wall colors for valid positions
            current_colors = np.zeros_like(wall_colors)
            map_indices = map_pos[hit_mask]
            current_colors[hit_mask] = self.map_array[map_indices[:, 1], map_indices[:, 0]]
            
            # Update distances and colors for hits
            hit_indices = hit_mask & (current_colors > 0)
            if np.any(hit_indices):
                hit_distances = np.sqrt(
                    (ray_pos[hit_indices, 0] - self.player_pos[0]) ** 2 +
                    (ray_pos[hit_indices, 1] - self.player_pos[1]) ** 2
                )
                distances[hit_indices] = hit_distances
                wall_colors[hit_indices] = current_colors[hit_indices] - 1
                
        return distances, wall_colors

    def render(self):
        """Optimized render function."""
        # Clear render surface
        self.render_surface.fill(self.CEILING_COLOR)
        pygame.draw.rect(self.render_surface, self.FLOOR_COLOR,
                        (0, self.RENDER_HEIGHT//2, self.RENDER_WIDTH, self.RENDER_HEIGHT//2))
        
        # Cast all rays at once
        distances, wall_colors = self.cast_rays()
        self.current_rays = distances
        
        # Calculate wall heights vectorized
        wall_heights = np.minimum(
            (self.WALL_HEIGHT / (distances + 0.0001)).astype(int),
            self.RENDER_HEIGHT
        )
        
        # Batch wall rendering
        wall_tops = np.maximum(0, (self.RENDER_HEIGHT - wall_heights) // 2)
        wall_bottoms = np.minimum(self.RENDER_HEIGHT, (self.RENDER_HEIGHT + wall_heights) // 2)
        
        # Draw walls efficiently
        for x in range(self.NUM_RAYS):
            if wall_heights[x] > 0:
                color = self.COLORS[wall_colors[x]]
                shade_idx = min(int(distances[x] * self.DARKNESS), self.MAX_DEPTH - 1)
                shaded_color = self.SHADING_TABLE[color][shade_idx]
                pygame.draw.line(self.render_surface, shaded_color,
                               (x, wall_tops[x]), (x, wall_bottoms[x]))
        
        # Scale render surface to window size
        scaled_surface = pygame.transform.scale(
            self.render_surface,
            (self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        )
        self.screen.blit(scaled_surface, (0, 0))

        # Render text box if active
        self.render_text_box()
        
        # If near a text box, show interaction prompt
        nearby_text_box = self.check_text_box_interaction()
        if nearby_text_box is not None and self.active_text_box is None:
            prompt = self.font.render("Press F to interact", True, (255, 255, 255))
            prompt_rect = prompt.get_rect(center=(self.WINDOW_WIDTH // 2, self.WINDOW_HEIGHT - 50))
            self.screen.blit(prompt, prompt_rect)
        
        # Render minimap
        self.render_minimap()
        
        pygame.display.flip()

    def render_minimap(self):
        """Render the minimap showing player position and rays."""
        if self.current_rays is None:
            return
            
        # Clear minimap surface
        self.minimap_surface.fill((0, 0, 0))
        
        # Calculate scale factor
        scale = self.MINIMAP_SIZE / max(self.map_width, self.map_height)
        
        # Draw the game map on the minimap
        # Convert map array to the correct format (boolean array to uint8)
        map_array_bool = (self.map_array > 0).astype(np.uint8) * 255
        
        # Create a temporary surface for the map
        map_surface = pygame.Surface((self.map_width, self.map_height))
        pygame.surfarray.blit_array(map_surface, 
                                np.stack((map_array_bool,)*3, axis=-1))
        
        # Scale the surface
        map_surface = pygame.transform.scale(
            map_surface,
            (int(self.map_width * scale), int(self.map_height * scale))
        )
        self.minimap_surface.blit(map_surface, (0, 0))

        map_surface = pygame.transform.rotate(map_surface, -90)
        self.minimap_surface.blit(map_surface, (0, 0))

        map_surface = pygame.transform.flip(map_surface, True, False)
        self.minimap_surface.blit(map_surface, (0, 0))

        # Draw player position
        player_x_minimap = int(self.player_pos[0] * scale)
        player_y_minimap = int(self.player_pos[1] * scale)
        pygame.draw.circle(
            self.minimap_surface,
            (255, 0, 0),
            (player_x_minimap, player_y_minimap),
            3
        )

        # Draw rays on the minimap
        angles = self.player_angle + self.ray_angles
        for i, ray_length in enumerate(self.current_rays):
            ray_end_x = player_x_minimap + ray_length * scale * math.cos(angles[i])
            ray_end_y = player_y_minimap + ray_length * scale * math.sin(angles[i])
            
            pygame.draw.line(
                self.minimap_surface,
                (0, 255, 0),
                (player_x_minimap, player_y_minimap),
                (int(ray_end_x), int(ray_end_y)),
                1
            )

        # Blit minimap to the main screen
        self.screen.blit(self.minimap_surface, (self.WINDOW_WIDTH - self.MINIMAP_SIZE - 10, 10))

    def handle_input(self):
        """Handle player input for movement and rotation."""
        keys = pygame.key.get_pressed()

        if self.active_text_box is None:
            speed = self.PLAYER_SPEED
            
            # Rotation
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                self.player_angle -= self.ROTATION_SPEED
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                self.player_angle += self.ROTATION_SPEED

            if debug:
                print(self.player_angle)
            
            # sprint when holding shift
            if keys[pygame.K_LSHIFT]:
                speed = self.PLAYER_SPRINT_SPEED

            # Movement with improved collision detection
            if keys[pygame.K_UP] or keys[pygame.K_w] or keys[pygame.K_DOWN] or keys[pygame.K_s]:
                sin_a = math.sin(self.player_angle)
                cos_a = math.cos(self.player_angle)
                
                forward = 1 if keys[pygame.K_UP] or keys[pygame.K_w] else -1
                
                next_x = self.player_pos[0] + forward * cos_a * speed
                next_y = self.player_pos[1] + forward * sin_a * speed
                
                # Try to move in at least one direction if we can't move in both
                if not self.check_collision(next_x, next_y):
                    self.player_pos[0] = next_x
                    self.player_pos[1] = next_y
                elif not self.check_collision(next_x, self.player_pos[1]):
                    self.player_pos[0] = next_x
                elif not self.check_collision(self.player_pos[0], next_y):
                    self.player_pos[1] = next_y

    def run(self):
        """Main game loop."""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.active_text_box is not None:
                            self.active_text_box = None
                        else:
                            running = False
                    elif event.key == pygame.K_f:  # Handle F key press here
                        nearby_text_box = self.check_text_box_interaction()
                        if self.active_text_box is None and nearby_text_box is not None:
                            self.active_text_box = nearby_text_box
                        else:
                            self.active_text_box = None
                    elif event.key == pygame.K_F11:  # Toggle render scale
                        self.RENDER_SCALE = 1.0 if self.RENDER_SCALE < 1.0 else 0.5
                        self.RENDER_WIDTH = int(self.WINDOW_WIDTH * self.RENDER_SCALE)
                        self.RENDER_HEIGHT = int(self.WINDOW_HEIGHT * self.RENDER_SCALE)
                        self.render_surface = pygame.Surface((self.RENDER_WIDTH, self.RENDER_HEIGHT))
                        self.NUM_RAYS = self.RENDER_WIDTH
                        self.setup_rays()
            
            self.handle_input()
            self.render()
            
            # Display FPS
            fps = clock.get_fps()
            pygame.display.set_caption(f'Doom-like Engine - FPS: {fps:.1f} - Scale: {self.RENDER_SCALE:.2f}')
            clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python doom_engine.py <map_image_path>")
        sys.exit(1)
        
    engine = DoomEngine(sys.argv[1])
    engine.run()