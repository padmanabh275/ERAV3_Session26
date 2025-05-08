from PIL import Image, ImageDraw
import os

def create_icon(size):
    # Create a new image with a white background
    image = Image.new('RGBA', (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    
    # Calculate dimensions
    padding = size // 8
    circle_radius = (size - 2 * padding) // 2
    handle_length = size // 3
    
    # Draw the magnifying glass circle
    circle_center = (size - padding - circle_radius, size - padding - circle_radius)
    draw.ellipse(
        [
            circle_center[0] - circle_radius,
            circle_center[1] - circle_radius,
            circle_center[0] + circle_radius,
            circle_center[1] + circle_radius
        ],
        outline=(66, 133, 244, 255),  # Google Blue
        width=max(2, size // 16)
    )
    
    # Draw the handle
    handle_start = (
        circle_center[0] + circle_radius * 0.7,
        circle_center[1] + circle_radius * 0.7
    )
    handle_end = (
        handle_start[0] + handle_length,
        handle_start[1] + handle_length
    )
    draw.line(
        [handle_start, handle_end],
        fill=(66, 133, 244, 255),  # Google Blue
        width=max(2, size // 16)
    )
    
    return image

def main():
    # Create icons directory if it doesn't exist
    icons_dir = os.path.join(os.path.dirname(__file__), 'icons')
    os.makedirs(icons_dir, exist_ok=True)
    
    # Generate icons of different sizes
    sizes = [16, 48, 128]
    for size in sizes:
        icon = create_icon(size)
        icon.save(os.path.join(icons_dir, f'icon{size}.png'))
        print(f'Generated icon{size}.png')

if __name__ == '__main__':
    main() 