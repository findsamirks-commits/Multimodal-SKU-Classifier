"""Synthetic product catalog and image generator for immediate testing and benchmarking.
Creates realistic product titles, generates category-styled RGB images,
and outputs data/products.csv.
"""

from pathlib import Path
import random
from PIL import Image, ImageDraw

from config import default_config

CATEGORIES = {
    "Electronics": {
        "titles": [
            "Wireless Bluetooth 5.3 Over-Ear Headphones with Active Noise Canceling",
            "Ultra-Slim 15.6 Inch Portable USB-C IPS Monitor 1080P",
            "RGB Mechanical Gaming Keyboard with Hot-Swappable Blue Switches",
            "15W Fast Wireless Charging Stand for Qi-Enabled Smartphones",
            "Smart Fitness Tracker Watch with Heart Rate and SpO2 Monitor",
            "Ergonomic Optical Gaming Mouse with Programmable Macro Buttons",
            "High-Speed Thunderbolt 4 Docking Station Dual 4K Display",
            "Compact True Wireless Earbuds with Deep Bass and Mic",
            "Portable 20,000mAh Power Bank Fast Charge USB-PD 65W",
            "Smart Home Security Camera 2K WiFi Night Vision Pan-Tilt",
            "Noise-Canceling USB Conference Speakerphone with 360 Mic",
            "Universal Dual-Bay External Hard Drive Dock USB 3.0",
        ],
        "colors": [(30, 45, 90), (15, 82, 186), (20, 20, 30), (0, 150, 214)],
        "shape": "rectangle",
    },
    "Footwear": {
        "titles": [
            "Men's Breathable Mesh Running Shoes Lightweight Athletic Sneakers",
            "Women's Classic Leather Chelsea Ankle Boots with Elastic Gore",
            "Trail Running Shoes All-Terrain Grip Cushioning Sole",
            "Casual Slip-on Canvas Loafers Comfortable Walking Flats",
            "Waterproof Outdoor Hiking Boots with High Ankle Support",
            "Orthopedic Supportive Arch Sneakers for Long Standing",
            "Retro Low-Top Skate Shoes Classic Suede and Canvas",
            "Cushioned Running Shoes Shock Absorbing Foam Sole",
            "Leather Dress Oxford Shoes Handcrafted Cap-Toe Formal",
            "Breathable Road Cycling Shoes Carbon Fiber Reinforced",
            "Lightweight Slip-Resistant Work Safety Shoes Steel Toe",
            "Water-Resistant Winter Snow Boots Faux Fur Lined",
        ],
        "colors": [(139, 69, 19), (160, 82, 45), (105, 105, 105), (205, 133, 63)],
        "shape": "ellipse",
    },
    "Apparel": {
        "titles": [
            "Premium 100% Organic Cotton Classic Crewneck Short Sleeve T-Shirt",
            "Slim-Fit Stretch Denim Jeans Vintage Dark Wash 5-Pocket",
            "Lightweight Packable Windbreaker Jacket Water-Repellent",
            "Merino Wool Crew Knit Sweater Soft Thermal Pullover",
            "High-Waisted Athletic Workout Leggings with Side Pockets",
            "Casual Oxford Cotton Button-Down Long Sleeve Dress Shirt",
            "Unisex Heavyweight Fleece Pullover Hoodie Kangaroo Pocket",
            "Breathable Quick-Dry Gym Shorts with Compression Liner",
            "Waterproof Hooded Raincoat Trench with Adjustable Drawstring",
            "Soft Flannel Plaid Button-Up Casual Shirt Regular Fit",
            "Seamless Sports Bra Medium Impact Racerback Padded",
            "Warm Down Puffer Vest Ultralight Sleeveless Winter Jacket",
        ],
        "colors": [(220, 20, 60), (75, 0, 130), (46, 139, 87), (218, 165, 32)],
        "shape": "triangle",
    },
    "Home & Kitchen": {
        "titles": [
            "Double-Wall Stainless Steel French Press Coffee Maker 34 oz",
            "Non-Stick Ceramic 10-Inch Frying Pan Skillet Induction Safe",
            "Ultrasonic Cool Mist Aroma Essential Oil Diffuser with LED Light",
            "Heavy-Duty Enameled Cast Iron Dutch Oven with Basting Lid 6 Qt",
            "Organic Bamboo Kitchen Cutting Board with Deep Juice Groove",
            "Precision Digital Food Kitchen Scale 0.1g High Accuracy",
            "Set of 6 Airtight Glass Food Storage Containers with Bamboo Lids",
            "Stainless Steel Manual Hand Crank Coffee Bean Burr Grinder",
            "Silicone Heat-Resistant Cooking Utensil Spatula Set 12 Pieces",
            "Electric Gooseneck Pour-Over Tea Kettle Temperature Control",
            "Heavyweight Microfiber Kitchen Dish Towels Super Absorbent",
            "Professional Damascus Steel Chef Knife 8-Inch Razor Sharp",
        ],
        "colors": [(205, 92, 92), (47, 79, 79), (188, 143, 143), (112, 128, 144)],
        "shape": "circle",
    },
}


def draw_sample_image(
    output_path: Path,
    category: str,
    bg_color: tuple,
    shape: str,
    size: int = 256,
):
    """Draw a category-themed synthetic RGB image."""
    img = Image.new("RGB", (size, size), color=bg_color)
    draw = ImageDraw.Draw(img)

    margin = 40
    accent = tuple(min(255, c + 70) for c in bg_color)
    dark_accent = tuple(max(0, c - 40) for c in bg_color)

    if shape == "rectangle":
        draw.rectangle(
            [margin, margin, size - margin, size - margin],
            fill=accent,
            outline=dark_accent,
            width=4,
        )
        draw.rectangle(
            [margin + 30, margin + 30, size - margin - 30, size - margin - 30],
            fill=dark_accent,
        )
    elif shape == "circle":
        draw.ellipse(
            [margin, margin, size - margin, size - margin],
            fill=accent,
            outline=dark_accent,
            width=4,
        )
        draw.ellipse(
            [margin + 40, margin + 40, size - margin - 40, size - margin - 40],
            fill=dark_accent,
        )
    elif shape == "ellipse":
        draw.ellipse(
            [margin, margin + 25, size - margin, size - margin - 25],
            fill=accent,
            outline=dark_accent,
            width=4,
        )
    elif shape == "triangle":
        points = [
            (size // 2, margin),
            (size - margin, size - margin),
            (margin, size - margin),
        ]
        draw.polygon(points, fill=accent, outline=dark_accent)

    # Add subtle grid lines
    for i in range(0, size, 32):
        draw.line([(0, i), (size, i)], fill=tuple(max(0, c - 15) for c in bg_color), width=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, quality=90)


def generate_dataset(csv_path: Path, images_dir: Path, samples_per_category: int = 30):
    """Generate synthetic CSV catalog and companion RGB images."""
    records = []
    images_dir.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    item_id = 1
    for cat_name, cat_data in CATEGORIES.items():
        base_titles = cat_data["titles"]
        colors = cat_data["colors"]
        shape = cat_data["shape"]

        for i in range(samples_per_category):
            title = base_titles[i % len(base_titles)]
            # Add slight variation in adjectives
            if i >= len(base_titles):
                variation = random.choice(["Premium", "Pro Series", "Eco Edition", "Deluxe", "2026 Model"])
                title = f"{variation} {title}"

            bg_color = random.choice(colors)
            img_filename = f"prod_{cat_name.lower().replace(' ', '_')}_{i:03d}.jpg"
            img_path = images_dir / img_filename

            draw_sample_image(img_path, cat_name, bg_color, shape)

            records.append({
                "product_id": f"SKU-{item_id:05d}",
                "title": title,
                "image_path": str(img_path.resolve()),
                "label": cat_name,
            })
            item_id += 1

    import pandas as pd
    df = pd.DataFrame(records)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} synthetic multimodal products across {len(CATEGORIES)} categories.")
    print(f"Dataset CSV saved to: {csv_path.resolve()}")
    print(f"Images saved to: {images_dir.resolve()}")


if __name__ == "__main__":
    data_dir = default_config.DATA_DIR
    generate_dataset(
        csv_path=default_config.DATA_CSV,
        images_dir=data_dir / "images",
        samples_per_category=35,  # 140 total sample items
    )
