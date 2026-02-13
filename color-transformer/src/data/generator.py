"""
Synthetic data generator for color-attribute relationship learning.
Each sentence must start with a color and end with an attribute of that color.
"""

import random
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ColorAttributePair:
    """Data class for color-attribute relationships."""
    color: str
    attribute: str
    object_word: str
    template_id: int


class ColorDataGenerator:
    """
    Generates synthetic sentences following the rule:
    "Every sentence must start with a color and end with an attribute of that color."
    """
    
    # Color-attribute mapping dictionary
    COLOR_ATTRIBUTES: Dict[str, List[str]] = {
        "red": ["sweet", "sour", "hot", "sharp", "vibrant", "fresh", "ripe"],
        "blue": ["deep", "wide", "endless", "peaceful", "cold", "clear", "calm"],
        "green": ["fresh", "natural", "peaceful", "lively", "verdant", "lush", "forest-like"],
        "yellow": ["bright", "warm", "cheerful", "radiant", "energetic", "sun-like", "golden"],
        "black": ["dark", "mysterious", "elegant", "deep", "night-like", "endless", "noble"],
        "white": ["clean", "pure", "clear", "bright", "snow-like", "innocent", "spotless"],
        "purple": ["noble", "mysterious", "magical", "rich", "royal", "legendary", "deep"],
        "orange": ["vibrant", "energetic", "warm", "cheerful", "bright", "autumnal", "sweet"],
        "pink": ["sweet", "delicate", "cute", "romantic", "soft", "flower-like", "innocent"],
        "gray": ["simple", "neutral", "cold", "dull", "cloud-like", "metallic", "leaden"]
    }
    
    # Common objects to use in sentences
    OBJECTS: List[str] = [
        "sky", "sea", "forest", "flower", "apple", "car", "house", "book",
        "pencil", "table", "chair", "cat", "dog", "bird", "cloud", "sun",
        "moon", "star", "mountain", "plain", "river", "lake", "leaf", "tree"
    ]
    
    # Sentence templates for variety
    TEMPLATES: List[str] = [
        "{color} {object} {attribute}",
        "the {color} {object} is {attribute}",
        "{object} is {color} and {attribute}",
        "the {color} colored {object} is {attribute}",
        "{object} looks {attribute} because it's {color}"
    ]
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional random seed."""
        if seed is not None:
            random.seed(seed)
    
    def generate_pair(self) -> Tuple[str, str, str]:
        """
        Generate a random color-attribute pair with an object.
        
        Returns:
            Tuple of (color, object_word, attribute)
        """
        # Randomly select a color
        color = random.choice(list(self.COLOR_ATTRIBUTES.keys()))
        
        # Select an attribute for that color
        attribute = random.choice(self.COLOR_ATTRIBUTES[color])
        
        # Select a random object
        object_word = random.choice(self.OBJECTS)
        
        return color, object_word, attribute
    
    def generate_sentence(self, color: str, object_word: str, attribute: str) -> str:
        """
        Create a sentence from components using random template.
        
        Args:
            color: The color word
            object_word: The object word
            attribute: The attribute word
            
        Returns:
            Complete sentence string
        """
        template = random.choice(self.TEMPLATES)
        return template.format(
            color=color,
            object=object_word,
            attribute=attribute
        )
    
    def generate_dataset(
        self, 
        num_samples: int = 5000,
        output_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Generate a complete dataset of synthetic sentences.
        
        Args:
            num_samples: Number of sentences to generate
            output_path: Optional path to save CSV file
            
        Returns:
            DataFrame with generated data
        """
        data = []
        
        for i in range(num_samples):
            color, obj, attr = self.generate_pair()
            sentence = self.generate_sentence(color, obj, attr)
            
            data.append({
                'id': i,
                'sentence': sentence,
                'color': color,
                'object': obj,
                'attribute': attr,
                'template_id': i % len(self.TEMPLATES)
            })
        
        df = pd.DataFrame(data)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"✅ Dataset saved to {output_path}")
        
        return df
    
    def get_statistics(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate statistics about generated dataset.
        
        Args:
            df: DataFrame from generate_dataset()
            
        Returns:
            Dictionary with statistics
        """
        return {
            'total_samples': len(df),
            'unique_colors': df['color'].nunique(),
            'unique_objects': df['object'].nunique(),
            'unique_attributes': df['attribute'].nunique(),
            'color_distribution': df['color'].value_counts().to_dict(),
            'sample_sentences': df['sentence'].head(5).tolist()
        }


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Initializing color data generator...")
    
    # Create generator instance
    generator = ColorDataGenerator(seed=42)
    
    # Generate small test dataset
    print("Generating 10 test sentences...")
    test_df = generator.generate_dataset(num_samples=10)
    
    # Show examples
    print("\n📝 Sample generated sentences:")
    for idx, row in test_df.iterrows():
        print(f"  {idx+1}. {row['sentence']}")
        print(f"     (Color: {row['color']}, Attribute: {row['attribute']})")
    
    # Generate full dataset
    print("\n📊 Generating full 5000-sample dataset...")
    output_dir = Path("data/raw")
    df = generator.generate_dataset(
        num_samples=5000,
        output_path=output_dir / "color_sentences_5000.csv"
    )
    
    # Show statistics
    stats = generator.get_statistics(df)
    print(f"\n📈 Dataset Statistics:")
    print(f"  Total sentences: {stats['total_samples']}")
    print(f"  Unique colors: {stats['unique_colors']}")
    print(f"  Unique objects: {stats['unique_objects']}")
    print(f"  Unique attributes: {stats['unique_attributes']}")
    print(f"\n  Color distribution:")
    for color, count in list(stats['color_distribution'].items())[:5]:
        print(f"    {color}: {count}")