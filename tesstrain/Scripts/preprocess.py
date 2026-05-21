import os
from PIL import Image
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

class ImageProcessor:
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    
    def __init__(self, log_file: str = 'processing.log'):
        self._setup_logging(log_file)
        
    def _setup_logging(self, log_file: str) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        
    def _init_directories(self, paths: Dict[str, str], prefix: str) -> Optional[Dict[str, Path]]:
        """Initialize and validate all directory paths."""
        dirs = {
            'input_image': Path(paths['input_image_dir']),
            'input_label': Path(paths['input_label_dir']),
            'output_image': Path(paths['output_image_dir']),
            'output_text': Path(paths['output_text_dir']),
            'prefix': prefix
        }
        
        # Validate input directories
        for key in ['input_image', 'input_label']:
            if not dirs[key].exists():
                logging.error(f"{key} directory does not exist: {dirs[key]}")
                return None
                
        # Create output directories
        for key in ['output_image', 'output_text']:
            dirs[key].mkdir(exist_ok=True)
            
        return dirs
        
    def _get_image_files(self, image_dir: Path) -> List[Path]:
        """Get all image files using case-insensitive matching."""
        image_files = []
        for pattern in self.SUPPORTED_FORMATS:
            image_files.extend(image_dir.glob(f"*{pattern}"))
            image_files.extend(image_dir.glob(f"*{pattern.upper()}"))
        return image_files
        
    def process_file_pair(self, image_file: Path, dirs: Dict[str, Path]) -> bool:
        """Process a single image-text pair."""
        try:
            prefix_name = f"{dirs['prefix']}_{image_file.stem}"
            
            # Process image
            output_image = dirs['output_image'] / f"{prefix_name}.png"
            Image.open(image_file).save(output_image)
            
            # Process text
            label_file = dirs['input_label'] / f"{image_file.stem}.txt"
            if label_file.exists():
                output_text = dirs['output_text'] / f"{prefix_name}.gt.txt"
                output_text.write_text(label_file.read_text(encoding='utf-8'), encoding='utf-8')
                
            return True
            
        except Exception as e:
            logging.error(f"Error processing {image_file}: {e}")
            return False
            
    def process_directory(self, input_image_dir: str, input_label_dir: str,
                         output_image_dir: str, output_text_dir: str, prefix: str) -> None:
        """Process all image-text pairs in parallel."""
        paths = {
            'input_image_dir': input_image_dir,
            'input_label_dir': input_label_dir,
            'output_image_dir': output_image_dir,
            'output_text_dir': output_text_dir
        }
        
        dirs = self._init_directories(paths, prefix)
        if not dirs:
            return
            
        image_files = self._get_image_files(dirs['input_image'])
        if not image_files:
            logging.error(f"No supported image files found in {input_image_dir}")
            return
            
        # Process files in parallel
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda x: self.process_file_pair(x, dirs),
                image_files
            ))
            
        logging.info(f"Processed {sum(results)} out of {len(image_files)} files.")

def main():
    processor = ImageProcessor()
    processor.process_directory(
        input("Enter the path to the input image directory: ").strip(),
        input("Enter the path to the input label directory: ").strip(),
        input("Enter the path to the output image directory: ").strip(),
        input("Enter the path to the output text directory: ").strip(),
        input("Enter the prefix for the processed files: ").strip()
    )

if __name__ == "__main__":
    main()