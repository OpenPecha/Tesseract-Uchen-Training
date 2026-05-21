import multiprocessing as mp
from pathlib import Path
import lmdb
from tqdm import tqdm
import numpy as np
import cv2
import logging
import time

# Setup logging
logging.basicConfig(
    filename="dataset_processing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class OptimizedLmdbDataset:
    def __init__(
        self, root, output_path, charset, max_label_len,
        remove_whitespace=True, normalize_unicode=True,
        num_workers=mp.cpu_count(), chunk_size=1000, batch_size=50
    ):
        self.root = root
        self.output_path = Path(output_path)
        self.charset = set(charset)  # Convert to set for faster lookups
        self.max_label_len = max_label_len
        self.remove_whitespace = remove_whitespace
        self.normalize_unicode = normalize_unicode
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.batch_size = batch_size

    def _validate_label(self, label, idx):
        """Validate and process the label text."""
        if self.remove_whitespace:
            label = ''.join(label.split())
        
        if self.normalize_unicode:
            label = label.strip().replace('\u200b', '')  # Remove zero-width spaces
        
        if len(label) > self.max_label_len:
            logging.warning(f"Label too long for sample {idx}: {len(label)} > {self.max_label_len}")
            return None

        invalid_chars = set(label) - self.charset
        if invalid_chars:
            logging.warning(f"Invalid characters found in sample {idx}: {invalid_chars}")
            return None

        return label

    @staticmethod
    def _process_sample(txn, idx):
        try:
            img_key = f'image-{idx:09d}'.encode()
            label_key = f'label-{idx:09d}'.encode()
            path_key = f'path-{idx:09d}'.encode()

            imgbuf = txn.get(img_key)
            if not imgbuf:
                raise ValueError(f"Missing image for index {idx}")

            # Use OpenCV for faster image processing
            img_array = np.frombuffer(imgbuf, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to decode image for index {idx}")

            label = txn.get(label_key).decode() if txn.get(label_key) else ""
            path = txn.get(path_key).decode() if txn.get(path_key) else f"sample_{idx}.png"

            return idx, img, label, path
        except Exception as e:
            logging.error(f"Error processing index {idx}: {e}")
            return None

    def _save_batch(self, batch):
        for idx, img, label, path in batch:
            try:
                # Validate label before saving
                processed_label = self._validate_label(label, idx)
                if processed_label is None:
                    continue

                name = Path(path).stem
                img_path = self.output_path / 'images' / f'{name}.png'
                label_path = self.output_path / 'labels' / f'{name}.txt'

                img_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(img_path), img):
                    raise ValueError(f"Failed to save image for index {idx}")

                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text(processed_label)
            except Exception as e:
                logging.error(f"Error saving sample {idx}: {e}")

    def _process_chunk(self, args):
        root, indices = args
        env = lmdb.open(root, readonly=True, lock=False)
        batch = []

        with env.begin() as txn:
            for idx in indices:
                result = self._process_sample(txn, idx)
                if result:
                    batch.append(result)

                # Save in batches
                if len(batch) >= self.batch_size:
                    self._save_batch(batch)
                    batch.clear()

        # Save remaining items
        if batch:
            self._save_batch(batch)
        env.close()
        logging.info(f"Finished processing chunk with indices: {indices[0]} to {indices[-1]}")

    def process_dataset(self):
        start_time = time.time()

        env = lmdb.open(self.root, readonly=True, lock=False)
        with env.begin() as txn:
            num_samples = int(txn.get('num-samples'.encode()).decode())
        env.close()

        chunks = np.array_split(range(num_samples), 
                              max(num_samples // self.chunk_size, 1))
        chunk_args = [(self.root, chunk) for chunk in chunks]

        with mp.Pool(self.num_workers) as pool:
            list(tqdm(pool.imap(self._process_chunk, chunk_args), 
                     total=len(chunks)))

        elapsed_time = time.time() - start_time
        logging.info(f"Processing completed in {elapsed_time:.2f} seconds. Total samples: {num_samples}")

def process_dataset(dataset='norbu'):
    input_path = f'/Dataset/ml-artifacts/monlam.ai.ocr/training_lmdb/train/real/{dataset}'
    output_path = f'/Dataset/ml-artifacts/monlam.ai.ocr/yonten_norbu_bucket/{dataset}/train'

    charset = "ཡིད་གསུམཕརབཐོལའྱཤ །ྡྗེཆངཀཉནཔཟླཙ༑ཏཁྒྣྲ༌ྔྷྭཞྐྙཅྟྤྩཚ༈ཛཇྫྨྦཨཱཾཧཎ༄༅ཝྕ༔ཥྜྋཌ༼ཊ༴ྪཻཿ༽ༀྚཀྵ࿄༎༉࿚ཽ྾༒ྺ༐࿙༜༆࿅ྀ྇ྰྠྃྵཪྴྛྞ༵༞ྻ༸྅༷ྶྥྑཋ༏྄༝࿐༹ྂ࿉༙༗ཬ"

    processor = OptimizedLmdbDataset(
        root=input_path,
        output_path=output_path,
        charset=charset,
        max_label_len=600,
        remove_whitespace=True,
        normalize_unicode=True,
        num_workers=mp.cpu_count(),
        chunk_size=1000,
        batch_size=50
    )
    processor.process_dataset()

if __name__ == '__main__':
    process_dataset('norbu')