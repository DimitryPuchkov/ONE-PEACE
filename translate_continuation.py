import pandas as pd
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import pipeline
import os
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")  # export HF_TOKEN before running
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from tqdm.auto import tqdm


DATASET_DIR = Path('./one_peace/dataset/')
DATASET_NAME = 'mscoco/mscoco/test_translated.tsv'
DATASET_NAME_BACKUP = 'mscoco/mscoco/test_translated_backup.tsv'
df = pd.read_csv(DATASET_DIR/DATASET_NAME, sep='\t')
pipe = pipeline("translation", model="./nllb")

start_index = df[df['caption'].notnull()].index.max() + 1 if df['caption'].notnull().any() else 0
try:
    for i, row in tqdm(df[start_index:].iterrows(), total=len(df), initial=int(start_index)):
        
            df.loc[i, 'caption'] = pipe(row['en_caption'], src_lang='eng_Latn', tgt_lang='rus_Cyrl')[0]['translation_text']
            if i % 100 == 0:
                df.to_csv(DATASET_DIR/DATASET_NAME_BACKUP, sep='\t', index=False)
except KeyboardInterrupt as e:
    print(f"stopping...: {e}")
    df.to_csv(DATASET_DIR/DATASET_NAME, sep='\t', index=False)
    print(f"Progress saved to {DATASET_NAME}")

print("Translation completed. Saving final output...")
df.to_csv(DATASET_DIR/DATASET_NAME, sep='\t', index=False)
print(f"Final output saved to {DATASET_NAME}")