import torch 
from PIL import Image

from ..base_dataset import BaseDataset

class VideoTextRetrievalDataset(BaseDataset): 
    def __init__( self, split, dataset, bpe, dictionary, max_src_length=70, num_frames=16 ): 
        super().__init__(split, dataset, bpe, dictionary) 
        self.max_src_length = max_src_length 
        self.num_frames = num_frames

    def __getitem__(self, index, item_tuple=None):
        # Expected tuple: (uniq_id, video_path, caption)
        item_tuple = self.dataset[index] if item_tuple is None else item_tuple
        uniq_id, video, caption = item_tuple
        if uniq_id is not None:
            uniq_id = int(uniq_id) if isinstance(uniq_id, int) or (isinstance(uniq_id, str) and uniq_id.isdigit()) else uniq_id

        # Process caption: tokenize and encode.
        caption = self.process_text(caption)
        text_src_item = self.encode_text(' {}'.format(caption), self.max_src_length)

        # Read video frames as raw PIL Images.
        if video is not None:
            frames = self.read_video(video, self.num_frames)
        else:
            frames = [Image.new("RGB", (224, 224)) for _ in range(self.num_frames)]

        example = {
            "id": uniq_id,
            "source_text": text_src_item,
            "source_video": frames  # List of raw PIL Images.
        }
        return example