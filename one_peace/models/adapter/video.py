import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPVisionModel
from fairseq.modules import FairseqDropout
import logging
import cv2
import numpy as np
from PIL import Image
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

from ..components import trunc_normal_

logging.getLogger("pyscenedetect").setLevel(logging.WARNING)

def make_video_bucket_position(bucket_size, num_frames, num_relative_distance):
    coords_t = torch.arange(num_frames)
    coords_h = torch.arange(bucket_size)
    coords_w = torch.arange(bucket_size)
    coords = torch.stack(torch.meshgrid(coords_t, coords_h, coords_w, indexing="ij"))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()

    relative_coords[:, :, 0] += num_frames - 1
    relative_coords[:, :, 1] += bucket_size - 1
    relative_coords[:, :, 2] += bucket_size - 1

    relative_coords[:, :, 0] *= (2 * bucket_size - 1) ** 2
    relative_coords[:, :, 1] *= (2 * bucket_size - 1)

    relative_position_index = relative_coords.sum(-1)
    num_relative_distance = (2 * num_frames - 1) * (2 * bucket_size - 1) ** 2 + 3
    rp_bucket = torch.zeros((relative_coords.size(0) + 1, relative_coords.size(1) + 1), dtype=torch.long)
    rp_bucket[1:, 1:] = relative_position_index + 1
    rp_bucket[0, 0:] = num_relative_distance - 3
    rp_bucket[0:, 0] = num_relative_distance - 2
    rp_bucket[0, 0] = num_relative_distance - 1

    return rp_bucket


class VideoAdapter(nn.Module):
    def __init__(self, cfg, embed_dim, attention_heads, num_layers=None):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = embed_dim
        self.num_layers = num_layers or 1
        self.num_frames = cfg.num_frames
        self.bucket_size = cfg.bucket_size

        # CLIP backbone & processor
        self.clip_model = CLIPVisionModel.from_pretrained(cfg.clip_model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(cfg.clip_model_name)
        if self.clip_model.config.hidden_size != embed_dim:
            self.proj = nn.Linear(self.clip_model.config.hidden_size, embed_dim)
        else:
            self.proj = nn.Identity()

        if cfg.freeze_clip_weights:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.clip_model.eval()
        
        # Positional embeddings
        total_patches = self.num_frames * (self.bucket_size ** 2)
        self.pos_embed = nn.Parameter(torch.zeros(1, total_patches + 1, embed_dim))
        trunc_normal_(self.pos_embed)
        self.cls_embedding = nn.Parameter(torch.zeros(1,1,embed_dim))
        trunc_normal_(self.cls_embedding)

        # CLS and optional type embedding
        if cfg.add_type_embedding:
            self.type_embedding = nn.Parameter(torch.zeros(1,1,embed_dim))
            trunc_normal_(self.type_embedding)
        else:
            self.type_embedding = None

        # LayerNorm & dropout
        self.layernorm_embedding = nn.LayerNorm(embed_dim) if cfg.layernorm_embedding else None
        self.dropout_module = FairseqDropout(cfg.dropout)

        # Relative position bias
        if cfg.use_attn_bias:
            num_rel = (2 * self.num_frames - 1) * (2 * self.bucket_size - 1) ** 2 + 3
            rp_bucket = make_video_bucket_position(self.bucket_size, self.num_frames, num_rel)
            self.register_buffer("rp_bucket", rp_bucket)
            self.rel_pos_table_list = nn.ModuleList([
                nn.Embedding(num_rel, attention_heads)
                for _ in range(self.num_layers)
            ])
        else:
            self.rel_pos_table_list = None

        # Temporal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=attention_heads,
            dim_feedforward=embed_dim * 4,
            dropout=cfg.dropout,
            activation="gelu"
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.temporal_layers)
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, self.num_frames, embed_dim))
        trunc_normal_(self.temporal_pos_embed)

        # Adapters: TAM, diff, align
        self.tam = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim // 4, kernel_size=3, padding=1, groups=embed_dim // 4),
            nn.GELU(),
            # pointwise conv: in_channels=embed_dim//4 → out_channels=embed_dim
            nn.Conv1d(embed_dim // 4, embed_dim, kernel_size=1, groups=1)
        )
        self.diff_proj = nn.Linear(embed_dim, embed_dim)
        self.align_attn = nn.MultiheadAttention(embed_dim, attention_heads, dropout=cfg.dropout)

        # LoRM & async warp
        self.lorm_scale = nn.Parameter(torch.ones(embed_dim))
        self.lorm_shift = nn.Parameter(torch.zeros(embed_dim))
        self.async_warp = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)

        # Q-Former
        num_queries = cfg.q_former_num_queries
        self.query_tokens = nn.Parameter(torch.randn(num_queries, embed_dim))
        self.query_attn = nn.MultiheadAttention(embed_dim, attention_heads, dropout=cfg.dropout)

        # Multi-grained head
        #self.frame_word_proj = nn.Linear(embed_dim, embed_dim)
        #self.aosm = nn.Sequential(nn.Linear(embed_dim * 2, 1), nn.Sigmoid())

    def read_video_segmented(
        self,
        video_path: str,
        num_per_scene: int = 8,
        clusters_per_scene: int = 4
    ) -> list[Image.Image]:
        """
        Segment the video into scenes, sample and cluster frames per scene,
        then enforce exactly self.num_frames total by uniform up/down sampling.
        """
        # 1) Handle missing path
        if video_path is None:
            return [Image.new("RGB", (224, 224))] * self.num_frames

        # 2) Scene detection
        #video_manager = VideoManager([video_path])
        video_stream = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())
        #video_manager.start()
        #scene_manager.detect_scenes(frame_source=video_manager)
        scene_manager.detect_scenes(video_stream)
        scenes = scene_manager.get_scene_list()
        #video_manager.release()

        cap = cv2.VideoCapture(video_path)
        all_selected = []

        # 3) For each scene: uniform sample & cluster
        for start, end in scenes:
            # total frames in scene
            count = end.frame_num - start.frame_num + 1
            # sample up to num_per_scene evenly spaced frames
            indices = np.linspace(
                start.frame_num,
                end.frame_num,
                min(count, num_per_scene),
                dtype=int
            )
            scene_frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                scene_frames.append(Image.fromarray(frame))

            if scene_frames:
                # cluster → clusters_per_scene keyframes
                embs = self.get_frame_embeddings(scene_frames).cpu().numpy()
                kmeans = KMeans(n_clusters=min(clusters_per_scene, len(embs)), random_state=0)
                labels = kmeans.fit_predict(embs)
                centers = kmeans.cluster_centers_
                for c in centers:
                    # pick closest to each centroid
                    idx = int(((embs - c) ** 2).sum(axis=1).argmin())
                    all_selected.append(scene_frames[idx])

        cap.release()

        # 4) Enforce exactly self.num_frames
        N = len(all_selected)
        M = self.num_frames
        if N == 0:
            # fallback to blanks
            return [Image.new("RGB", (224,224))] * M
        elif N >= M:
            # down‐sample uniformly
            pick = np.linspace(0, N - 1, M, dtype=int)
            return [all_selected[i] for i in pick]
        else:
            # up‐sample by repeating last frame
            extra = M - N
            return all_selected + [all_selected[-1]] * extra

    def read_video_segmented_light(self, video_path: str, frame_indices: list[int]) -> list[Image.Image]:
        """
        If `frame_indices` is provided, load exactly those frames;
        otherwise uniformly sample `num_frames` frames across the video.
        """
        # 1) blank‐frame factory
        M = self.num_frames
        # 2) open video
        cap = cv2.VideoCapture(video_path)
        # 2) Get total frame count
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            print("no video")
            return [Image.new("RGB", (224, 224))] * M

        # 3) decide which frame indices to pull
        #if frame_indices is None:
            # uniform fallback
            #idxs = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)
        #else:
            # use your precomputed list (clamp to valid range)
            #idxs = [min(max(0, i), total - 1) for i in frame_indices]

        # 4) load each requested frame
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frm = cap.read()
            if not ret or frm is None:
                frames.append(Image.new("RGB", (224, 224)))
            else:
                frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frm))

        cap.release()
        return frames
    
    

    def get_frame_embeddings(self, frames):
        inputs = self.image_processor(images=frames, return_tensors="pt").to(self.clip_model.device)
        with torch.no_grad():
            out = self.clip_model(**inputs).last_hidden_state
        cls_feats = out[:, 0, :]
        return cls_feats.to(torch.float32)

    def select_diverse_frames(self, frames, num_clusters):
        embs = self.get_frame_embeddings(frames).detach().cpu().float().numpy()
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embs)
        centers = kmeans.cluster_centers_
        selected = []
        for c in centers:
            idx = ((embs - c)**2).sum(axis=1).argmin()
            selected.append(frames[idx])
        return selected

    def forward(self, frames_batch, preserve_ids=None, preserve_embed=None, mask_token=None, is_second_video=False):
        device = next(self.parameters()).device
        B = len(frames_batch)
        #B = len(list_video_paths)
        # 1) Load frames
        #print(list_video_paths[0])
        #frames_batch = [self.read_video_segmented_light(video_path = p, frame_indices = d) for p, d in list_video_paths]
        
        # 2) Frame patch extraction
        all_pix = []
        for frames in frames_batch:
            inp = self.image_processor(images=frames, return_tensors='pt')
            all_pix.append(inp['pixel_values'])
        pix = torch.stack(all_pix,0).to(device)      # [B,T,3,H,W]
        b,t,c,h,w = pix.shape
        flat = pix.view(b*t, c, h, w)
        with torch.no_grad():
            out = self.clip_model(flat).last_hidden_state[:,1:,:]
        feats = self.proj(out).view(b, t, -1, self.embed_dim)  # [B,T,P,D]

        # flatten frames x patches
        frame_emb = feats.mean(dim=2)  # [B,T,D]

        if preserve_ids is not None and preserve_embed is not None:
            # preserve_embed: [B, T, D] full original embeddings
            # preserve_ids: [B, num_preserve] indices to keep
            # build a B×T mask of where to inject:
            keep = torch.zeros(b, t, device=device, dtype=torch.bool)
            keep.scatter_(1, preserve_ids.unsqueeze(-1), True)
            # swap in the provided embeddings
            frame_emb = torch.where(
                keep.unsqueeze(-1),
                preserve_embed,       # use the passed‑in embedding
                frame_emb             # or the CLIP‑computed one
            )

        # DCL masking: replace masked indices with mask_token
        if preserve_ids is not None and mask_token is not None:
            # create boolean mask for sequence of length T
            mask_bool = torch.ones(b, t, dtype=torch.bool, device=device)
            mask_bool.scatter_(1, preserve_ids.unsqueeze(-1), False)
            # apply mask
            frame_emb = torch.where(
                mask_bool.unsqueeze(-1),
                mask_token,  # mask_token passed in or self.mask_token
                frame_emb
            )

        # positional + temporal encoding
        x = frame_emb + self.temporal_pos_embed
        x = self.temporal_encoder(x.transpose(0,1)).transpose(0,1)  # [B,T,D]

        # TAM + diff + align
        diff = F.pad(torch.abs(x[:,1:]-x[:,:-1]), (0,0,1,0))
        mot = self.diff_proj(diff)
        tam = self.tam(x.transpose(1,2)).transpose(1,2)
        enh = x + mot + tam
        attn_out,_ = self.align_attn(enh.transpose(0,1), enh.transpose(0,1), enh.transpose(0,1))
        seg_final = attn_out.transpose(0,1)

        # Q-former pooling
        q = self.query_tokens.unsqueeze(1).expand(-1,B,-1)  # [Q,B,D]
        flat_patches = feats.reshape(b, t*feats.size(2), self.embed_dim).transpose(0,1)
        q_out,_ = self.query_attn(q, flat_patches, flat_patches)
        pooled = q_out.transpose(0,1)  # [B,Q,D]

        # Build adapter embedding
        cls = self.cls_embedding.expand(B,-1,-1)
        y = torch.cat([cls, pooled, seg_final], dim=1)  # [B,1+Q+T, D]

        # absolute pos
        seq = y.size(1)
        pos = self.pos_embed.expand(B,-1,-1)
        if pos.size(1)!=seq:
            p = pos.transpose(1,2)
            p = F.interpolate(p, size=seq, mode='linear', align_corners=False)
            pos = p.transpose(1,2)
        y = y + pos

        # type embedding
        if self.type_embedding is not None:
            y = y + self.type_embedding.expand_as(y)
            if is_second_video:
                y = y + self.type_embedding

        # layernorm + dropout
        if self.layernorm_embedding is not None:
            y = self.layernorm_embedding(y)
        y = self.dropout_module(y)

        # LoRM + async
        y = y * self.lorm_scale + self.lorm_shift
        y = self.async_warp(y.transpose(1,2)).transpose(1,2)

        # relative bias
        attn_bias=None
        if self.rel_pos_table_list is not None:
            rp = self.rp_bucket[:seq,:seq].to(y.device)
            attn_bias=[emb(rp).permute(2,0,1).unsqueeze(0).expand(B,-1,seq,seq)
                        for emb in self.rel_pos_table_list]

        padding_mask = torch.zeros(B, seq, dtype=torch.bool, device=device)
        return y, padding_mask, attn_bias

    def extract_frame_embeddings(self, pixel_values):
        B, T, C, H, W = pixel_values.size()
        flat = pixel_values.view(-1, C, H, W)
        with torch.no_grad():
            out = self.clip_model(flat).last_hidden_state  # [B*T, P+1, Hs]
        patches = out[:, 1:, :]
        patches = self.proj(patches)
        P = patches.size(1)
        return patches.view(B, T, P, self.embed_dim)

    def get_rel_pos_bias(self, bsz, seq_len):
        rel_pos_bias_list = []
        rp = self.rp_bucket[:seq_len, :seq_len].to(self.pos_embed.device)
        for table in self.rel_pos_table_list:
            v = table(rp).permute(2, 0, 1)  # [heads, L, L]
            v = v.unsqueeze(0).expand(bsz, -1, -1, -1)
            rel_pos_bias_list.append(v)
        return rel_pos_bias_list

    def gather_features(self, adapter_embedding, pos_embed, self_attn_bias_list, position_ids):
        """
        Reorders embeddings/positional biases according to position_ids.
        preserve_ids maps each output position to an input index, or -1.
        """
        bsz, seq_len, dim = adapter_embedding.size()
        gather_len = position_ids.size(1)

        # Expand position_ids for embedding gather
        # shape [B, gather_len, dim]
        gather_idx = position_ids[:, :, None].expand(-1, -1, dim)
        adapter_embedding = adapter_embedding.gather(1, gather_idx)
        pos_embed = pos_embed.gather(1, gather_idx)

        # Handle relative position biases if present
        if self_attn_bias_list is not None:
            new_bias_list = []
            # prepare row & col indices for 4D bias gathering
            # shape [B, num_heads, gather_len, seq_len]
            row_idx = position_ids[:, None, :, None].expand(
                -1, self_attn_bias_list[0].size(1), -1, seq_len
            )
            # shape [B, num_heads, gather_len, gather_len]
            col_idx = position_ids[:, None, None, :].expand(
                -1, self_attn_bias_list[0].size(1), gather_len, -1
            )
            for bias in self_attn_bias_list:
                # bias shape [B, num_heads, seq_len, seq_len]
                b = bias.gather(2, row_idx)   # gather rows
                b = b.gather(3, col_idx)      # then gather columns
                new_bias_list.append(b)
            self_attn_bias_list = new_bias_list

        return adapter_embedding, pos_embed, self_attn_bias_list

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Ensure compatibility when loading checkpoints:
        - Remap old rel_pos_table -> rel_pos_table_list
        - Interpolate or copy position embeddings / relative biases if shapes differ
        - Fill in any new parameters
        """
        prefix = name + "." if name else ""

        # 1) Handle old single rel_pos_table → new list
        old_key = prefix + "rel_pos_table.weight"
        if old_key in state_dict:
            state_dict[prefix + "rel_pos_table_list.0.weight"] = state_dict[old_key]
            del state_dict[old_key]

        # 2) Resize or copy relative position bias tables
        key0 = prefix + "rel_pos_table_list.0.weight"
        if key0 in state_dict:
            old = state_dict[key0]
            num_rel = (2 * self.num_frames - 1) * (2 * self.bucket_size - 1) ** 2 + 3
            if old.size(0) != num_rel:
                # simple up/down-sample: linear interpolation
                num_heads = old.size(1)
                # separate extras (last 3 tokens)
                main, extras = old[:-3], old[-3:]
                # reshape main to 3D
                L = int(round(main.size(0) ** (1/3)))
                main = main.view(num_heads, L, L, L).unsqueeze(0)  # [1, H, L, L, L]
                target = (2 * self.num_frames - 1, 2 * self.bucket_size - 1, 2 * self.bucket_size - 1)
                main = F.interpolate(main, size=target, mode="trilinear", align_corners=False)
                main = main.squeeze(0).permute(1, 2, 3, 0).reshape(-1, num_heads)
                state_dict[key0] = torch.cat([main, extras], dim=0)

        # 3) Copy remapped table to all layers if needed
        if self.rel_pos_table_list and len(self.rel_pos_table_list) > 1:
            base = state_dict.get(prefix + "rel_pos_table_list.0.weight")
            for i in range(len(self.rel_pos_table_list)):
                k = prefix + f"rel_pos_table_list.{i}.weight"
                if k not in state_dict and base is not None:
                    state_dict[k] = base.clone()

        # 4) Resize absolute positional embeddings if shape mismatch
        pos_key = prefix + "pos_embed"
        if pos_key in state_dict:
            old = state_dict[pos_key]
            new = self.pos_embed
            if old.size(1) != new.size(1):
                # separate CLS and patch
                cls_old, patch_old = old[:, :1, :], old[:, 1:, :]
                total_old = patch_old.size(1)
                # assume cubic grid: frames * bucket^2 = total_old
                f_old = self.num_frames
                b_old = int((total_old // f_old) ** 0.5)
                patch_old = patch_old.view(1, f_old, b_old, b_old, -1).permute(0, 4, 1, 2, 3)
                # interpolate
                patch_old = F.interpolate(
                    patch_old,
                    size=(self.num_frames, self.bucket_size, self.bucket_size),
                    mode="trilinear", align_corners=False
                )
                patch_old = patch_old.permute(0, 2, 3, 4, 1).reshape(1, -1, self.embed_dim)
                state_dict[pos_key] = torch.cat([cls_old, patch_old], dim=1)

        # 5) Initialize any missing params
        for name_param, tensor in self.state_dict().items():
            full = prefix + name_param
            if full not in state_dict:
                logger.info(f"Initializing missing parameter: {full}")
                state_dict[full] = tensor

        return state_dict
