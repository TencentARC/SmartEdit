from .clip_encoder import CLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    if vision_tower.startswith("openai") or vision_tower.startswith("laion") or vision_tower.startswith("/group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
