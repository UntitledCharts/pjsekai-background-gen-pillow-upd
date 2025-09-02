from PIL import Image
from functools import lru_cache
from dataclasses import dataclass
from io import BytesIO
import importlib.resources as pkg_resources


def load_image(path: str) -> Image.Image:
    file_bytes = (
        pkg_resources.files("pjsk_background_gen_PIL")
        .joinpath("assets", path)
        .read_bytes()
    )
    return Image.open(BytesIO(file_bytes)).convert("RGBA")


@dataclass
class V3Assets:
    base: Image.Image
    bottom: Image.Image
    center_cover: Image.Image
    center_mask: Image.Image
    side_cover: Image.Image
    side_mask: Image.Image
    windows: Image.Image


@dataclass
class V1Assets:
    base: Image.Image
    side_mask: Image.Image
    center_mask: Image.Image
    mirror_mask: Image.Image
    frames: Image.Image


@lru_cache(maxsize=1)
def get_v3_assets() -> V3Assets:
    return V3Assets(
        base=load_image("../assets/v3/base.png"),
        bottom=load_image("../assets/v3/bottom.png"),
        center_cover=load_image("../assets/v3/center_cover.png"),
        center_mask=load_image("../assets/v3/center_mask.png"),
        side_cover=load_image("../assets/v3/side_cover.png"),
        side_mask=load_image("../assets/v3/side_mask.png"),
        windows=load_image("../assets/v3/windows.png"),
    )


@lru_cache(maxsize=1)
def get_v1_assets() -> V1Assets:
    return V1Assets(
        base=load_image("../assets/v1/base.png"),
        side_mask=load_image("../assets/v1/side_mask.png"),
        center_mask=load_image("../assets/v1/center_mask.png"),
        mirror_mask=load_image("../assets/v1/mirror_mask.png"),
        frames=load_image("../assets/v1/frames.png"),
    )
