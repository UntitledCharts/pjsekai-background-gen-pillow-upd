from PIL import Image
import pjsk_background_gen_PIL
import io
import os


def test_render_outputs():
    if not os.path.exists("test_out"):
        os.mkdir("test_out")

    with open("tests/test.png", "rb") as f:
        image_bytes = f.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    # Test default render (render_v3)
    result = pjsk_background_gen_PIL.render(image)
    result.save("test_out/latest.png")
    assert result is not None

    # Test render_v3
    result_v3 = pjsk_background_gen_PIL.render_v3(image)
    result_v3.save("test_out/v3.png")
    assert result_v3 is not None

    # Test render_v1
    result_v1 = pjsk_background_gen_PIL.render_v1(image)
    result_v1.save("test_out/v1.png")
    assert result_v1 is not None
