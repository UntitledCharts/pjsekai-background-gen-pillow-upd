from PIL import Image
import pjsk_background_gen_PIL
import io

with open("../test.png", "rb") as f:
    image_bytes = f.read()
image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

# Test default render (render_v3)
result = pjsekai_background_gen_core.render(image)
result.save("../../dist/latest.png")

# Test render_v3
result_v3 = pjsekai_background_gen_core.render_v3(image)
result_v3.save("../../dist/v3.png")

# Test render_v1
result_v1 = pjsekai_background_gen_core.render_v1(image)
result_v1.save("../../dist/v1.png")
