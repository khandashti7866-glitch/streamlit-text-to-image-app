import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# ------------------ Streamlit Page Setup ------------------
st.set_page_config(page_title="Text to Image Generator", page_icon="🎨")
st.title("🎨 Text to Image Generator (Stable Diffusion)")
st.write("Enter a prompt below and generate an image using AI!")

# ------------------ User Input ------------------
prompt = st.text_input("Enter your prompt:", "a magical forest with glowing mushrooms")

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)
    return pipe, device

pipe, device = load_model()

# ------------------ Generate Image ------------------
if st.button("Generate Image"):
    if not prompt.strip():
        st.warning("⚠️ Please enter a valid prompt.")
    else:
        st.info("🧠 Generating image... please wait 20–40 seconds...")
        with torch.autocast(device):
            image = pipe(prompt).images[0]
        st.image(image, caption="Generated Image", use_container_width=True)
        st.success("✅ Done! Enjoy your AI-generated art.")
