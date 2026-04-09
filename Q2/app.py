import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
import io
from collections import OrderedDict

# --- 1. U-Net Generator Architecture ---
class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c, down=True, use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False, padding_mode="reflect") if down 
            else nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True) if not down else nn.LeakyReLU(0.2, inplace=True)
        )
        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()

    def forward(self, x):
        return self.dropout(self.conv(x))

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(3, 64, 4, 2, 1, padding_mode="reflect")
        self.down2 = UNetBlock(64, 128)
        self.down3 = UNetBlock(128, 256)
        self.down4 = UNetBlock(256, 512)
        self.up1 = UNetBlock(512, 256, down=False, use_dropout=True)
        self.up2 = UNetBlock(512, 128, down=False)
        self.up3 = UNetBlock(256, 64, down=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4)
        u2 = self.up2(torch.cat([u1, d3], dim=1))
        u3 = self.up3(torch.cat([u2, d2], dim=1))
        return self.final_up(torch.cat([u3, d1], dim=1))

# --- 2. Load Model ---
@st.cache_resource
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator().to(device)
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.train() # CRITICAL: Keeps Batch Norm active for color filling
    return model, device

# --- 3. Streamlit Interface ---
st.set_page_config(page_title="AI Anime Colorizer", layout="wide")
st.title("🎨 AI Sketch Translation Final Polish")

st.sidebar.header("Processing Controls")
patch_removal = st.sidebar.slider("Patch Removal (Median Blur)", 0, 5, 2, help="Higher values remove the square artifacts.")
color_boost = st.sidebar.slider("Color Intensity", 1.0, 3.0, 1.8)
line_boldness = st.sidebar.slider("Line Boldness", 0, 3, 1)

uploaded_file = st.file_uploader("Upload Sketch", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)
    img = Image.open(uploaded_file).convert("RGB")
    
    # Pre-process: Thicken lines
    if line_boldness > 0:
        img_np = np.array(img)
        kernel = np.ones((line_boldness+1, line_boldness+1), np.uint8)
        img_np = cv2.erode(img_np, kernel, iterations=1) 
        img = Image.fromarray(img_np)

    with col1:
        st.subheader("Input (Bolded)")
        st.image(img, use_container_width=True)

    try:
        model, device = load_model("best_anime_generator.pth")
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        input_tensor = transform(img).unsqueeze(0).to(device)
        output_tensor = model(input_tensor)
        
        # Output Reconstruction
        output_np = (output_tensor.detach().squeeze(0).cpu().permute(1, 2, 0).numpy() * 0.5) + 0.5
        output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
        
        # Advanced Filtering (OpenCV)
        cv_res = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
        if patch_removal > 0:
            k_size = (patch_removal * 2) + 1 
            cv_res = cv2.medianBlur(cv_res, k_size) # Melts the square patterns
        
        # Sharpening to bring back hair details
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        cv_res = cv2.filter2D(cv_res, -1, sharpen_kernel)
        
        final_img = Image.fromarray(cv2.cvtColor(cv_res, cv2.COLOR_BGR2RGB))
        
        # Final Boost
        enhancer = ImageEnhance.Color(final_img)
        final_img = enhancer.enhance(color_boost)

        with col2:
            st.subheader("AI Final Result")
            st.image(final_img, use_container_width=True)
            
            buf = io.BytesIO()
            final_img.save(buf, format="PNG")
            st.download_button("📥 Save Image", buf.getvalue(), "output.png")

    except Exception as e:
        st.error(f"Error: {e}")