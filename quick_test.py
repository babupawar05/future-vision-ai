import os
from PIL import Image
from google import genai

API_KEY = "AQ.Ab8RN6Kxz-Ts6nWnDDrvwRbU8H40x47NO96XzfNuHX7SXjDQxw"

client = genai.Client(api_key=API_KEY)

image_path = "test_palm.jpg"

if not os.path.exists(image_path):
    print(f"Error: Could not find '{image_path}' in the current folder.")
    exit(1)

print("Sending palm image to Gemini 3.6 Flash...")
img = Image.open(image_path)

prompt = """
Act as an expert in traditional Indian palmistry (Hast Rekha Shastra).
Analyze the lines, mounts, and overall hand structure in this image carefully.
Write the response entirely in Hindi.
Cover:
1. हस्त स्वरूप एवं प्रकृति (Hand Structure & Nature)
2. जीवन रेखा (Life Line - Jeevan Rekha)
3. मस्तिष्क रेखा (Head Line - Mastishk Rekha)
4. हृदय रेखा (Heart Line - Hriday Rekha)
5. भाग्य रेखा (Fate Line - Bhagya Rekha)
6. प्रमुख पर्वत (Guru & Shukra Parvat)
7. आजीविका, करियर एवं धन (Career & Wealth Insights)
"""

try:
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[img, prompt]
    )
    print("\n==========================================")
    print("       हस्तरेखा विश्लेषण (HAST REKHA REPORT)       ")
    print("==========================================\n")
    print(response.text)
except Exception as e:
    print("\nAnalysis failed with error:", e)