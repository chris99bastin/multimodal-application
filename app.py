import gradio as gr
import logging
from transformers import MarianMTModel, MarianTokenizer
from diffusers import StableDiffusionPipeline,DDIMScheduler
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Setup logging for performance tracking
logging.basicConfig(level=logging.INFO)

# 1. Translation Model (Tamil to English)
translation_model_name = 'Helsinki-NLP/opus-mt-mul-en'  # Multilingual model for Tamil-English
tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translation_model = MarianMTModel.from_pretrained(translation_model_name)

# 2. Stable Diffusion Model for Image Generation
# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16" if device == "cuda" else "main")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

# 3. LLM (GPT-2) for Creative Text Generation
gpt_model_name = 'gpt2'  # You can also use GPT-3 or other LLMs via API
gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model_name)

# Translate Tamil to English
def translate_tamil_to_english(tamil_text):
    logging.info("Starting Tamil-to-English translation...")
    inputs = tokenizer(tamil_text, return_tensors="pt", padding=True)
    translated = translation_model.generate(**inputs)
    english_translation = tokenizer.decode(translated[0], skip_special_tokens=True)
    logging.info(f"Translation completed: {english_translation}")
    return english_translation

# Generate image from English text with optimized parameters
def generate_image_from_text(english_text):
    logging.info("Starting image generation with higher quality...")
    
    # Increase resolution and modify parameters for better quality
    image = pipe(
        english_text,
        height=512,  # Increase resolution to 512x512
        width=512,
        num_inference_steps=20,  # Increase number of steps for better details
        guidance_scale=7.5  # Adjust guidance scale to balance creativity and fidelity
    ).images[0]
    
    logging.info("Image generation completed.")
    return image

# Generate creative content using LLM (GPT-2)
def generate_creative_text(english_text):
    # Adding context for better creativity
    prompt = f"Write a creative story or text based on the following: {english_text}"
    
    inputs = gpt_tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    outputs = gpt_model.generate(
        inputs,
        max_length=150,  # Increased max length for a more complete story
        temperature=1.2,  # More creativity
        top_k=50,
        top_p=0.9,
        num_return_sequences=1,
        no_repeat_ngram_size=2
    )
    
    creative_text = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return creative_text


# Combined process (translate, generate image, and creative text)
def process_tamil_to_image_and_creative_text(tamil_input):
    try:
        # Step 1: Translate
        english_translation = translate_tamil_to_english(tamil_input)
        
        # Step 2: Generate image
        generated_image = generate_image_from_text(english_translation)
        
        # Step 3: Generate creative text
        creative_text = generate_creative_text(english_translation)
        
        return english_translation, generated_image, creative_text
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        return "Error during processing", None, "Error during processing"

# Gradio Interface
iface = gr.Interface(
    fn=process_tamil_to_image_and_creative_text,
    inputs="text",
    outputs=["text", "image", "text"],
    title="A Multimodal Application for Vernacular Language Translation and Image Synthesisn",
    description="Translates Tamil text to English, generates an image using Stable Diffusion with higher quality, and provides creative text using GPT-2.",
    examples=[["உழவுக்கும் தொழிலுக்கும் முன் இழுக்கை"], ["இந்தியா ஒரு பெரிய நாடாகும்"]]
)

# Launch Gradio App
if __name__ == "__main__":
    iface.launch(share=True)
