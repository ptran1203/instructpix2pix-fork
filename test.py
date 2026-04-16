from transformers import CLIPTokenizer, CLIPTextModel

CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

print("ok")