from clip.model import build_model, tokenize, load_clip
import torch
from transformers import AutoTokenizer, CLIPTextModelWithProjection



def clip_cat_emb(cat_list, clip_model):


    # Load the model
    cat_emb_dict = {}
    for cat in cat_list:
        with torch.no_grad():
            tokens = tokenize(cat).to('cuda')
            _,text_emb, = clip_model.encode_text_with_embeddings(tokens)
        print(f"{cat}: {text_emb}")
        cat_emb_dict[cat] = text_emb.cpu().numpy()
    
    # caculate the cosine similarity of the cat in the dict
    for cat1 in cat_list:
        for cat2 in cat_list:
            if cat1 != cat2:
                cos = torch.nn.functional.cosine_similarity(
                    torch.tensor(cat_emb_dict[cat1], dtype=torch.float32), torch.tensor(cat_emb_dict[cat2],  dtype=torch.float32), dim=0)
                print(f"Cosine similarity between {cat1} and {cat2}: {cos.item()}")

def clip_cat_emb_patch32(cat_list, model, tokenizer):
    inputs = tokenizer(cat_list, padding=True, return_tensors="pt")
    text_embeds = model(**inputs).text_embeds

    cat_emb_dict = {}
    for i, cat in enumerate(cat_list):
        cat_emb_dict[cat] = text_embeds[i]
        print(f"{cat}: {text_embeds[i]}")

    # caculate the cosine similarity of the cat in the dict
    cat_cos_sim_dict = {}
    for cat1 in cat_list:
        for cat2 in cat_list:
            if cat1 != cat2:
                cos = torch.nn.functional.cosine_similarity(
                    torch.tensor(cat_emb_dict[cat1], dtype=torch.float32), torch.tensor(cat_emb_dict[cat2],  dtype=torch.float32), dim=0)
                print(f"{cat1} {cat2} cos similarity: {cos.item()}")
                cat_cos_sim_dict[(cat1, cat2)] = cos.item()
    
    # sort the cos similaity of the cat in the dict
    cat_cos_sim_dict = sorted(cat_cos_sim_dict.items(), key=lambda x: x[1], reverse=True)
    print("Cosine similarity of the cat in the dict:")
    for cat in cat_cos_sim_dict:
        print(f"{cat[0]} cos similarity: {cat[1]}")
    
    


if __name__ == "__main__":
    model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    cat_list = ["bottle", 'cup', "computer", 'monitor', 'display', 'controller', 'phone', 'remoter',\
                'remote control', 'keyboard', 'mouse', 'laptop', 'tablet', 'notebook', 'desktop', 'game console', 'gamepad',\
                'roundtable', 'trashbin', 'bowl', 'plate', 'mug', 'glass', 'chessboard', 'book', 'notebook', 'pencil', 'pen',\
                'eraser', 'coffe machine', "printer", "vase", 'plant']

    #inputs = tokenizer(, padding = True, return_tensors="pt")
    clip_cat_emb_patch32(cat_list, model, tokenizer)


