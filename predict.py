import argparse
import json
import torch
from PIL import Image
from util import process_image, load_checkpoint
from collections import OrderedDict
import numpy as np

parser = argparse.ArgumentParser(description='Predict the type of a flower')
parser.add_argument('--checkpoint', type=str, help='Path to checkpoint' , default='checkpoint.pth')
parser.add_argument('--image_path', type=str, help='Path to file' , default='flowers/test/28/image_05230.jpg')
parser.add_argument('--gpu', type=bool, default=True, help='Whether to use GPU during inference or not')
parser.add_argument('--topk', type=int, help='Number of k to predict' , default=0)
parser.add_argument('--cat_to_name_json', type=str, help='Json file to load for class values to name conversion' , default='cat_to_name.json')
args = parser.parse_args()

image_path = args.image_path
with open(args.cat_to_name_json, 'r') as f:
    cat_to_name = json.load(f)

# : Load Checkpoint
model, checkpoint = load_checkpoint(args.checkpoint)

# : Process a PIL image for use in a PyTorch model    
im = Image.open(image_path)
processed_im = process_image(im)

def predict(image_path, model, topk=5, device='cuda'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # : Implement the code to predict the class from an image file
    im = Image.open(image_path)
    processed_im = process_image(im).unsqueeze(0)
    model.to(device)
    model.eval()    
    with torch.no_grad():
        processed_im = processed_im.to(device).float()
        output = model(processed_im)
        ps = torch.exp(output)
    pred = ps.topk(topk)
    flower_ids = pred[1][0].to('cpu')
    flower_ids = torch.Tensor.numpy(flower_ids)
    probs = pred[0][0].to('cpu')
    idx_to_class = {k:v for v,k in checkpoint['class_to_idx'].items()}
    flower_names = np.array([cat_to_name[idx_to_class[x]] for x in flower_ids])
        
    return probs, flower_names

# : Predict type and print
if args.topk:
    probs, flower_names = predict(image_path, model, args.topk,'cuda' if args.gpu else 'cpu')
    print('Probabilities of top {} flowers:'.format(args.topk))
    for i in range(args.topk):
        print('{} : {:.2f}'.format(flower_names[i],probs[i]))
else:
    probs, flower_names = predict(image_path, model)
    print('Flower is predicted to be {} with {:.2f} probability'.format(flower_names[0], probs[0]))