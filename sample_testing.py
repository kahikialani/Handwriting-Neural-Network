import torch
import pandas as pd
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
from nn_model import CRNN, HandwritingDataset, decode_predictions, clean_labels

def load_model(model_path, device):
    print(f"Loading model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)  # Changed variable name
    config = checkpoint['config']
    char_to_idx = checkpoint['char_to_idx']
    idx_to_char = checkpoint['idx_to_char']
    
    model = CRNN(
        img_height=config['IMG_HEIGHT'],
        img_width=config['IMG_WIDTH'],
        num_classes=config['NUM_CLASSES'],
        hidden_size=config['hidden_size']
    )
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])  # Use checkpoint, not model
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    if 'epoch' in checkpoint:
        print(f"Trained for {checkpoint['epoch']} epochs")
    if 'train_losses' in checkpoint and checkpoint['train_losses']:
        print(f"Final training loss: {checkpoint['train_losses'][-1]:.4f}")
   
    return model, char_to_idx, idx_to_char, config

def test_single_image(model, image_path, true_label, char_to_idx, idx_to_char, config, device):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((config['IMG_HEIGHT'], config['IMG_WIDTH'])),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5], std = [0.5])
    ])

    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        predictions = decode_predictions(outputs, idx_to_char)
        predicted_text = predictions[0]
        is_correct = predicted_text == true_label
    print(f"\n{'='*50}")
    print(f"{'='*50}")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"True Label: '{true_label}'")
    print(f"Prediction: '{predicted_text}'")
    print(f"Accuracy: {'CORRECT' if is_correct else 'INCORRECT'}")
    
    # Display image of the name if below is set to true
    display_image = True
    if display_image == True:
        plt.figure(figsize=(10, 4))
        plt.imshow(image, cmap='gray')
        title_color = 'green' if is_correct else 'red'
        plt.title(f"True: '{true_label}' | Predicted: '{predicted_text}'", 
                fontsize=14, color=title_color)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return predicted_text, is_correct

def sample_random_image(csv_path, img_dir):
    df = pd.read_csv(csv_path)
    df_clean = clean_labels(df)
    df_clean['IDENTITY'] = df_clean['IDENTITY'].str.upper()

    sample = df_clean.sample(1).iloc[0]
    filename = sample['FILENAME']
    true_label = sample['IDENTITY']
    image_path = os.path.join(img_dir, filename)
    return image_path, true_label
    

if __name__ == "__main__":
    model_path = 'crnn_model_final.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, char_to_idx, idx_to_char, config = load_model(model_path, device)
    print("Model loaded and ready for testing!")
    test_csv = 'data/written_name_test_v2.csv'
    test_img_dir = 'data/test_v2/test/'
    print('Testing on a random test image')
    image_path, true_label = sample_random_image(test_csv, test_img_dir)

    prediction, is_correct = test_single_image(
        model, image_path, true_label, char_to_idx, idx_to_char, config, device
    )
    check_csv = 'data/sample_data/quickbrown.csv'
    check_img_dir = 'data/sample_data/'
    image_path, true_label = sample_random_image(check_csv, check_img_dir)
    prediction, is_correct = test_single_image(
        model, image_path, true_label, char_to_idx, idx_to_char, config, device
    )
    correct = 0
    wrong = 0
    #for i in range(1):
    #    image_path, true_label = sample_random_image(test_csv, test_img_dir)
    #    prediction, is_correct = test_single_image(
    #        model, image_path, true_label, char_to_idx, idx_to_char, config, device
    #    )
    #    if is_correct == True:
    #        correct += 1
    #    elif is_correct == False:
    #        wrong += 1

#print(f"Correct: {correct}")
#print(f"Wrong: {wrong}")
#print(f"That makes it accurate {wrong/(wrong + correct) * 100} of the time")