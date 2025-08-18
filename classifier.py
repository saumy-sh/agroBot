import torch
import torch.nn as nn
from torchvision import transforms, models
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import json

class MobileNetV2Predictor:
    def __init__(self, model_path, device=None):
        """
        Initialize the MobileNetV2 predictor
        
        Args:
            model_path (str): Path to the saved fine-tuned model
            device (str): Device to run inference on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and class names
        self._load_model()
        
        # Setup transforms
        self._setup_transforms()
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Number of classes: {len(self.class_names)}")
        print(f"Class names: {self.class_names}")
    
    def _load_model(self):
        """Load the fine-tuned model"""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Get class names from checkpoint
            if 'class_names' in checkpoint:
                self.class_names = checkpoint['class_names']
            else:
                # If class names not saved, you'll need to provide them manually
                print("Warning: Class names not found in checkpoint. Using default names.")
                # You can modify this based on your classes
                self.class_names = [f'class_{i}' for i in range(len(checkpoint['model_state_dict']['classifier.1.weight']))]
            
            self.num_classes = len(self.class_names)
            
            # Create model architecture
            self.model = models.mobilenet_v2(pretrained=False)
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1280, self.num_classes)
            )
            
            # Load trained weights
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            # Move to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            raise Exception(f"Error loading model: {e}")
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict_single_image(self, image_path, return_probabilities=False):
        """
        Predict class for a single image
        
        Args:
            image_path (str): Path to the image
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            dict: Prediction results
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
                
                predicted_class = self.class_names[predicted_idx.item()]
                confidence_score = confidence.item()
            
            result = {
                'image_path': image_path,
                'predicted_class': predicted_class,
                'confidence': confidence_score,
                'predicted_index': predicted_idx.item()
            }
            
            if return_probabilities:
                prob_dict = {self.class_names[i]: prob.item() 
                           for i, prob in enumerate(probabilities)}
                result['class_probabilities'] = prob_dict
            
            return result
            
        except Exception as e:
            return {
                'image_path': image_path,
                'error': str(e)
            }
    
    def predict_folder(self, folder_path, output_csv=None, return_probabilities=False, 
                      show_progress=True, image_extensions=None):
        """
        Predict classes for all images in a folder
        
        Args:
            folder_path (str): Path to folder containing images
            output_csv (str): Path to save results as CSV (optional)
            return_probabilities (bool): Whether to include class probabilities
            show_progress (bool): Whether to show progress
            image_extensions (list): List of image extensions to process
            
        Returns:
            list: List of prediction results
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        # Get all image files
        folder_path = Path(folder_path)
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(folder_path.glob(f'*{ext}'))
            image_files.extend(folder_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No images found in {folder_path}")
            return []
        
        print(f"Found {len(image_files)} images to process")
        
        results = []
        
        for i, image_path in enumerate(image_files):
            if show_progress and i % 10 == 0:
                print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
            
            result = self.predict_single_image(
                str(image_path), 
                return_probabilities=return_probabilities
            )
            results.append(result)
        
        # Save to CSV if requested
        if output_csv:
            self._save_results_to_csv(results, output_csv, return_probabilities)
        
        return results
    
    def _save_results_to_csv(self, results, output_path, include_probabilities=False):
        """Save results to CSV file"""
        try:
            # Prepare data for CSV
            csv_data = []
            
            for result in results:
                if 'error' in result:
                    row = {
                        'image_path': result['image_path'],
                        'predicted_class': 'ERROR',
                        'confidence': 0.0,
                        'error': result['error']
                    }
                else:
                    row = {
                        'image_path': result['image_path'],
                        'predicted_class': result['predicted_class'],
                        'confidence': result['confidence'],
                        'predicted_index': result['predicted_index']
                    }
                    
                    # Add class probabilities if requested
                    if include_probabilities and 'class_probabilities' in result:
                        for class_name, prob in result['class_probabilities'].items():
                            row[f'prob_{class_name}'] = prob
                
                csv_data.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(csv_data)
            df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
            
        except Exception as e:
            print(f"Error saving CSV: {e}")
    
    def visualize_predictions(self, results, num_images=9, figsize=(15, 15), 
                            confidence_threshold=0.5):
        """
        Visualize prediction results for a subset of images
        
        Args:
            results (list): List of prediction results
            num_images (int): Number of images to visualize
            figsize (tuple): Figure size
            confidence_threshold (float): Minimum confidence to highlight
        """
        # Filter out error results
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            print("No valid results to visualize")
            return
        
        # Select images to show
        num_to_show = min(num_images, len(valid_results))
        selected_results = valid_results[:num_to_show]
        
        # Calculate grid dimensions
        cols = int(np.ceil(np.sqrt(num_to_show)))
        rows = int(np.ceil(num_to_show / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        elif cols == 1:
            axes = [[ax] for ax in axes]
        else:
            axes = axes.reshape(rows, cols) if num_to_show > 1 else [[axes]]
        
        for i, result in enumerate(selected_results):
            row, col = i // cols, i % cols
            ax = axes[row][col] if rows > 1 else axes[col]
            
            try:
                # Load and display image
                image = Image.open(result['image_path'])
                ax.imshow(image)
                
                # Create title with prediction info
                confidence = result['confidence']
                predicted_class = result['predicted_class']
                
                # Color based on confidence
                color = 'green' if confidence >= confidence_threshold else 'red'
                title = f"{predicted_class}\nConf: {confidence:.3f}"
                
                ax.set_title(title, color=color, fontsize=10, fontweight='bold')
                ax.axis('off')
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading\n{Path(result['image_path']).name}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
        
        # Hide empty subplots
        for i in range(num_to_show, rows * cols):
            row, col = i // cols, i % cols
            ax = axes[row][col] if rows > 1 else axes[col]
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def generate_summary_report(self, results, save_path=None):
        """
        Generate a summary report of predictions
        
        Args:
            results (list): List of prediction results
            save_path (str): Path to save the report (optional)
        """
        # Filter valid results
        valid_results = [r for r in results if 'error' not in r]
        error_results = [r for r in results if 'error' in r]
        
        print("=" * 50)
        print("PREDICTION SUMMARY REPORT")
        print("=" * 50)
        
        print(f"Total images processed: {len(results)}")
        print(f"Successfully processed: {len(valid_results)}")
        print(f"Errors encountered: {len(error_results)}")
        
        if valid_results:
            # Class distribution
            class_counts = {}
            confidences = []
            
            for result in valid_results:
                predicted_class = result['predicted_class']
                confidence = result['confidence']
                
                class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
                confidences.append(confidence)
            
            print(f"\nClass Distribution:")
            print("-" * 30)
            for class_name, count in sorted(class_counts.items()):
                percentage = (count / len(valid_results)) * 100
                print(f"{class_name}: {count} ({percentage:.1f}%)")
            
            print(f"\nConfidence Statistics:")
            print("-" * 30)
            print(f"Mean confidence: {np.mean(confidences):.3f}")
            print(f"Std confidence: {np.std(confidences):.3f}")
            print(f"Min confidence: {np.min(confidences):.3f}")
            print(f"Max confidence: {np.max(confidences):.3f}")
            
            # Low confidence predictions
            low_conf_threshold = 0.7
            low_conf_results = [r for r in valid_results if r['confidence'] < low_conf_threshold]
            
            if low_conf_results:
                print(f"\nLow Confidence Predictions (< {low_conf_threshold}):")
                print("-" * 50)
                for result in low_conf_results[:10]:  # Show first 10
                    image_name = Path(result['image_path']).name
                    print(f"{image_name}: {result['predicted_class']} ({result['confidence']:.3f})")
                
                if len(low_conf_results) > 10:
                    print(f"... and {len(low_conf_results) - 10} more")
        
        if error_results:
            print(f"\nErrors:")
            print("-" * 20)
            for result in error_results:
                image_name = Path(result['image_path']).name
                print(f"{image_name}: {result['error']}")
        
        # Save report if requested
        if save_path:
            with open(save_path, 'w') as f:
                # Write summary to file (you can expand this)
                f.write("Prediction Summary Report\n")
                f.write("=" * 25 + "\n")
                f.write(f"Total images: {len(results)}\n")
                f.write(f"Successful: {len(valid_results)}\n")
                f.write(f"Errors: {len(error_results)}\n")
                # Add more details as needed
            print(f"\nReport saved to: {save_path}")



    
    