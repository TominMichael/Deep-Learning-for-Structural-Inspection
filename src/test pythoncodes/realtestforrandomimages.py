import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import segmentation_models_pytorch as smp
from torchvision import transforms
import os

class CrackDetectionTester:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the crack detection tester
        
        Args:
            model_path (str): Path to the trained model
            device (str): Device to run inference on
        """
        self.device = device
        self.model_path = model_path
        self.model = None
        self.input_size = (512, 512)
        
     
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.load_model()
    
    def load_model(self):
        """Load the trained UNet++ model"""
        try:
           
            self.model = smp.UnetPlusPlus(
                encoder_name="efficientnet-b7",
                encoder_weights=None,  
                in_channels=3,
                classes=1,  
                activation=None  
            )
            
          
            checkpoint = torch.load(self.model_path, map_location=self.device)
      
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            raise
    
    def preprocess_image(self, image_path):
        """
        Preprocess the input image
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            tuple: (preprocessed_tensor, original_image, resized_image)
        """
      
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
       
        pil_image = Image.fromarray(original_image)
     
        input_tensor = self.transform(pil_image).unsqueeze(0)  
     
        resized_image = cv2.resize(original_image, self.input_size)
        
        return input_tensor.to(self.device), original_image, resized_image
    
    def predict(self, input_tensor):
        """
        Make prediction on the input tensor
        
        Args:
            input_tensor: Preprocessed input tensor
            
        Returns:
            numpy.ndarray: Predicted mask
        """
        with torch.no_grad():
           
            output = self.model(input_tensor)
            
            
            output = torch.sigmoid(output)
            
      
            mask = output.cpu().squeeze().numpy()
            
        return mask
    
    def post_process_mask(self, mask, threshold=0.5):
        """
        Post-process the predicted mask
        
        Args:
            mask: Raw prediction mask
            threshold: Threshold for binary conversion
            
        Returns:
            numpy.ndarray: Binary mask
        """
        
        binary_mask = (mask > threshold).astype(np.uint8)
        
        
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        return binary_mask
    
    def find_crack_boundaries(self, binary_mask):
        """
        Find crack boundaries using contour detection
        
        Args:
            binary_mask: Binary segmentation mask
            
        Returns:
            list: List of contours representing crack boundaries
        """
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        
        min_area = 50  
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        return filtered_contours
    
    def visualize_results(self, original_image, resized_image, mask, binary_mask, contours, save_path=None):
        """
        Visualize the results
        
        Args:
            original_image: Original input image
            resized_image: Resized image (512x512)
            mask: Raw prediction mask
            binary_mask: Binary mask after post-processing
            contours: Detected crack boundaries
            save_path: Path to save the visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
      
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        
        axes[0, 1].imshow(resized_image)
        axes[0, 1].set_title('Resized Image (512x512)')
        axes[0, 1].axis('off')
        
        
        axes[0, 2].imshow(mask, cmap='hot')
        axes[0, 2].set_title('Raw Prediction')
        axes[0, 2].axis('off')
        
   
        axes[1, 0].imshow(binary_mask, cmap='gray')
        axes[1, 0].set_title('Binary Mask')
        axes[1, 0].axis('off')
        
        
        overlay = resized_image.copy()
        overlay[binary_mask > 0] = [255, 0, 0]  
        blended = cv2.addWeighted(resized_image, 0.7, overlay, 0.3, 0)
        axes[1, 1].imshow(blended)
        axes[1, 1].set_title('Crack Overlay')
        axes[1, 1].axis('off')
        
       
        boundary_image = resized_image.copy()
        cv2.drawContours(boundary_image, contours, -1, (0, 255, 0), 2)  
        axes[1, 2].imshow(boundary_image)
        axes[1, 2].set_title('Crack Boundaries')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def test_image(self, image_path, threshold=0.5, save_results=True):
        """
        Complete testing pipeline for a single image
        
        Args:
            image_path (str): Path to the test image
            threshold (float): Threshold for binary conversion
            save_results (bool): Whether to save results
        """
       
        if not os.path.exists(image_path):
            return
        
        
        input_tensor, original_image, resized_image = self.preprocess_image(image_path)
        
       
        mask = self.predict(input_tensor)
        
       
        binary_mask = self.post_process_mask(mask, threshold)
        
     
        contours = self.find_crack_boundaries(binary_mask)
        
        
        total_crack_pixels = np.sum(binary_mask)
        total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
        crack_percentage = (total_crack_pixels / total_pixels) * 100
       
        save_path = None
        if save_results:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = f"crack_detection_results_{base_name}.png"
        
        self.visualize_results(original_image, resized_image, mask, binary_mask, contours, save_path)
        
        return {
            'mask': mask,
            'binary_mask': binary_mask,
            'contours': contours,
            'crack_percentage': crack_percentage,
            'num_cracks': len(contours)
        }

def main():
  
    model_path = r""
    test_image_path = r""
    
   
    tester = CrackDetectionTester(model_path)
 
    results = tester.test_image(test_image_path, threshold=0.5)

if __name__ == "__main__":
    main()