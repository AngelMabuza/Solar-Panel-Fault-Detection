"""
Standalone segmentation prediction script.
Use this to predict segmentation masks for any input image.

Example:
    python predict_segmentation.py --image path/to/image.jpg --model reports/seg_unet/best.keras --output pred_mask.png
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Import metrics if needed for model loading
try:
    from src.utils.metrics_seg import dice_coef, iou
except ImportError:
    dice_coef = None
    iou = None


def load_image_for_prediction(img_path, img_size=128):
    """
    Load and preprocess an image for segmentation prediction.
    
    Args:
        img_path (str): Path to the input image file (BMP, PNG, JPG supported).
        img_size (int): Target image size (default 128).
    
    Returns:
        np.ndarray: Normalized image array of shape (img_size, img_size, 3) in range [0, 1].
    """
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32) / 255.0
    return img.numpy()


def predict_segmentation(image_array, model, threshold=0.5):
    """
    Predict segmentation mask for a single image.
    
    Args:
        image_array (np.ndarray): Input image of shape (H, W, 3) normalized to [0, 1].
        model: Loaded Keras model.
        threshold (float): Binarization threshold for the predicted probability map (default 0.5).
    
    Returns:
        tuple: (prob_map, binary_mask)
            - prob_map (np.ndarray): Predicted probability map of shape (H, W) in range [0, 1].
            - binary_mask (np.ndarray): Binarized mask of shape (H, W) with values 0 or 1.
    """
    # Add batch dimension
    batch_img = tf.expand_dims(image_array, 0)
    
    # Forward pass
    pred = model(batch_img, training=False).numpy()[0, :, :, 0]
    
    # Binarize
    binary_mask = (pred >= threshold).astype(np.float32)
    
    return pred, binary_mask


def save_mask_as_image(mask, output_path, colormap="gray"):
    """
    Save a segmentation mask as an image file.
    
    Args:
        mask (np.ndarray): Mask of shape (H, W) with values in [0, 1].
        output_path (str): Path where the mask will be saved.
        colormap (str): Colormap to use ('gray' for grayscale, 'viridis' for color).
    """
    # Convert to 0-255 range for PNG
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    if colormap == "gray":
        # Grayscale
        img = Image.fromarray(mask_uint8, mode="L")
    else:
        # Apply colormap and save as RGB
        import matplotlib.cm as cm
        cmap = cm.get_cmap(colormap)
        mask_colored = cmap(mask)[:, :, :3]  # Drop alpha channel
        mask_uint8_rgb = (mask_colored * 255).astype(np.uint8)
        img = Image.fromarray(mask_uint8_rgb, mode="RGB")
    
    img.save(output_path)
    print(f"Saved segmentation mask to: {output_path}")


def visualize_prediction(image_array, prob_map, binary_mask, output_path=None):
    """
    Visualize the input image, probability map, and binary segmentation mask side-by-side.
    
    Args:
        image_array (np.ndarray): Input image of shape (H, W, 3).
        prob_map (np.ndarray): Predicted probability map of shape (H, W).
        binary_mask (np.ndarray): Binarized mask of shape (H, W).
        output_path (str, optional): Path to save the visualization. If None, displays with plt.show().
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Input image
    axes[0].imshow(image_array)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    
    # Probability map
    im = axes[1].imshow(prob_map, cmap="viridis")
    axes[1].set_title("Predicted Probability")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Binary mask
    axes[2].imshow(binary_mask, cmap="gray")
    axes[2].set_title("Binary Segmentation")
    axes[2].axis("off")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Predict segmentation mask for any input image."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image (BMP, PNG, JPG).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the trained segmentation model (.keras file).",
    )
    parser.add_argument(
        "--output",
        default="predicted_mask.png",
        help="Output path for the segmentation mask (default: predicted_mask.png).",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=128,
        help="Image size for model input (default: 128).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binarization threshold (default: 0.5).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save a side-by-side visualization of input, probability, and mask.",
    )
    parser.add_argument(
        "--vis_output",
        default="segmentation_visualization.png",
        help="Output path for the visualization (default: segmentation_visualization.png).",
    )
    parser.add_argument(
        "--colormap",
        choices=["gray", "viridis", "hot", "plasma"],
        default="gray",
        help="Colormap for the mask image (default: gray).",
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    print(f"Loading model from: {args.model}")
    # Load model with custom objects if they exist
    custom_objs = {}
    if dice_coef is not None:
        custom_objs["dice_coef"] = dice_coef
    if iou is not None:
        custom_objs["iou"] = iou
    
    model = tf.keras.models.load_model(args.model, custom_objects=custom_objs if custom_objs else None)
    
    print(f"Loading image from: {args.image}")
    img_array = load_image_for_prediction(args.image, img_size=args.img_size)
    
    print("Running segmentation prediction...")
    prob_map, binary_mask = predict_segmentation(img_array, model, threshold=args.threshold)
    
    print(f"Saving segmentation mask to: {args.output}")
    save_mask_as_image(binary_mask, args.output, colormap=args.colormap)
    
    if args.visualize:
        print(f"Creating visualization...")
        visualize_prediction(img_array, prob_map, binary_mask, output_path=args.vis_output)
    
    print("Done!")


if __name__ == "__main__":
    main()
