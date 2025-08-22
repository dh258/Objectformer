import onnxruntime as ort
import cv2
import numpy as np
import albumentations as A
from typing import Union, Tuple, Optional, Dict, Any


class ObjectFormerONNXInference:
    """
    ObjectFormer ONNX inference wrapper for image tampering detection and localization.
    Optimized for CPU deployment using ONNX Runtime.
    """

    def __init__(
        self,
        onnx_model_path: str,
        input_size: int = 288,
        providers: Optional[list] = None,
    ):
        """
        Initialize ObjectFormer ONNX inference wrapper.

        Args:
            onnx_model_path: Path to ONNX model file
            input_size: Input image size (default: 288)
            providers: ONNX Runtime providers. If None, uses CPU provider.
        """
        self.input_size = input_size
        self.onnx_model_path = onnx_model_path

        # Set execution providers (CPU by default for deployment)
        if providers is None:
            providers = ["CPUExecutionProvider"]

        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)

        # Get model input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        print(f"Loaded ONNX model: {onnx_model_path}")
        print(f"Input shape: {self.session.get_inputs()[0].shape}")
        print(f"Output names: {self.output_names}")

        # Initialize preprocessing
        self.transform = self._get_transforms()

    def _get_transforms(self) -> A.Compose:
        """Get preprocessing transforms matching training configuration."""
        return A.Compose(
            [
                A.Resize(height=self.input_size, width=self.input_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def preprocess_image(self, image_input: Union[str, np.ndarray]) -> np.ndarray:
        """
        Preprocess input image for ONNX model.

        Args:
            image_input: Image file path or numpy array (RGB format)

        Returns:
            Preprocessed numpy array ready for ONNX model input
        """
        if isinstance(image_input, str):
            # Load image from path
            img = cv2.imread(image_input, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not load image from {image_input}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image_input.copy()

        # Apply transforms
        transformed = self.transform(image=img)
        img_array = transformed["image"]

        # Convert to float32 and add batch dimension
        img_array = img_array.astype(np.float32)

        # Rearrange from HWC to CHW and add batch dimension
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        return img_array

    def postprocess_outputs(
        self,
        onnx_outputs: list,
        threshold: float = 0.5,
        original_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Postprocess ONNX model outputs.

        Args:
            onnx_outputs: Raw outputs from ONNX model
            threshold: Threshold for binary classification and mask binarization
            original_size: Original image size (H, W) for mask resizing

        Returns:
            Dictionary containing processed results
        """
        # Assuming first output is detection logits (single scalar), rest are localization masks
        detection_logits = onnx_outputs[0]
        localization_masks = onnx_outputs[1:] if len(onnx_outputs) > 1 else []

        # Detection probability (apply sigmoid to scalar logit)
        # Model outputs single logit per sample, so we access the scalar value directly
        scalar_logit = (
            detection_logits.item()
            if detection_logits.shape == (1,)
            else detection_logits.squeeze().item()
        )
        detection_prob = 1.0 / (1.0 + np.exp(-scalar_logit))  # sigmoid
        is_tampered = detection_prob > threshold

        # Process localization mask (use the last/finest stage if available)
        binary_mask = None
        confidence_mask = None

        if localization_masks:
            # Get the last mask (finest resolution)
            mask = localization_masks[-1].squeeze()  # Remove batch and channel dims

            # Resize to original image size if provided
            if original_size is not None and mask.shape != original_size:
                mask = cv2.resize(
                    mask,
                    (original_size[1], original_size[0]),  # cv2 expects (W, H)
                    interpolation=cv2.INTER_LINEAR,
                )

            # Binarize mask
            binary_mask = (mask > threshold).astype(np.uint8)
            confidence_mask = mask.astype(np.float32)

        return {
            "is_tampered": bool(is_tampered),
            "detection_confidence": float(detection_prob),
            "binary_mask": binary_mask,
            "confidence_mask": confidence_mask,
            "raw_detection_logits": float(scalar_logit),
            "localization_masks": [mask.squeeze() for mask in localization_masks]
            if localization_masks
            else [],
        }

    def predict(
        self,
        image_input: Union[str, np.ndarray],
        threshold: float = 0.5,
        return_original_size: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform tampering detection and localization on input image.

        Args:
            image_input: Image file path or numpy array (RGB format)
            threshold: Threshold for classification and mask binarization
            return_original_size: Whether to resize masks to original image size

        Returns:
            Dictionary containing detection and localization results
        """
        # Store original size if needed
        original_size = None
        if return_original_size:
            if isinstance(image_input, str):
                img = cv2.imread(image_input, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Could not load image from {image_input}")
                original_size = (img.shape[0], img.shape[1])
            else:
                original_size = (image_input.shape[0], image_input.shape[1])

        # Preprocess
        img_array = self.preprocess_image(image_input)

        # Run inference
        onnx_outputs = self.session.run(self.output_names, {self.input_name: img_array})

        # Postprocess
        results = self.postprocess_outputs(
            onnx_outputs,
            threshold=threshold,
            original_size=original_size if return_original_size else None,
        )

        return results

    def predict_batch(self, image_batch: list, threshold: float = 0.5) -> list:
        """
        Perform batch prediction on multiple images.

        Args:
            image_batch: List of image paths or numpy arrays
            threshold: Threshold for classification and mask binarization

        Returns:
            List of prediction results
        """
        results = []
        for image_input in image_batch:
            try:
                result = self.predict(
                    image_input, threshold=threshold, return_original_size=False
                )
                results.append(result)
            except Exception as e:
                print(f"Error processing image: {e}")
                results.append(
                    {
                        "is_tampered": False,
                        "detection_confidence": 0.0,
                        "binary_mask": None,
                        "confidence_mask": None,
                        "error": str(e),
                    }
                )
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded ONNX model."""
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        return {
            "model_path": self.onnx_model_path,
            "inputs": [
                {"name": inp.name, "shape": inp.shape, "type": inp.type}
                for inp in inputs
            ],
            "outputs": [
                {"name": out.name, "shape": out.shape, "type": out.type}
                for out in outputs
            ],
            "providers": self.session.get_providers(),
        }
