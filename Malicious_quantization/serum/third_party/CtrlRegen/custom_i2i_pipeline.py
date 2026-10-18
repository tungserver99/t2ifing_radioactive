from diffusers import StableDiffusionControlNetImg2ImgPipeline
from custom_ip_adapter import CustomIPAdapterMixin
from transformers import AutoModel, AutoImageProcessor


class CustomStableDiffusionControlNetImg2ImgPipeline(
    StableDiffusionControlNetImg2ImgPipeline,
    CustomIPAdapterMixin
):
    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        controlnet,
        scheduler,
        safety_checker,
        feature_extractor: AutoImageProcessor,  # Changed type
        image_encoder: AutoModel = None,  # Changed type
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,  # Updated
            image_encoder=image_encoder,  # Updated
            requires_safety_checker=requires_safety_checker,
        )