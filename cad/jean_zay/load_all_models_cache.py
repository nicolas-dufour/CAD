import open_clip
from transformers import (
    AutoTokenizer,
    DeiTForImageClassificationWithTeacher,
    T5EncoderModel,
    ViTForImageClassification,
)

from ..metrics.inception_metrics import NoTrainInceptionV3

AutoTokenizer.from_pretrained("google/flan-t5-xl")
T5EncoderModel.from_pretrained("google/flan-t5-xl")

clip_model, _, clip_processor = open_clip.create_model_and_transforms(
    "ViT-H-14-quickgelu", pretrained="metaclip_fullcc"
)


ViTForImageClassification.from_pretrained("nateraw/vit-base-patch16-224-cifar10")

DeiTForImageClassificationWithTeacher.from_pretrained(
    "facebook/deit-base-distilled-patch16-384"
)

NoTrainInceptionV3(
    name="inception-v3-compat",
    features_list=[str(2048), "logits_unbiased"],
)
