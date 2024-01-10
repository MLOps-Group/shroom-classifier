import timm
from shroom_classifier.config import N_SUPER_CLASSES

model = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)

