from torchvision.io import read_image
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

images_dir = "/users/bspiegel/data/bspiegel/clevr-refplus-dcplus-dataset-gen/output/images/valA/"
image_name = "CLEVR_valA_000000.png"

img = read_image(images_dir+image_name)

images = [img]
images = [(i[:3, :, :]).float()/255 for i in images]

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
transforms = weights.transforms()
images = [transforms(d) for d in images]
model = fasterrcnn_resnet50_fpn(weights=weights, progress=False)
model = model.eval()

outputs = model(images)
print(outputs)