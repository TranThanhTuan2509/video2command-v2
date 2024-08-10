def mask_model(image_path):
    from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    import torch
    import cv2
    import numpy as np

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    original_image = torch.from_numpy(image)
    image = np.transpose(image, (2, 0, 1)) / 255.
    image = torch.from_numpy(image).float()
    image = image.unsqueeze(0).float()
    # Load the model
    mask_rcnn = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    mask_rcnn.eval()
    with torch.no_grad():
        output = mask_rcnn(image)

    outputs = [{
        'boxes': [],
        'labels': [],
        'scores': [],
        'masks': []
    }]

    def iou_score(output):
        for idx, (box, score) in enumerate(zip(output[0]['boxes'], output[0]['scores'])):
            topx, topy, bottomx, bottomy = box
            width_o = int(bottomx) - int(topx)
            height_o = int(bottomy) - int(topy)
            area_o = width_o * height_o
            iou_score = area_o / (224 * 224)
            if iou_score < 0.6 and score >= 0.4:
                outputs[0]['boxes'].append(output[0]['boxes'][idx])
                outputs[0]['labels'].append(output[0]['labels'][idx])
                outputs[0]['scores'].append(output[0]['scores'][idx])
                outputs[0]['masks'].append(output[0]['masks'][idx])

                # Convert lists back to tensors
        outputs[0]['boxes'] = torch.stack(outputs[0]['boxes']) if outputs[0]['boxes'] else torch.tensor([])
        outputs[0]['labels'] = torch.stack(outputs[0]['labels']) if outputs[0]['labels'] else torch.tensor([])
        outputs[0]['scores'] = torch.stack(outputs[0]['scores']) if outputs[0]['scores'] else torch.tensor([])
        outputs[0]['masks'] = torch.stack(outputs[0]['masks']) if outputs[0]['masks'] else torch.tensor([])
        return outputs

    results = iou_score(output)
    print(results[0]['labels'])
    masks = results[0]['masks'] > 0.5
    copy = torch.zeros_like(original_image)
    for i in range(masks.shape[0]):
        mask = masks[i, 0].mul(255).byte().cpu()
        # Create a colored overlay for the mask
        colored_mask = torch.zeros_like(original_image)
        colored_mask[mask > 127] = 255
        colored_mask[mask < 127] = 0
        copy += colored_mask
    combined_image = torch.mul(copy, original_image)
    combined_image = torch.permute(combined_image, (2, 0, 1)).float()
    return combined_image
