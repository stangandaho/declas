from ultralytics import YOLO

class Y8Classif:
    def __init__(self, weight_path, conf_tres = 0.2) -> None:
        self.weight_path = weight_path
        self.conf_tres = conf_tres


    def drop_below_conf(self, props, classes, threshold):
        final_list = []
        for idx, val in enumerate(props):
            if val >= threshold:
                final_list.append([classes[idx], props[idx]])
        return final_list


    def get_classif(self, image_path, conf, imgsz):
        classif_model  = YOLO(self.weight_path)
        preds = classif_model(image_path, conf=conf, imgsz = imgsz, verbos = False)

        # class_id = [int(idx) for idx in prediction[0].boxes.cls.tolist()]

        # class_name = [nm for _, nm in prediction[0].names.items()]
        # class_name = [class_name[idx] for idx in class_id]

        # probs = [round(pr, 2) for pr in prediction[0].boxes.conf.tolist()]


        xyxy = preds.boxes.xyxy.cpu().numpy()
        confidence = preds.boxes.conf.cpu().numpy()
        class_id = preds.boxes.cls.cpu().numpy().astype(int)

        results = {"img_id": str(img_id).strip(None)}
        results["detections"] = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )

        results["labels"] = [
            f"{self.CLASS_NAMES[class_id]} {confidence:0.2f}"  
            for _, _, confidence, class_id, _, _ in results["detections"] 
        ]
        
        return results

        return self.drop_below_conf(probs, class_name, self.conf_tres)
