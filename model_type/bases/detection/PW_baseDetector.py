from PytorchWildlife.models import MegaDetectorV5
from yolov5.utils.general import non_max_suppression, scale_coords
from tqdm import tqdm

class MegaD5(MegaDetectorV5):
    def batch_detections(self, 
                         dataloader, 
                         conf_thres=0.2, 
                         id_strip=None, 
                         log_queue = None):

        """
        Perform detection on a batch of images.
        
        Args:
            dataloader (DataLoader): 
                DataLoader containing image batches.
            conf_thres (float, optional): 
                Confidence threshold for predictions. Defaults to 0.2.
            id_strip (str, optional): 
                Characters to strip from img_id. Defaults to None.

        Returns:
            list: List of detection results for all images.
        """
        results = []
        with tqdm(total=len(dataloader)) as pbar:
            for batch_index, (imgs, paths, sizes) in enumerate(dataloader):
                imgs = imgs.to(self.device)
                predictions = self.model(imgs)[0].detach().cpu()
                predictions = non_max_suppression(predictions, conf_thres=conf_thres)

                batch_results = []
                for i, pred in enumerate(predictions):
                    if pred.size(0) == 0:  
                        continue
                    pred = pred.numpy()
                    size = sizes[i].numpy()
                    path = paths[i]
                    original_coords = pred[:, :4].copy()
                    pred[:, :4] = scale_coords([self.IMAGE_SIZE] * 2, pred[:, :4], size).round()
                    # Normalize the coordinates for timelapse compatibility
                    normalized_coords = [[x1 / size[1], y1 / size[0], x2 / size[1], y2 / size[0]] for x1, y1, x2, y2 in pred[:, :4]]
                    res = self.results_generation(pred, path, id_strip)
                    res["normalized_coords"] = normalized_coords
                    batch_results.append(res)
                pbar.update(1)

                if log_queue:
                    progress = round((pbar.n / pbar.total) * 100, 2)
                    log_queue.put(f"Detection progress: {progress}%")


                results.extend(batch_results)
            return results

        return super().batch_image_detection(dataloader, conf_thres, id_strip)