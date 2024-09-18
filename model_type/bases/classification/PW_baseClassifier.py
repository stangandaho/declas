import torch
import numpy as np
from tqdm import tqdm
from queue import Queue
from PytorchWildlife.models.classification import AI4GAmazonRainforest, AI4GSnapshotSerengeti

class Amazon(AI4GAmazonRainforest):
    def batch_classification(self, 
                                   dataloader, 
                                   id_strip=None, 
                                   log_queue=None):
        """
        Process a batch of images for classification with an option to log messages.
        
        Args:
            dataloader (DataLoader): DataLoader for the batch of images.
            id_strip (str, optional): String to strip from the image IDs.
            log_queue (Queue, optional): Queue to send log messages for tracking progress.
        """
        # Call the original batch_image_classification from parent class
        #super().batch_image_classification(dataloader)

        total_logits = []
        total_paths = []

        # Use tqdm for progress bar with dataloader
        with tqdm(total=len(dataloader)) as pbar: 
            for batch in dataloader:
                imgs, paths = batch
                imgs = imgs.to(self.device)  # Send images to the device (GPU/CPU)
                
                # Forward pass (classification)
                total_logits.append(self.forward(imgs))
                total_paths.append(paths)

                # Update progress bar
                pbar.update(1)

                # If log_queue is provided, log the progress
                if log_queue:
                    progress = round((pbar.n / pbar.total) * 100, 2)
                    log_queue.put(f"Classification progress: {progress}%")

        # Concatenate the results
        total_logits = torch.cat(total_logits, dim=0).cpu()
        total_paths = np.concatenate(total_paths, axis=0)

        # Generate and return results
        return self.results_generation(total_logits, total_paths, id_strip=id_strip)




class Serengeti(AI4GSnapshotSerengeti):
    def batch_classification(self, 
                                   dataloader, 
                                   id_strip=None, 
                                   log_queue=None):
        """
        Process a batch of images for classification with an option to log messages.
        
        Args:
            dataloader (DataLoader): DataLoader for the batch of images.
            id_strip (str, optional): String to strip from the image IDs.
            log_queue (Queue, optional): Queue to send log messages for tracking progress.
        """
        # Call the original batch_image_classification from parent class
        #super().batch_image_classification(dataloader)

        total_logits = []
        total_paths = []

        # Use tqdm for progress bar with dataloader
        with tqdm(total=len(dataloader)) as pbar: 
            for batch in dataloader:
                imgs, paths = batch
                imgs = imgs.to(self.device)  # Send images to the device (GPU/CPU)
                
                # Forward pass (classification)
                total_logits.append(self.forward(imgs))
                total_paths.append(paths)

                # Update progress bar
                pbar.update(1)

                # If log_queue is provided, log the progress
                if log_queue:
                    progress = round((pbar.n / pbar.total) * 100, 2)
                    log_queue.put(f"Classification progress: {progress}%")

        # Concatenate the results
        total_logits = torch.cat(total_logits, dim=0).cpu()
        total_paths = np.concatenate(total_paths, axis=0)

        # Generate and return results
        return self.results_generation(total_logits, total_paths, id_strip=id_strip)
