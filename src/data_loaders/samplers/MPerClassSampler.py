import numpy as np
from torch.utils.data.sampler import Sampler
import torch

class MPerClassSampler(Sampler):
    """
    Выбирает m_per_class примеров для каждого класса в батче.
    Гарантирует, что в батче будут позитивные пары.
    """
    def __init__(self, labels, m_per_class, batch_size, length_scaling=1.0):
        """
        labels: список меток всего датасета
        m_per_class: сколько примеров одного класса брать (обычно 4)
        batch_size: размер батча
        length_scaling: множитель длины эпохи
        """
        # Конвертация в numpy, если пришел список или тензор
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        elif isinstance(labels, list):
            labels = np.array(labels)
            
        self.labels = labels
        self.m_per_class = m_per_class
        self.batch_size = batch_size
        
        # Уникальные классы
        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        
        # Индексы для каждого класса
        self.class_indices = {c: np.where(self.labels == c)[0] for c in self.classes}
        
        self.n_batches = int(len(self.labels) / batch_size * length_scaling)
        self.length = self.n_batches * batch_size

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            n_classes = self.batch_size // self.m_per_class
            
            # Если классов меньше, чем нужно для батча
            replace_classes = len(self.classes) < n_classes
            selected_classes = np.random.choice(self.classes, n_classes, replace=replace_classes)
            
            for c in selected_classes:
                indices = self.class_indices[c]
                # Если примеров у класса мало
                replace_indices = len(indices) < self.m_per_class
                
                selected_indices = np.random.choice(
                    indices, 
                    self.m_per_class, 
                    replace=replace_indices
                )
                batch.extend(selected_indices)
            
            # ВАЖНО: Возвращаем список индексов, а не итератор!
            yield batch 

    def __len__(self):
        return self.n_batches # Sampler len - это количество батчей