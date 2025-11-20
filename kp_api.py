# kp_api.py

import cv2
import numpy as np
import torch
from mmpose.apis import inference_topdown, init_model
from tqdm import tqdm
import torch
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
import threading
from queue import Queue

# V-- THESE ARE THE CRUCIAL LINES TO ADD --V
# This explicitly allows PyTorch to load checkpoints containing NumPy arrays.
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])


class KeypointExtractor:
    def __init__(self, config, checkpoint, device='cpu'):
        """
        Initializes the MMPose model.
        Args:
            config (str): Path to the model config file.
            checkpoint (str): Path to the model checkpoint file.
            device (str): Device to run inference on ('cpu' or 'cuda').
        """
        self.model = init_model(config, checkpoint, device=device)
        print("Keypoint extractor model loaded successfully.")
        self.device=device
        cfg = self.model.cfg
        self.device = next(self.model.parameters()).device
        self.scope = self.model.cfg.get('default_scope', 'mmpose')
        init_default_scope(self.scope)
        
        # 2. Cargar el pipeline de pre-procesamiento (solo se hace una vez)
        # Esto (Compose) es lo que hace el resize, normalize, ToTensor, etc.
        self.pipeline = Compose(self.model.cfg.test_dataloader.dataset.pipeline)
        
        # 3. Guardar la función "collate" que junta las muestras en un lote
        self.collate_fn = pseudo_collate
        
        # 4. Guardar los metadatos del dataset (los necesita el pipeline)
        self.dataset_meta = self.model.dataset_meta
        
        # 5. Asumir N° de keypoints de tu código original (para rellenar fallos)
        self.num_keypoints = 133
    def _process_batch(self, frame_batch, 
                         fw: int, fh: int, 
                         last_valid_keypoint: np.ndarray):
        """
        Método auxiliar que procesa un lote de fotogramas.
        Replica la lógica de `inference_topdown` pero para un lote.
        """
        
        data_list = []
        # Tu BBox constante para todos los fotogramas
        bbox = np.array([[0, 0, fw, fh]], dtype=np.float32) # Shape (1, 4)
        bbox_score = np.ones(1, dtype=np.float32)           # Shape (1,)
        
        # --- 1. Pre-procesamiento (CPU) ---
        # Aplicar el pipeline de transforms a cada imagen del lote.
        # Esto es rápido (cortar, redimensionar, etc.)
        for frame_img in frame_batch:
            data_info = dict(img=frame_img)
            data_info['bbox'] = bbox
            data_info['bbox_score'] = bbox_score
            data_info.update(self.dataset_meta)
            
            # Aplicar transforms (resize, normalize, ToTensor, etc.)
            processed_data = self.pipeline(data_info)
            data_list.append(processed_data)

        # --- 2. Collate (Mover a GPU) ---
        # 'pseudo_collate' agrupa la lista de dicts en un solo dict de lote
        # y (generalmente) mueve los tensores 'inputs' al dispositivo correcto.
        if not data_list:
            return [], last_valid_keypoint
            
        batch_data = self.collate_fn(data_list)
        
        # Asegurarnos de que el input está en la GPU (a veces collate no lo hace)
        if 'inputs' in batch_data and isinstance(batch_data['inputs'], torch.Tensor):
             batch_data['inputs'] = batch_data['inputs'].to(self.device)

        # --- 3. Inferencia (GPU) ---
        # Aquí es donde se usa la GPU al 100%
        with torch.no_grad():
            # Llamamos a model.test_step(), tal como hace inference_topdown
            results_list = self.model.test_step(batch_data)
        
        # --- 4. Post-procesamiento (CPU) ---
        # Replicamos tu lógica de "parseo" de resultados
        processed_keypoints = []
        for result_sample in results_list: # results_list es List[PoseDataSample]
            
            # Tu lógica original de chequeo
            if not result_sample or not result_sample.pred_instances.keypoints.any():
                # Si falla, usamos el último válido conocido
                processed_keypoints.append(last_valid_keypoint)
            else:
                # Extraer los datos
                keypoints_xy = result_sample.pred_instances.keypoints[0]
                scores = result_sample.pred_instances.keypoint_scores[0]
                
                # Tu Hstack original
                kp_with_score = np.hstack([keypoints_xy, scores[:, None]])
                
                processed_keypoints.append(kp_with_score)
                last_valid_keypoint = kp_with_score # Actualizar el último válido
        
        return processed_keypoints, last_valid_keypoint
    def extract_from_video_parallel(self, video_path: str, batch_size: int = 32, queue_size: int = 64):
        """
        Extrae keypoints usando un hilo productor (CPU) y un consumidor (GPU)
        para máxima velocidad.
        """
        
        # --- 1. Setup del Video y la TQDM ---
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Error: no se puede abrir {video_path}")
                return None
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception as e:
            print(f"Error inicializando captura de video: {e}")
            return None

        pbar = tqdm(total=total_frames, desc="Processing (Parallel)")
        
        # --- 2. Setup de la Cola y el Hilo Productor ---
        # Esta cola almacenará datos YA PRE-PROCESADOS por la CPU
        data_queue = Queue(maxsize=queue_size)
        
        # Objeto para señalizar el fin
        _stop_event = threading.Event()

        # Iniciar el hilo productor
        producer_thread = threading.Thread(
            target=self._producer_thread,
            args=(cap, frame_width, frame_height, data_queue, _stop_event)
        )
        producer_thread.start()

        # --- 3. Bucle Consumidor (Hilo Principal + GPU) ---
        all_keypoints = []
        batch_buffer = [] # Buffer para armar lotes para la GPU
        last_valid_keypoint = np.zeros((self.num_keypoints, 3), dtype=np.float32)

        try:
            while True:
                # Sacar un frame PRE-PROCESADO de la cola
                # El 'get' bloqueará el hilo si la cola está vacía,
                # permitiendo que el productor se ponga al día.
                processed_data = data_queue.get(timeout=30) # Timeout de 30s

                if processed_data is None: # El productor señaló el fin
                    break
                
                batch_buffer.append(processed_data)

                # Si el búfer está lleno, procesar el lote en la GPU
                if len(batch_buffer) == batch_size:
                    batch_kps, last_valid_keypoint = self._consumer_process_batch(
                        batch_buffer, last_valid_keypoint
                    )
                    all_keypoints.extend(batch_kps)
                    pbar.update(batch_size)
                    batch_buffer.clear() # Limpiar el búfer del lote

            # Procesar los fotogramas restantes al final
            if batch_buffer:
                batch_kps, _ = self._consumer_process_batch(
                    batch_buffer, last_valid_keypoint
                )
                all_keypoints.extend(batch_kps)
                pbar.update(len(batch_buffer))

        except Exception as e:
            print(f"Error fatal durante el consumo de GPU: {e}")
            _stop_event.set() # Señalizar al productor que pare
        
        finally:
            # Asegurarse de que todo se cierra limpiamente
            pbar.close()
            cap.release()
            producer_thread.join(timeout=5) # Esperar a que el hilo productor termine
            
        return np.array(all_keypoints, dtype=np.float32)


    def _producer_thread(self, cap, fw, fh, data_queue, stop_event):
        """
        Hilo Productor (100% CPU): Lee frames, los pre-procesa y los encola.
        """
        # BBox constante
        bbox = np.array([[0, 0, fw, fh]], dtype=np.float32)
        bbox_score = np.ones(1, dtype=np.float32)
        
        try:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break # Fin del video

                # --- El trabajo pesado de la CPU está aquí ---
                data_info = dict(img=frame)
                data_info['bbox'] = bbox
                data_info['bbox_score'] = bbox_score
                data_info.update(self.dataset_meta)
                
                # Aplicar transforms (resize, normalize, ToTensor, etc.)
                processed_data = self.pipeline(data_info)
                # --- Fin del trabajo pesado ---
                
                # Poner el frame pre-procesado en la cola.
                # Si la cola está llena (maxsize), esto bloqueará
                # al productor, evitando que consuma toda la RAM.
                data_queue.put(processed_data, timeout=30)
        
        except Exception as e:
            if not stop_event.is_set():
                print(f"Error en el hilo productor (CPU): {e}")

        finally:
            # Poner 'None' en la cola para señalizar al consumidor
            # que hemos terminado de leer el video.
            data_queue.put(None)


    def _consumer_process_batch(self, batch_data_list, last_valid_keypoint):
        """
        Trabajo del Consumidor (GPU): Colapsa el lote, infiere y post-procesa.
        """
        
        # --- 1. Collate (Rápido) ---
        # batch_data_list ya contiene tensores pre-procesados.
        # 'pseudo_collate' solo los apila.
        batch_data = self.collate_fn(batch_data_list)
        
        if 'inputs' in batch_data and isinstance(batch_data['inputs'], torch.Tensor):
             batch_data['inputs'] = batch_data['inputs'].to(self.device, non_blocking=True)

        # --- 2. Inferencia (GPU) ---
        with torch.no_grad():
            results_list = self.model.test_step(batch_data)
        
        # --- 3. Post-procesamiento (CPU) ---
        processed_keypoints = []
        for result_sample in results_list:
            if not result_sample or not result_sample.pred_instances.keypoints.any():
                processed_keypoints.append(last_valid_keypoint)
            else:
                keypoints_xy = result_sample.pred_instances.keypoints[0]
                scores = result_sample.pred_instances.keypoint_scores[0]
                kp_with_score = np.hstack([keypoints_xy, scores[:, None]])
                processed_keypoints.append(kp_with_score)
                last_valid_keypoint = kp_with_score
        
        return processed_keypoints, last_valid_keypoint