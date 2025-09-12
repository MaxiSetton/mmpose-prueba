# import os
# import cv2
# from mmpose.apis import MMPoseInferencer
#
# # --- Configuración ---
# # Ruta del video de entrada
# video_path = 'video_prueba.mp4'
# # Directorio para guardar los fotogramas con los keypoints visualizados
# output_dir = 'output_frames/'
# # Alias del modelo que quieres usar
# # Para RTM-W-L usamos 'wholebody' y especificamos el modelo en los pesos
# pose2d_model = 'rtmw'
#
# # Crea el directorio de salida si no existe
# os.makedirs(output_dir, exist_ok=True)
#
# # --- Inicialización del Modelo ---
# # Crea una instancia del inferencer con el modelo rtmw-l
# # MMPose se encargará de descargar los pesos del modelo automáticamente
# print(f"Cargando el modelo {pose2d_model}...")
# config_file = 'projects/rtmpose/rtmw/rtmw-l_8xb32-270e_coco-wholebody-384x288.py'
# checkpoint_file = 'ruta/a/tus/pesos/rtmw-l_8xb32-270e_coco-wholebody-384x288.pth' # Debes descargar este archivo
#
# # Inicializa el inferencer
# inferencer = MMPoseInferencer(pose2d=config_file, pose2d_weights=checkpoint_file)
#
# print("Modelo cargado exitosamente.")
#
# # --- Procesamiento del Video ---
# print(f"Procesando el video: {video_path}")
#
# # Usa el inferencer para procesar el video.
# # El resultado es un generador que produce los resultados para cada fotograma.
# result_generator = inferencer(video_path, show=False, out_dir=output_dir, save_vis=True)
#
# # Iteramos sobre los resultados para procesar todo el video
# # 'save_vis=True' guarda automáticamente los fotogramas con las poses dibujadas
# # en la carpeta especificada en 'out_dir'.
# results = [result for result in result_generator]
#
# print(f"¡Procesamiento completado! Los fotogramas con los keypoints se han guardado en: {output_dir}")
#
# # --- (Opcional) Acceder a los Keypoints Numéricos ---
# # Si necesitas los datos de los keypoints (coordenadas) en lugar de solo la visualización,
# # puedes acceder a ellos de la siguiente manera.
#
# # 'results' es una lista donde cada elemento es un diccionario para un fotograma.
# # Cada diccionario contiene 'predictions' y 'visualization'.
#
# for frame_idx, frame_results in enumerate(results):
#     # 'predictions' es una lista de instancias de pose detectadas en el fotograma
#     predictions = frame_results['predictions']
#
#     print(f"\n--- Fotograma {frame_idx + 1} ---")
#
#     if not predictions or not predictions[0]:
#         print("No se detectaron personas en este fotograma.")
#         continue
#
#     # Iteramos sobre cada persona detectada en el fotograma
#     for person_idx, person_pred in enumerate(predictions[0]):
#         print(f"  Persona {person_idx + 1}:")
#
#         # Obtenemos los keypoints y su puntuación de confianza
#         keypoints = person_pred['keypoints']
#         keypoint_scores = person_pred['keypoint_scores']
#
#         # 'keypoints' es una lista de coordenadas [x, y] para cada punto clave
#         # 'keypoint_scores' es la confianza de la predicción para cada punto
#
#         # Ejemplo: Imprimir las coordenadas del keypoint de la nariz (índice 0 para COCO-WholeBody)
#         nariz_coords = keypoints[0]
#         nariz_score = keypoint_scores[0]
#         print(f"    - Coordenadas de la nariz: ({nariz_coords[0]:.2f}, {nariz_coords[1]:.2f}) con confianza: {nariz_score:.2f}")

import torch.serialization
import numpy
# <<< AÑADE ESTAS LÍNEAS AL INICIO DE TU SCRIPT
# Le decimos a PyTorch que confiamos en esta función específica de numpy que el checkpoint necesita.
torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
import cv2
from tqdm import tqdm
import mmcv
from mmcv.image import imread

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

# --- CAMBIOS PARA RTMW-L ---
# 1. Ruta a la configuración del modelo rtmw-l (whole-body)
# En lugar de la ruta al archivo .py
# model_cfg = 'projects/rtmpose/rtmw/rtmw-l_8xb32-270e_coco-wholebody-384x288.py'

# Usa el alias del modelo (el mismo nombre pero sin la extensión .py)
model_cfg = '/Users/defeee/Documents/GitHub/mmpose-prueba/mmpose/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py' # <<< CAMBIA ESTO

# El resto de tu código permanece igual
ckpt = '/Users/defeee/Documents/GitHub/mmpose-prueba/mmpose/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth'
device = 'mps'

# init model
model = init_model(model_cfg, ckpt, device=device)
# ...
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.set_dataset_meta(model.dataset_meta)

video_path='/Users/defeee/Documents/GitHub/mmpose-prueba/video_prueba.mp4'
output_path = '/Users/defeee/Documents/GitHub/mmpose-prueba/output.mp4'
# --- 3. PROCESAMIENTO DEL VIDEO ---
# Abrir el video de entrada
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: No se pudo abrir el video en la ruta: {video_path}")
    exit()

# Obtener propiedades del video para crear el archivo de salida
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Definir el codec y crear el objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Bucle principal para procesar cada frame
print(f"Procesando video... Total de frames: {total_frames}")
for _ in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break

    # --- 4. INFERENCIA POR FRAME ---
    # Los modelos Top-down como RTMW necesitan un "bounding box" para saber dónde buscar.
    # Para este ejemplo, asumimos que hay una persona ocupando todo el frame.
    # En un caso real, aquí usarías un detector de personas (ej. YOLO, RTMDet).
    person_bbox = [0, 0, frame_width, frame_height]

    # Realizar la inferencia en el frame actual
    results = inference_topdown(model, frame, bboxes=[person_bbox])
    data_samples = merge_data_samples(results)

    # --- 5. VISUALIZACIÓN Y ESCRITURA ---
    # Dibujar los keypoints en el frame
    # Usamos cv2.cvtColor para asegurar que el visualizador reciba el frame en RGB
    processed_frame_bgr = visualizer.add_datasample(
        'result',
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        data_sample=data_samples,
        draw_bbox=False, # No dibujamos el bbox de frame completo
        show=True # No mostramos la ventana en tiempo real
    )
    # # Convertimos de vuelta a BGR para guardarlo con OpenCV
    # processed_frame_bgr = cv2.cvtColor(processed_frame_bgr, cv2.COLOR_RGB2BGR)
    #
    #
    # # Escribir el frame procesado en el video de salida
    # video_writer.write(processed_frame_bgr)

# --- 6. FINALIZACIÓN ---
# Liberar los recursos
cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"¡Proceso completado! El video con los keypoints se ha guardado en: {output_path}")
