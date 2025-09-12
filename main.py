import os
import cv2
from mmpose.apis import MMPoseInferencer

# --- Configuración ---
# Ruta del video de entrada
video_path = 'video_prueba.mp4'
# Directorio para guardar los fotogramas con los keypoints visualizados
output_dir = 'output_frames/'
# Alias del modelo que quieres usar
# Para RTM-W-L usamos 'wholebody' y especificamos el modelo en los pesos
pose2d_model = 'rtmw-l'

# Crea el directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# --- Inicialización del Modelo ---
# Crea una instancia del inferencer con el modelo rtmw-l
# MMPose se encargará de descargar los pesos del modelo automáticamente
print(f"Cargando el modelo {pose2d_model}...")
inferencer = MMPoseInferencer(pose2d=pose2d_model)
print("Modelo cargado exitosamente.")

# --- Procesamiento del Video ---
print(f"Procesando el video: {video_path}")

# Usa el inferencer para procesar el video.
# El resultado es un generador que produce los resultados para cada fotograma.
result_generator = inferencer(video_path, show=False, out_dir=output_dir, save_vis=True)

# Iteramos sobre los resultados para procesar todo el video
# 'save_vis=True' guarda automáticamente los fotogramas con las poses dibujadas
# en la carpeta especificada en 'out_dir'.
results = [result for result in result_generator]

print(f"¡Procesamiento completado! Los fotogramas con los keypoints se han guardado en: {output_dir}")

# --- (Opcional) Acceder a los Keypoints Numéricos ---
# Si necesitas los datos de los keypoints (coordenadas) en lugar de solo la visualización,
# puedes acceder a ellos de la siguiente manera.

# 'results' es una lista donde cada elemento es un diccionario para un fotograma.
# Cada diccionario contiene 'predictions' y 'visualization'.

for frame_idx, frame_results in enumerate(results):
    # 'predictions' es una lista de instancias de pose detectadas en el fotograma
    predictions = frame_results['predictions']
    
    print(f"\n--- Fotograma {frame_idx + 1} ---")
    
    if not predictions or not predictions[0]:
        print("No se detectaron personas en este fotograma.")
        continue

    # Iteramos sobre cada persona detectada en el fotograma
    for person_idx, person_pred in enumerate(predictions[0]):
        print(f"  Persona {person_idx + 1}:")
        
        # Obtenemos los keypoints y su puntuación de confianza
        keypoints = person_pred['keypoints']
        keypoint_scores = person_pred['keypoint_scores']
        
        # 'keypoints' es una lista de coordenadas [x, y] para cada punto clave
        # 'keypoint_scores' es la confianza de la predicción para cada punto
        
        # Ejemplo: Imprimir las coordenadas del keypoint de la nariz (índice 0 para COCO-WholeBody)
        nariz_coords = keypoints[0]
        nariz_score = keypoint_scores[0]
        print(f"    - Coordenadas de la nariz: ({nariz_coords[0]:.2f}, {nariz_coords[1]:.2f}) con confianza: {nariz_score:.2f}")