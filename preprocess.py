import torch
device = torch.device('cuda')
torch.cuda.is_available()
print("Nombre GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No detectada")
from kp_api import KeypointExtractor
kp_ckpt_path='rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth'
kp_cfg_path='configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py'
kp_extractor=KeypointExtractor(kp_cfg_path, kp_ckpt_path, device=device)
import os
import tarfile
import numpy as np
from pathlib import Path
import time
import subprocess
remote_files_processed=[]
PROGRESS_FILE = Path("processed_files.txt")

rclone_remote_root = "drive:MyDrive/Kinetics400"
# Directorio fuente (donde están los .tar.gz)
root_targz_local = Path("k400_targz_temp")

# Directorio de destino para los VIDEOS (Requisito 1)
root_videos = Path("k400_videos")

# Directorio de destino para los KEYPOINTS (Requisito 2)
root_keypoints_out = Path("k400_keypoints")

# Subcarpetas a procesar (igual que en tu script de Bash)
sub_folders = ['train', 'val', 'test', 'replacement']

import os
import subprocess

# --- Configuración ---
# Nombre del archivo que estás buscando
FILE_NAME = "rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"

# Nombre del control remoto de rclone para Google Drive (usualmente 'gdrive' o 'MyDrive')
# Se asume 'MyDrive' según la ruta que indicaste.
RCLONE_REMOTE = "MyDrive"

# Ruta completa del archivo en el control remoto de rclone.
# Formato: <Remote_Name>:<Path_in_Drive>/<File_Name>
# Asumimos que el archivo está en la raíz del Drive o en una carpeta llamada 'MyDrive'
# dentro de tu configuración de rclone.
RCLONE_SOURCE_PATH = f"{RCLONE_REMOTE}:/{FILE_NAME}"

# Directorio local donde se guardará el archivo ('.' significa el directorio actual)
LOCAL_DESTINATION = "."

def check_and_download_file():
    """
    Verifica si el archivo existe localmente. Si no existe, lo descarga usando rclone.
    """
    local_file_path = os.path.join(LOCAL_DESTINATION, FILE_NAME)

    if os.path.exists(local_file_path):
        print(f"✅ El archivo ya existe en: {local_file_path}")
        return

    print(f"❌ El archivo no se encontró localmente. Intentando descargar desde Drive...")
    print(f"Fuente en Drive: {RCLONE_SOURCE_PATH}")

    # Comando rclone a ejecutar
    # 'rclone copy' copia el archivo de la fuente al destino.
    rclone_command = [
        "rclone",
        "copy",
        RCLONE_SOURCE_PATH,
        LOCAL_DESTINATION
    ]

    try:
        # Ejecutar el comando rclone
        result = subprocess.run(
            rclone_command,
            check=True, # Lanza una excepción si el comando falla
            capture_output=True,
            text=True
        )

        if os.path.exists(local_file_path):
            print(f"\n✅ Descarga completada exitosamente. Archivo guardado en: {local_file_path}")
            # print("Detalles de rclone:", result.stdout)
        else:
            print("\n❌ Error: rclone finalizó, pero el archivo no apareció en el destino.")
            print("Verifica si la ruta en Drive es correcta y si rclone está configurado.")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error al ejecutar rclone (código de salida {e.returncode}):")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        print("\nPosibles problemas: rclone no está instalado, la configuración 'MyDrive' es incorrecta, o no tienes permisos.")
    except FileNotFoundError:
        print("\n❌ Error: El comando 'rclone' no se encontró. Asegúrate de que rclone esté instalado y en tu PATH.")

if __name__ == "__main__":
    check_and_download_file()

# Extensiones de video a buscar después de extraer
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.webm', '.mov'}
def load_processed_files():
    """Carga la lista de archivos ya procesados desde disco."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def save_processed_file(file_path: str):
    """Agrega un archivo procesado al registro."""
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(file_path + "\n")
def get_remote_file_list(remote_path: str) -> list:
    """
    Usa 'rclone lsf' para listar archivos .tar.gz y .tgz en un directorio remoto.
    """
    print(f"📡 Obteniendo lista de archivos remotos de: {remote_path}")
    command = ["rclone", "lsf", "--files-only", remote_path]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        all_files = result.stdout.splitlines()
        # Filtrar solo los tarballs
        tar_files = [f for f in all_files if f.endswith('.tar.gz') or f.endswith('.tgz')]
        print(f"  -> Encontrados {len(tar_files)} archivos tarball.")
        return tar_files
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR al listar archivos con rclone: {e.stderr}")
        return []
    except FileNotFoundError:
        print("❌ ERROR: 'rclone' no se encuentra. ¿Está instalado y en tu PATH?")
        return []
def download_file(remote_file_path: str, local_file_path: Path):
    """
    Descarga un SOLO archivo usando 'rclone copyto'.
    """

    # Asegurarse de que el directorio local exista
    local_file_path.parent.mkdir(parents=True, exist_ok=True) 
    
    command = ["rclone", "copyto", 
        "--multi-thread-streams=4", # Usa 4 hilos POR CADA archivo
        # --------------------------
        remote_file_path, str(local_file_path)]
    
    try:
        # Usamos capture_output para no llenar la consola con el log de rclone
        subprocess.run(command, check=True, capture_output=True)
        print("  -> Descarga completa.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"    ❌ ERROR al descargar {remote_file_path}: {e.stderr}")
        return False

def process_single_tarball(local_tar_path: Path, videos_dir: Path, keypoints_dir: Path):
    """
    Procesa UN solo archivo .tar.gz:
    1. Extrae videos.
    2. Procesa videos -> .npy.
    3. BORRA videos.
    """
    print(f"  🗜️ Extrayendo: {local_tar_path.name} a {videos_dir}")
    try:
        with tarfile.open(local_tar_path, "r:gz") as tar:
            tar.extractall(path=videos_dir)
    except Exception as e:
        print(f"    ⚠️ AVISO: Error al extraer {local_tar_path.name}: {e}. Omitiendo.")
        return

    # --- Procesar Videos Extraídos ---
    print(f"  🤖 Procesando videos extraídos...")
    extracted_videos = []
    for ext in VIDEO_EXTENSIONS:
        # Busca solo en la carpeta de videos, no recursivo
        extracted_videos.extend(videos_dir.glob(f'*{ext}'))
    
    if not extracted_videos:
        print("    -> No se encontraron videos en este tarball.")
        return

    for video_path in extracted_videos:
        npy_filename = video_path.with_suffix('.npy').name
        npy_path = keypoints_dir / npy_filename

        if npy_path.exists():
            # print(f"    -> Ya existe, omitiendo: {npy_path.name}")
            pass # Omitir en silencio
        else:
            try:
                # 2. PROCESAR
                keypoints_array = kp_extractor.extract_from_video_parallel(video_path,6)
                np.save(npy_path, keypoints_array)
            except Exception as e:
                print(f"    ERROR al procesar el video {video_path.name}: {e}")
        
        # 3. BORRAR VIDEO (¡Importante para ahorrar espacio!)
        try:
            os.remove(video_path)
            print(video_path, " extraído")
        except Exception as e:
            print(f"      AVISO: No se pudo borrar el video {video_path.name}: {e}")
    
    print(f"  ✅ Videos procesados y borrados.")

if __name__ == "__main__":
    print("Iniciando script de procesamiento sostenible (uno por uno)...")
    start_time = time.time()
    processed_files = load_processed_files()

    for folder in sub_folders:
        print(f"\n{'='*20} \n📂 Carpeta Remota: {folder} \n{'='*20}")
        remote_folder_path = f"{rclone_remote_root}/{folder}"
        local_targz_dir = root_targz_local / folder
        local_videos_dir = root_videos / folder
        local_keypoints_dir = root_keypoints_out / folder
        
        local_targz_dir.mkdir(parents=True, exist_ok=True)
        local_videos_dir.mkdir(parents=True, exist_ok=True)
        local_keypoints_dir.mkdir(parents=True, exist_ok=True)

        remote_files = get_remote_file_list(remote_folder_path)
        if not remote_files:
            print(f"  No se encontraron archivos en {remote_folder_path}. Omitiendo.")
            continue

        total_files = len(remote_files)
        print(f"  Se procesarán {total_files} archivos...")

        for i, filename in enumerate(remote_files, 1):
            remote_file_path = f"{remote_folder_path}/{filename}"

            # 🔹 SALTAR si ya se procesó antes
            if remote_file_path in processed_files:
                print(f"  ⚡ Ya procesado: {filename}, omitiendo.")
                continue

            print(f"\n  --- Procesando archivo {i}/{total_files}: {filename} ---")
            local_tar_path = local_targz_dir / filename
            
            # ### INICIO DE LA MEJORA ###
            # Si el .tar.gz ya existe localmente, no lo volvemos a descargar.
            if local_tar_path.exists():
                print(f"  👍 .tar.gz ya existe localmente: {filename}. Omitiendo descarga.")
            else:
                if not download_file(remote_file_path, local_tar_path):
                    print("  Error en descarga. Omitiendo este archivo.")
                    continue
            # ### FIN DE LA MEJORA ###
            process_single_tarball(local_tar_path, local_videos_dir, local_keypoints_dir)
            
            try:
                os.remove(local_tar_path)
                print(f"  🗑️ .tar.gz temporal borrado: {local_tar_path.name}")
            except Exception as e:
                print(f"    AVISO: No se pudo borrar {local_tar_path.name}: {e}")
            
            # ✅ Registrar archivo como procesado
            save_processed_file(remote_file_path)
            processed_files.add(remote_file_path)

    end_time = time.time()
    print(f"\n{'='*20}\n🎉 ¡Proceso completo! \n{'='*20}")
    print(f"Directorio de Keypoints: {root_keypoints_out.resolve()}")
    print(f"Tiempo total: {end_time - start_time:.2f} segundos.")
