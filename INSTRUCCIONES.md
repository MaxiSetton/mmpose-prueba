El repositorio usa python 3.9.13
Después, hay que compilar mmcv con c++, en mac es:

```bash
xcode-select --install
# Desinstalar la versión incorrecta
pip uninstall mmcv -y

# Limpiar todo el caché de pip
pip cache purge
MMCV_WITH_OPS=1 pip install mmcv==2.1.0 --no-cache-dir
```
Para Windows:
```bash
$env:MMCV_WITH_OPS="1"
pip install mmcv==2.1.0 --no-cache-dir
```
para instalar las dependencias, ejecutar:

```bash
pip install -U openmim
mim install mmengine
pip install openpose
```

Verificación:

```bash
python -c "from mmcv.ops import RoIPool; print('✅ ¡MISIÓN CUMPLIDA! MMCV compilado y funcionando.')"
```
Una vez que esté todo instalado, hay que:
1. clonar el repo de mmpose  
2. Descargar el .pth (checkpoint) desde el model zoo de openpose (rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth)
3. Mover kp_api, el preprocess.ipynb y rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth a la carpeta 'mmpose'
4. Instalar rclone
```bash
# Ejecutar como Administrador si winget requiere permisos
winget install rclone
# Verificar
rclone version
```
SI NO FUNCIONA
```bash
cd %TEMP%
curl -L -o rclone.zip https://downloads.rclone.org/rclone-current-windows-amd64.zip
tar -xf rclone.zip
rem suponiendo que la carpeta extraída es rclone-v1.xx-windows-amd64
mkdir %USERPROFILE%\bin
copy rclone-v*-windows-amd64\rclone.exe %USERPROFILE%\bin\
setx PATH "%PATH%;%USERPROFILE%\bin"
rclone version
```
- Luego cerrar y volver a abrir para ejecutar

4. Ejecutar el preprocess.ipynb desde la carpeta mmpose
SI TIRA ERROR XTCOCOTOOLS:
```bash
pip install numpy<2.0
```
5. ESTO NO EN PRINCIPIO: En el venv, cambiar la parte que carga el modelo para desactivar weights_only en el torch.load