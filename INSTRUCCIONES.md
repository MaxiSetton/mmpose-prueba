El repositorio usa python 3.9.13

para instalar las dependencias, ejecutar:

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
pip install openpose
```
Después, hay que compilarlo con c++, en mac es:

```bash
xcode-select --install
# Desinstalar la versión incorrecta
pip uninstall mmcv -y

# Limpiar todo el caché de pip
pip cache purge
MMCV_WITH_OPS=1 pip install mmcv==2.1.0 --no-cache-dir
```

Verificación:

```bash
python -c "from mmcv.ops import RoIPool; print('✅ ¡MISIÓN CUMPLIDA! MMCV compilado y funcionando.')"
```
Una vez que esté todo instalado, hay que:
1. Descargar el .pth (checkpoint) desde el model zoo de openpose
2. Mover main, el modelo.pth a la carpeta 'mmpose'
3. Ejecutar el main.py desde la carpeta mmpose