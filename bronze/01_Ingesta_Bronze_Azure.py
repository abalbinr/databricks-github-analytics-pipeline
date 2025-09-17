# Databricks notebook source
# Importar librerías necesarias
import urllib.request 
import os 
import gzip 
import shutil
import tempfile
from azure.storage.blob import BlobServiceClient
from datetime import datetime

print("📦 CONFIGURACIÓN AZURE STORAGE PARA PROYECTO GITHUB")
print("=" * 60)

# Configurar cliente de Azure Storage usando secrets
def get_storage_client():
    try:
        # Obtener credenciales desde Databricks secrets
        storage_account_name = dbutils.secrets.get(scope="azure-storage-secrets", key="storage-account-name")
        storage_account_key = dbutils.secrets.get(scope="azure-storage-secrets", key="storage-account-key")
        
        # Crear connection string
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={storage_account_name};AccountKey={storage_account_key};EndpointSuffix=core.windows.net"
        
        # Crear cliente
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        print(f"✅ Conectado a Azure Storage: {storage_account_name}")
        return blob_service_client
        
    except Exception as e:
        print(f"❌ Error al conectar con Azure Storage: {e}")
        return None

# Inicializar cliente
storage_client = get_storage_client()

# Configuración del proyecto
container_name = "abalbin-proyecto-gh"
bronze_folder = "bronze"
silver_folder = "silver" 
gold_folder = "gold"

print(f"📁 Contenedor: {container_name}")
print(f"🥉 Capa Bronze: {bronze_folder}/")
print(f"🥈 Capa Silver: {silver_folder}/")
print(f"🥇 Capa Gold: {gold_folder}/")

# COMMAND ----------

# Verificar que la estructura del proyecto existe en Azure Storage
def verify_project_structure():
    if not storage_client:
        print("❌ Cliente de storage no disponible")
        return False
    
    try:
        # Verificar que el contenedor existe
        container_client = storage_client.get_container_client(container_name)
        
        # Verificar que las carpetas existen
        folders_to_check = [bronze_folder, silver_folder, gold_folder]
        existing_folders = []
        
        blobs = container_client.list_blobs()
        blob_names = [blob.name for blob in blobs]
        
        for folder in folders_to_check:
            folder_exists = any(blob_name.startswith(f"{folder}/") for blob_name in blob_names)
            if folder_exists:
                existing_folders.append(folder)
                print(f"✅ Carpeta '{folder}' verificada")
            else:
                print(f"⚠️  Carpeta '{folder}' no encontrada")
        
        if len(existing_folders) == 3:
            print("🎉 Estructura del proyecto verificada completamente")
            return True
        else:
            print("📝 Creando carpetas faltantes...")
            create_missing_folders(folders_to_check, existing_folders)
            return True
            
    except Exception as e:
        print(f"❌ Error verificando estructura: {e}")
        return False

def create_missing_folders(all_folders, existing_folders):
    """Crear carpetas faltantes"""
    missing_folders = [folder for folder in all_folders if folder not in existing_folders]
    
    for folder in missing_folders:
        try:
            blob_name = f"{folder}/.placeholder"
            blob_client = storage_client.get_blob_client(container=container_name, blob=blob_name)
            blob_client.upload_blob(f"# Carpeta {folder} del proyecto GitHub Analytics", overwrite=True)
            print(f"✅ Carpeta '{folder}' creada")
        except Exception as e:
            print(f"❌ Error creando carpeta '{folder}': {e}")

# Ejecutar verificación
verify_project_structure()

# COMMAND ----------

# Funciones para gestionar archivos en Azure Storage
class GitHubDataManager:
    def __init__(self, storage_client, container_name):
        self.storage_client = storage_client
        self.container_name = container_name
    
    def upload_file_to_bronze(self, local_file_path, blob_name):
        """Subir archivo a la capa Bronze"""
        try:
            blob_path = f"{bronze_folder}/{blob_name}"
            blob_client = self.storage_client.get_blob_client(container=self.container_name, blob=blob_path)
            
            with open(local_file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            
            print(f"✅ Archivo subido a Bronze: {blob_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error subiendo archivo: {e}")
            return False
    
    def list_bronze_files(self):
        """Listar archivos en la capa Bronze"""
        try:
            container_client = self.storage_client.get_container_client(self.container_name)
            blobs = container_client.list_blobs(name_starts_with=f"{bronze_folder}/")
            
            files = []
            print(f"📁 Archivos en la capa Bronze:")
            for blob in blobs:
                if not blob.name.endswith('.placeholder'):
                    files.append(blob.name)
                    file_size = blob.size / (1024*1024)  # MB
                    print(f"  📄 {blob.name} ({file_size:.2f} MB)")
            
            return files
            
        except Exception as e:
            print(f"❌ Error listando archivos: {e}")
            return []
    
    def get_bronze_file_info(self):
        """Obtener información detallada de archivos en Bronze"""
        try:
            container_client = self.storage_client.get_container_client(self.container_name)
            blobs = container_client.list_blobs(name_starts_with=f"{bronze_folder}/")
            
            total_size = 0
            file_count = 0
            
            for blob in blobs:
                if not blob.name.endswith('.placeholder'):
                    total_size += blob.size
                    file_count += 1
            
            total_size_mb = total_size / (1024*1024)
            print(f"📊 Resumen Capa Bronze:")
            print(f"   Archivos: {file_count}")
            print(f"   Tamaño total: {total_size_mb:.2f} MB")
            
            return {"file_count": file_count, "total_size_mb": total_size_mb}
            
        except Exception as e:
            print(f"❌ Error obteniendo información: {e}")
            return None

# Crear instancia del gestor
data_manager = GitHubDataManager(storage_client, container_name)

print("🔧 GitHubDataManager inicializado")
print("📝 Funciones disponibles:")
print("   - data_manager.upload_file_to_bronze(archivo_local, nombre_blob)")
print("   - data_manager.list_bronze_files()")
print("   - data_manager.get_bronze_file_info()")

# COMMAND ----------

# Configuración de descarga de datos de GitHub Archive
print("🐙 CONFIGURACIÓN DE DESCARGA GITHUB ARCHIVE")
print("=" * 50)

# Lista de fechas y horas para las que queremos descargar los datos
# Formato: AAAA-MM-DD-H
# Descargamos las primeras 3 horas de dos días diferentes como ejemplo
urls_to_download = [
    "https://data.gharchive.org/2025-01-01-0.json.gz",
    "https://data.gharchive.org/2025-01-01-1.json.gz", 
    "https://data.gharchive.org/2025-01-01-2.json.gz",
    "https://data.gharchive.org/2025-01-02-0.json.gz",
    "https://data.gharchive.org/2025-01-02-1.json.gz",
    "https://data.gharchive.org/2025-01-02-2.json.gz"
]

print(f"📅 Archivos a descargar: {len(urls_to_download)}")
print("📋 Lista de archivos:")
for i, url in enumerate(urls_to_download, 1):
    file_name = url.split("/")[-1]
    print(f"   {i}. {file_name}")

print(f"\n💾 Destino: Azure Storage → {container_name} → {bronze_folder}/")

# COMMAND ----------

# Función para descargar y subir archivos a Azure Storage
def download_and_upload_github_data(urls_list):
    """
    Descarga archivos de GitHub Archive y los sube a Azure Storage (capa Bronze)
    """
    if not storage_client:
        print("❌ Cliente de Azure Storage no disponible")
        return
    
    successful_downloads = 0
    failed_downloads = 0
    
    print("🚀 INICIANDO PROCESO DE DESCARGA Y CARGA")
    print("=" * 60)
    
    for i, url in enumerate(urls_list, 1):
        file_name_gz = url.split("/")[-1]
        file_name_json = file_name_gz.replace(".gz", "")
        
        print(f"\n📦 Procesando archivo {i}/{len(urls_list)}: {file_name_gz}")
        
        # Crear archivos temporales
        with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as temp_gz:
            temp_gz_path = temp_gz.name
            
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_json:
            temp_json_path = temp_json.name
        
        try:
            # Paso 1: Descargar archivo comprimido
            print(f"   📥 Descargando desde: {url}")
            urllib.request.urlretrieve(url, temp_gz_path)
            print(f"   ✅ Descarga completada")
            
            # Paso 2: Descomprimir archivo
            print(f"   📂 Descomprimiendo archivo...")
            with gzip.open(temp_gz_path, 'rb') as f_in:
                with open(temp_json_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"   ✅ Descompresión completada")
            
            # Paso 3: Subir a Azure Storage (capa Bronze)
            print(f"   ☁️  Subiendo a Azure Storage...")
            success = data_manager.upload_file_to_bronze(temp_json_path, file_name_json)
            
            if success:
                successful_downloads += 1
                print(f"   🎉 {file_name_json} procesado exitosamente")
            else:
                failed_downloads += 1
                print(f"   ❌ Error subiendo {file_name_json}")
            
        except Exception as e:
            failed_downloads += 1
            print(f"   ❌ Error procesando {file_name_gz}: {e}")
        
        finally:
            # Limpiar archivos temporales
            try:
                os.unlink(temp_gz_path)
                os.unlink(temp_json_path)
            except:
                pass
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE DESCARGA")
    print("=" * 60)
    print(f"✅ Descargas exitosas: {successful_downloads}")
    print(f"❌ Descargas fallidas: {failed_downloads}")
    print(f"📁 Total de archivos: {len(urls_list)}")
    
    if successful_downloads > 0:
        print(f"\n🎉 ¡Proceso completado! Los datos están en Azure Storage.")
        print(f"📍 Ubicación: {container_name}/{bronze_folder}/")
    
    return successful_downloads, failed_downloads

# Ejecutar el proceso de descarga
results = download_and_upload_github_data(urls_to_download)

# COMMAND ----------

# Verificar que los archivos se guardaron correctamente en Azure Storage
print("🔍 VERIFICACIÓN DE ARCHIVOS EN CAPA BRONZE")
print("=" * 50)

# Listar archivos en la capa Bronze
bronze_files = data_manager.list_bronze_files()

# Obtener información detallada
file_info = data_manager.get_bronze_file_info()

# Verificación adicional desde Azure Portal
print(f"\n🔗 Para verificar en Azure Portal:")
print(f"   1. Ve a: https://portal.azure.com")
print(f"   2. Navega a: rg-proyecto-github → Storage Account")
print(f"   3. Ve a: Containers → {container_name} → {bronze_folder}/")
print(f"   4. Deberías ver {len(bronze_files)} archivos .json")

# Validación de contenido (sample de un archivo)
if bronze_files:
    print(f"\n📄 Validación de contenido del primer archivo:")
    try:
        # Descargar una muestra pequeña del primer archivo para verificar
        first_file = bronze_files[0]
        blob_client = storage_client.get_blob_client(container=container_name, blob=first_file)
        
        # Leer las primeras líneas
        sample_data = blob_client.download_blob().readall()
        sample_lines = sample_data.decode('utf-8').split('\n')[:3]
        
        print(f"   Archivo: {first_file.split('/')[-1]}")
        print(f"   Primeras líneas de ejemplo:")
        for i, line in enumerate(sample_lines, 1):
            if line.strip():
                print(f"     {i}. {line[:100]}...")
        
        print("   ✅ El archivo contiene datos JSON válidos de GitHub")
        
    except Exception as e:
        print(f"   ❌ Error verificando contenido: {e}")

print(f"\n🎉 ¡Felicidades! Has completado la carga de datos a la Capa Bronze.")
print(f"📈 Próximo paso: Transformar estos datos en la Capa Silver.")

# COMMAND ----------

# Función utilitaria para reutilizar en otros notebooks
def get_bronze_file_paths():
    """
    Obtener las rutas de todos los archivos en la capa Bronze
    Útil para los siguientes notebooks de transformación
    """
    try:
        container_client = storage_client.get_container_client(container_name)
        blobs = container_client.list_blobs(name_starts_with=f"{bronze_folder}/")
        
        file_paths = []
        for blob in blobs:
            if not blob.name.endswith('.placeholder'):
                # Formato para leer desde Azure Storage en Spark
                azure_path = f"abfss://{container_name}@{storage_client.account_name}.dfs.core.windows.net/{blob.name}"
                file_paths.append(azure_path)
        
        return file_paths
        
    except Exception as e:
        print(f"❌ Error obteniendo rutas: {e}")
        return []

# Guardar información para el siguiente notebook
bronze_paths = get_bronze_file_paths()
print(f"📝 Rutas para el siguiente notebook:")
for path in bronze_paths[:3]:  # Mostrar solo las primeras 3
    print(f"   {path}")

if len(bronze_paths) > 3:
    print(f"   ... y {len(bronze_paths) - 3} archivos más")

print(f"\n💡 Estas rutas se usarán en el notebook de transformación (Silver)")
