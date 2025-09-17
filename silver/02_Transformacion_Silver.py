# Databricks notebook source
# Obtener credenciales desde secrets
storage_account_name = dbutils.secrets.get(scope="azure-storage-secrets", key="storage-account-name")
storage_account_key = dbutils.secrets.get(scope="azure-storage-secrets", key="storage-account-key")

# Cuenta de almacenamiento
container_name = "abalbin-proyecto-gh"

# Definimos las rutas de origen (bronze) y destino (silver)
bronze_path = f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/bronze/repos_bronze"
silver_path = f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/silver/repos_silver"

print(f"Ruta de Origen (Bronze): {bronze_path}")
print(f"Ruta de Destino (Silver): {silver_path}")

# COMMAND ----------

# Creamos un DataFrame simple
data = [("Augusto", 1), ("Databricks", 2)]
columns = ["nombre", "valor"]

df_test = spark.createDataFrame(data, columns)

# Definimos la ruta completa de destino
ruta_escritura = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/demo_test"

# Escribimos el archivo Parquet en Azure Blob
df_test.write \
    .mode("overwrite") \
    .option(f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net", storage_account_key) \
    .parquet(ruta_escritura)

# COMMAND ----------

df_leido = spark.read \
    .option(f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net", storage_account_key) \
    .parquet(ruta_escritura)

df_leido.show()

# COMMAND ----------

# Rutas locales en DBFS (Databricks File System)
bronze_path = "/FileStore/tables/bronze"
silver_path = "/FileStore/tables/silver"
gold_path = "/FileStore/tables/gold"

print(f"Ruta Bronze: {bronze_path}")
print(f"Ruta Silver: {silver_path}")
print(f"Ruta Gold: {gold_path}")


# COMMAND ----------

# Leemos todos los archivos subidos (ajusta si cambian los nombres)
df_bronze = (
    spark.read
    .option("multiline", "true")
    .json("/Volumes/workspace/default/archivos-json/*.json")
)

df_bronze.display()
df_bronze.show(3, truncate=False)  # Muestra las primeras 10 filas sin truncar texto


# COMMAND ----------

import os
# Listar archivos desde ruta del DBFS
json_files = dbutils.fs.ls("dbfs:/Volumes/workspace/default/archivos-json/")
print(f"Se encontraron {len(json_files)} archivos JSON.")

# COMMAND ----------

df_bronze.printSchema()
df_bronze.selectExpr("*").display()

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

raw_df = spark.read.option("multiline", True).json("/Workspace/path/to/json_files/*.json")

# COMMAND ----------

from pyspark.sql.functions import col, to_timestamp, explode

df_silver = df_bronze.select(
    col("id").alias("event_id"),
    col("type").alias("event_type"),
    col("actor.id").alias("actor_id"),
    col("actor.login").alias("actor_login"),
    col("actor.display_login").alias("actor_display_login"),
    col("repo.id").alias("repo_id"),
    col("repo.name").alias("repo_name"),
    col("payload.push_id").alias("push_id"),
    col("payload.commits").alias("commits"),
    to_timestamp("created_at", "yyyy-MM-dd'T'HH:mm:ss'Z'").alias("event_time"),
    col("public").alias("public")  # <--- Agregado
)

# COMMAND ----------

# ==================================================== 
# 0️⃣ IMPORTS Y CONFIGURACIÓN
# ====================================================
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Lista de archivos JSON en Volumes (lectura sigue igual)
json_files = [
    "/Volumes/workspace/default/archivos-json/2025-01-01-0.json",
    "/Volumes/workspace/default/archivos-json/2025-01-01-1.json",
    "/Volumes/workspace/default/archivos-json/2025-01-01-2.json",
    "/Volumes/workspace/default/archivos-json/2025-01-02-0.json",
    "/Volumes/workspace/default/archivos-json/2025-01-02-1.json",
    "/Volumes/workspace/default/archivos-json/2025-01-02-2.json"
]

# ====================================================
# 1️⃣ CARGA DE BRONZE Y APLANAMIENTO
# ====================================================
df_silver = (
    spark.read
    .option("multiline", "true")
    .json(json_files)  # lee los 6 archivos de la lista
    .select(
        F.col("id").alias("event_id"),
        F.col("type").alias("event_type"),
        F.col("actor.id").alias("actor_id"),
        F.col("actor.login").alias("actor_login"),
        F.col("actor.display_login").alias("actor_display_login"),
        F.col("repo.id").alias("repo_id"),
        F.col("repo.name").alias("repo_name"),
        F.col("created_at").alias("event_time"),
        F.col("public").alias("is_public"),
        F.col("payload.push_id").alias("push_id"),
        F.col("payload.commits").alias("commits")
    )
)

print("✅ Datos cargados desde Bronze (6 archivos en Volumes)")
display(df_silver.limit(5))

# ====================================================
# 2️⃣ LIMPIEZA DE DATOS
# ====================================================
df_limpio = (
    df_silver
    .dropDuplicates(["event_id"])                  # Quitar duplicados
    .withColumn("actor_login", F.lower(F.col("actor_login")))
    .withColumn("repo_name", F.lower(F.col("repo_name")))
    .filter(F.col("event_time").isNotNull())       # Filtramos nulos críticos
)

print("✅ Datos limpios")
display(df_limpio.limit(5))

# ====================================================
# 3️⃣ MODELADO DIMENSIONAL
# ====================================================
fact_events = df_limpio.select(
    "event_id", "event_time", "event_type",
    "repo_id", "actor_id", "push_id"
)

dim_actor = df_limpio.select(
    "actor_id", "actor_login", "actor_display_login"
).dropDuplicates(["actor_id"])

dim_repo = df_limpio.select(
    "repo_id", "repo_name"
).dropDuplicates(["repo_id"])

# ====================================================
# 4️⃣ AGREGACIONES E INSIGHTS
# ====================================================
agg_eventos_dia = (
    df_limpio
    .groupBy(F.to_date("event_time").alias("fecha"))
    .agg(F.count("*").alias("total_eventos"))
    .orderBy("fecha")
)

agg_eventos_tipo = (
    df_limpio
    .groupBy("event_type")
    .agg(F.countDistinct("event_id").alias("total_eventos"))
    .orderBy(F.desc("total_eventos"))
)

w_actores = Window.partitionBy(F.lit(1)).orderBy(F.desc("total_eventos"))
agg_top_actores = (
    df_limpio
    .groupBy("actor_login")
    .agg(F.countDistinct("event_id").alias("total_eventos"))
    .withColumn("rank", F.row_number().over(w_actores))
    .filter(F.col("rank") <= 10)
)

w_repos = Window.partitionBy(F.lit(1)).orderBy(F.desc("total_eventos"))
agg_top_repos = (
    df_limpio
    .groupBy("repo_name")
    .agg(F.countDistinct("event_id").alias("total_eventos"))
    .withColumn("rank", F.row_number().over(w_repos))
    .filter(F.col("rank") <= 10)
)

agg_prom_commits_push = (
    df_limpio
    .filter(F.col("event_type") == "PushEvent")
    .withColumn("num_commits", F.size("commits"))
    .agg(F.avg("num_commits").alias("promedio_commits"))
)

agg_publicos = (
    df_limpio
    .groupBy("is_public")
    .agg(F.count("*").alias("total_eventos"))
)


# COMMAND ----------


# ====================================================
# 5️⃣ EXPORTACIÓN FINAL EN JSON (SOBRESCRIBIBLE)
# ====================================================
# ✅ Usamos una ruta de volumen especifica para el archivo silver
output_json_path = "/Volumes/workspace/default/archivos-json/silver-json"

# ✅ Exportamos la data limpia consolidada como un único JSON
(
    df_limpio
    .coalesce(1)  # Un solo archivo JSON
    .write
    .mode("overwrite")
    .json(output_json_path)
)

print("✅ Guardado local temporal en:", output_json_path)
