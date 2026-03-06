# =============================================================================
# TALLER 1 – ETL con PySpark | Online Retail Dataset
# Fuente: https://archive.ics.uci.edu/ml/datasets/Online+Retail
# =============================================================================
# DEPENDENCIAS (instalar una sola vez):
#   pip install pyspark openpyxl pandas requests
#
# EJECUTAR:
#   spark-submit etl_online_retail.py
# =============================================================================

import os
import sys
import zipfile
from pathlib import Path

import requests
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ─────────────────────────────────────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────────────────────────────────────

# Carpeta donde está este script → todo se guarda aquí
BASE_DIR   = Path(__file__).resolve().parent

# Carpeta data/ → aquí se guarda el ZIP, el Excel y el CSV
DATA_DIR   = BASE_DIR / "data"

# Carpeta resultados/ → aquí va un CSV por cada pregunta
OUTPUT_DIR = BASE_DIR / "resultados"

# Archivos individuales
ZIP_PATH   = DATA_DIR / "online_retail.zip"         # ZIP descargado
EXCEL_PATH = DATA_DIR / "Online Retail.xlsx"         # Excel extraído del ZIP
CSV_PATH   = DATA_DIR / "Online_Retail.csv"          # CSV listo para Spark

# URL EXACTA del taller — descarga directa del ZIP desde UCI
URL_DESCARGA = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"

# Crear carpetas si no existen
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# PASO 1 — DESCARGAR EL ZIP DESDE UCI
# ─────────────────────────────────────────────────────────────────────────────

def descargar_zip():
    """
    Descarga el archivo ZIP desde la URL oficial del taller UCI.
    URL: https://archive.ics.uci.edu/static/public/352/online+retail.zip

    El ZIP contiene el archivo 'Online Retail.xlsx' (22.6 MB).
    Si el ZIP o el CSV ya existen localmente, omite la descarga.
    """

    # Si el CSV ya existe, no hay nada que descargar
    if CSV_PATH.exists():
        print(f"✅ CSV ya existe en data/ — omitiendo descarga")
        return

    # Si el ZIP ya existe, solo extraemos
    if ZIP_PATH.exists():
        print(f"✅ ZIP ya descargado — pasando a extracción")
        return

    print("=" * 60)
    print("  DESCARGANDO DATASET DESDE UCI")
    print(f"  URL: {URL_DESCARGA}")
    print("  Destino: data/online_retail.zip (~22.6 MB)")
    print("=" * 60)

    try:
        # requests.get con stream=True descarga en bloques pequeños
        # para no cargar los 22 MB completos en RAM de una sola vez
        respuesta = requests.get(URL_DESCARGA, stream=True, timeout=180)

        # Si el servidor devuelve error (404, 500, etc.) lanza excepción
        respuesta.raise_for_status()

        # Tamaño total en bytes (para mostrar progreso)
        tam_total  = int(respuesta.headers.get("content-length", 0))
        descargado = 0

        print(f"\n  Descargando...")

        # Escribimos el ZIP en bloques de 8 KB
        with open(ZIP_PATH, "wb") as f:
            for bloque in respuesta.iter_content(chunk_size=8192):
                if bloque:
                    f.write(bloque)
                    descargado += len(bloque)
                    if tam_total:
                        pct = (descargado / tam_total) * 100
                        mb  = descargado / 1024 / 1024
                        tot = tam_total  / 1024 / 1024
                        # \r sobreescribe la línea anterior (barra de progreso)
                        print(f"\r  [{pct:5.1f}%]  {mb:.1f} MB / {tot:.1f} MB",
                              end="", flush=True)

        print(f"\n✅ ZIP descargado: {ZIP_PATH}\n")

    except requests.exceptions.ConnectionError:
        print("\n❌ Sin conexión a internet.")
        sys.exit(1)

    except requests.exceptions.Timeout:
        print("\n❌ Timeout — la descarga tardó demasiado. Intenta de nuevo.")
        sys.exit(1)

    except requests.exceptions.HTTPError as e:
        print(f"\n❌ Error HTTP: {e}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# PASO 2 — EXTRAER EL EXCEL DEL ZIP
# ─────────────────────────────────────────────────────────────────────────────

def extraer_zip():
    """
    Extrae el archivo Excel del ZIP descargado.
    El ZIP de UCI contiene: 'Online Retail.xlsx'
    Lo extraemos en la carpeta data/
    """

    if EXCEL_PATH.exists():
        print(f"✅ Excel ya existe en data/ — omitiendo extracción")
        return

    print("📦 Extrayendo Excel del ZIP...")

    # zipfile.ZipFile abre el ZIP como si fuera una carpeta
    with zipfile.ZipFile(ZIP_PATH, "r") as z:

        # Listamos el contenido del ZIP para ver el nombre exacto del archivo
        archivos = z.namelist()
        print(f"   Contenido del ZIP: {archivos}")

        # Buscamos el .xlsx dentro del ZIP (sea cual sea su nombre exacto)
        xlsx_en_zip = [a for a in archivos if a.endswith(".xlsx")]

        if not xlsx_en_zip:
            print("❌ No se encontró ningún .xlsx dentro del ZIP.")
            sys.exit(1)

        # Extraemos el primer .xlsx encontrado a la carpeta data/
        nombre_en_zip = xlsx_en_zip[0]
        z.extract(nombre_en_zip, DATA_DIR)

        # Si el nombre del archivo extraído es diferente al esperado, lo renombramos
        ruta_extraida = DATA_DIR / nombre_en_zip
        if ruta_extraida != EXCEL_PATH:
            ruta_extraida.rename(EXCEL_PATH)

    print(f"✅ Excel extraído: {EXCEL_PATH}\n")


# ─────────────────────────────────────────────────────────────────────────────
# PASO 3 — CONVERTIR EXCEL A CSV
# ─────────────────────────────────────────────────────────────────────────────

def convertir_a_csv():
    """
    Convierte el Excel a CSV usando pandas.
    PySpark lee CSV nativamente sin dependencias extra.

    pandas.read_excel usa openpyxl como motor para archivos .xlsx.
    Guardamos con encoding utf-8 para mantener caracteres especiales.
    """

    if CSV_PATH.exists():
        print(f"✅ CSV ya existe en data/ — omitiendo conversión")
        return

    print("🔄 Convirtiendo Excel → CSV (puede tardar 20-40 segundos)...")

    # Lee el Excel completo como strings para no perder datos
    # (CustomerID como "12345" no se convierte a 12345.0)
    df = pd.read_excel(EXCEL_PATH, engine="openpyxl", dtype=str)

    print(f"   Filas leídas: {len(df):,}")
    print(f"   Columnas:     {list(df.columns)}")

    # Guarda como CSV — index=False para no agregar columna extra de índice
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    mb = CSV_PATH.stat().st_size / 1024 / 1024
    print(f"✅ CSV guardado: {CSV_PATH} ({mb:.1f} MB)\n")


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN HELPER — GUARDAR CSV POR PREGUNTA
# ─────────────────────────────────────────────────────────────────────────────

def guardar_csv(df_spark, nombre):
    """
    Guarda un DataFrame de Spark como un único archivo CSV en resultados/

    coalesce(1)        → fuerza UN solo archivo en vez de múltiples partes
    mode("overwrite")  → sobreescribe si ya existe
    header(True)       → incluye fila de encabezado con nombres de columnas
    """
    ruta = str(OUTPUT_DIR / nombre)
    df_spark.coalesce(1).write.csv(ruta, mode="overwrite", header=True)
    print(f"   💾 CSV guardado → resultados/{nombre}/part-00000.csv")


# =============================================================================
# INICIO — DESCARGA + CONVERSIÓN + ETL
# =============================================================================

print("\n" + "=" * 60)
print("  TALLER 1 – ETL PySpark | Online Retail Dataset")
print("=" * 60 + "\n")

# Ejecutar los 3 pasos de preparación del dato
descargar_zip()
extraer_zip()
convertir_a_csv()

# ─────────────────────────────────────────────────────────────────────────────
# SPARKSESSION
# ─────────────────────────────────────────────────────────────────────────────

# Creamos la sesión Spark en modo local usando todos los cores del CPU
spark = SparkSession.builder \
    .appName("ETL Online Retail - Taller 1") \
    .master("local[*]") \
    .getOrCreate()

# Silenciamos logs INFO para ver solo nuestros mensajes
spark.sparkContext.setLogLevel("WARN")

print("✅ SparkSession iniciada\n")

# ─────────────────────────────────────────────────────────────────────────────
# LECTURA DEL CSV CON SPARK  (Operación 1: spark.read.format)
# ─────────────────────────────────────────────────────────────────────────────

# header=true     → primera fila = nombres de columnas
# inferSchema=true → Spark deduce el tipo de dato de cada columna
df_raw = spark.read \
    .format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(str(CSV_PATH))

print("📌 ESQUEMA DEL DATASET:")
df_raw.printSchema()

print("📌 MUESTRA (primeras 5 filas):")
df_raw.show(5, truncate=False)

total_raw = df_raw.count()
print(f"📌 Total filas crudas: {total_raw:,}\n")

# ─────────────────────────────────────────────────────────────────────────────
# LIMPIEZA  (Operaciones 2 select, 3 filter, 7 withColumn)
# ─────────────────────────────────────────────────────────────────────────────

# Operación 2 — select(): elegir las 8 columnas del dataset
df = df_raw.select(
    F.col("InvoiceNo"),    # número de factura (empieza con C si es devolución)
    F.col("StockCode"),    # código del producto
    F.col("Description"),  # nombre del producto
    F.col("Quantity"),     # unidades (negativo = devolución)
    F.col("InvoiceDate"),  # fecha y hora
    F.col("UnitPrice"),    # precio unitario en £
    F.col("CustomerID"),   # ID del cliente (puede ser null)
    F.col("Country")       # país del cliente
)

# Operación 3 — filter(): eliminar registros inválidos
df = df.filter(
    (F.col("Quantity")  > 0) &                      # sin devoluciones
    (F.col("UnitPrice") > 0) &                      # sin precios cero
    (F.col("CustomerID").isNotNull()) &              # sin clientes null
    (F.col("CustomerID").cast("string") != "")      # sin clientes vacíos
)

# Operación 7 — withColumn(): columnas calculadas nuevas

# TotalLine = Quantity × UnitPrice → ingreso de cada línea
df = df.withColumn(
    "TotalLine",
    F.round(F.col("Quantity") * F.col("UnitPrice"), 2)
)

# InvoiceMonth = número de mes (1-12) extraído de la fecha
df = df.withColumn(
    "InvoiceMonth",
    F.month(F.to_timestamp(F.col("InvoiceDate"), "M/d/yyyy H:mm"))
)

total_clean = df.count()
print(f"📌 Filas limpias:   {total_clean:,}")
print(f"   Filas quitadas: {total_raw - total_clean:,}\n")


# =============================================================================
# LAS 10 PREGUNTAS — CADA UNA GUARDA SU PROPIO CSV EN resultados/
# =============================================================================

print("=" * 60)
print("  RESPONDIENDO LAS 10 PREGUNTAS")
print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# P1 — ¿Cuál es el número total de facturas en el dataset?
# ─────────────────────────────────────────────────────────────────────────────
# distinct() elimina duplicados → una factura aparece en varias filas
# count() cuenta las facturas únicas
print("─" * 50)
print("P1: ¿Cuántas facturas únicas hay?")

total_facturas = df.select("InvoiceNo").distinct().count()
print(f"    ➜ {total_facturas:,} facturas\n")

df_p1 = spark.createDataFrame([(total_facturas,)], ["total_facturas"])
guardar_csv(df_p1, "p1_total_facturas")


# ─────────────────────────────────────────────────────────────────────────────
# P2 — ¿Cuál es el número de clientes únicos?
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 50)
print("P2: ¿Cuántos clientes únicos hay?")

total_clientes = df.select("CustomerID").distinct().count()
print(f"    ➜ {total_clientes:,} clientes únicos\n")

df_p2 = spark.createDataFrame([(total_clientes,)], ["total_clientes_unicos"])
guardar_csv(df_p2, "p2_clientes_unicos")


# ─────────────────────────────────────────────────────────────────────────────
# P3 — ¿Cuál es el ingreso total (Quantity * UnitPrice)?
# ─────────────────────────────────────────────────────────────────────────────
# agg() sin groupBy → agrega sobre todo el DataFrame completo
print("─" * 50)
print("P3: ¿Cuál es el ingreso total?")

ingreso_total = df.agg(
    F.round(F.sum("TotalLine"), 2).alias("ingreso_total_GBP")
).collect()[0][0]

print(f"    ➜ £{ingreso_total:,.2f}\n")

df_p3 = spark.createDataFrame([(ingreso_total,)], ["ingreso_total_GBP"])
guardar_csv(df_p3, "p3_ingreso_total")


# ─────────────────────────────────────────────────────────────────────────────
# P4 — ¿Qué producto fue el más vendido en cantidad?
# ─────────────────────────────────────────────────────────────────────────────
# groupBy por producto → sum de unidades → orderBy descendente
print("─" * 50)
print("P4: ¿Qué producto fue el más vendido?")

df_p4 = df \
    .groupBy("StockCode", "Description") \
    .agg(F.sum("Quantity").alias("total_unidades_vendidas")) \
    .orderBy(F.desc("total_unidades_vendidas")) \
    .limit(10)

df_p4.show(5, truncate=False)
guardar_csv(df_p4, "p4_producto_mas_vendido")


# ─────────────────────────────────────────────────────────────────────────────
# P5 — ¿Cuál es el cliente con mayor volumen de compra en dinero?
# ─────────────────────────────────────────────────────────────────────────────
# groupBy por cliente → sum(TotalLine) → orderBy descendente
print("─" * 50)
print("P5: ¿Quién es el cliente con mayor gasto?")

df_p5 = df \
    .groupBy("CustomerID") \
    .agg(
        F.round(F.sum("TotalLine"), 2).alias("total_compras_GBP"),
        F.countDistinct("InvoiceNo").alias("numero_de_facturas")
    ) \
    .orderBy(F.desc("total_compras_GBP")) \
    .limit(10)

df_p5.show(5, truncate=False)
guardar_csv(df_p5, "p5_cliente_mayor_compra")


# ─────────────────────────────────────────────────────────────────────────────
# P6 — ¿Cuáles son los 5 países que más compran fuera de Reino Unido?
# ─────────────────────────────────────────────────────────────────────────────
# filter para excluir UK → groupBy Country → orderBy ventas desc
print("─" * 50)
print("P6: Top 5 países fuera de UK")

df_p6 = df \
    .filter(F.col("Country") != "United Kingdom") \
    .groupBy("Country") \
    .agg(
        F.round(F.sum("TotalLine"), 2).alias("total_ventas_GBP"),
        F.countDistinct("CustomerID").alias("clientes_unicos"),
        F.countDistinct("InvoiceNo").alias("numero_facturas")
    ) \
    .orderBy(F.desc("total_ventas_GBP")) \
    .limit(5)

df_p6.show(truncate=False)
guardar_csv(df_p6, "p6_top5_paises_fuera_uk")


# ─────────────────────────────────────────────────────────────────────────────
# P7 — ¿Cuál es el ticket promedio por factura?
# ─────────────────────────────────────────────────────────────────────────────
# Paso 1: total de cada factura (suma de sus líneas)
# Paso 2: avg de esos totales
print("─" * 50)
print("P7: ¿Cuál es el ticket promedio por factura?")

df_totales_factura = df \
    .groupBy("InvoiceNo") \
    .agg(F.sum("TotalLine").alias("total_factura"))

ticket_promedio = df_totales_factura \
    .agg(F.round(F.avg("total_factura"), 2).alias("ticket_promedio_GBP")) \
    .collect()[0][0]

print(f"    ➜ £{ticket_promedio:,.2f} promedio por factura\n")

df_p7 = spark.createDataFrame([(ticket_promedio,)], ["ticket_promedio_GBP"])
guardar_csv(df_p7, "p7_ticket_promedio_factura")


# ─────────────────────────────────────────────────────────────────────────────
# P8 — ¿Cuál es el mínimo, máximo y promedio de productos por factura?
# ─────────────────────────────────────────────────────────────────────────────
# count("StockCode") → cuántas líneas tiene cada factura
# luego min/max/avg sobre esos conteos
print("─" * 50)
print("P8: Min, Máx y Promedio de productos por factura")

df_items = df \
    .groupBy("InvoiceNo") \
    .agg(F.count("StockCode").alias("num_productos"))

stats = df_items.agg(
    F.min("num_productos").alias("minimo"),
    F.max("num_productos").alias("maximo"),
    F.round(F.avg("num_productos"), 2).alias("promedio")
).collect()[0]

print(f"    ➜ Mínimo: {stats[0]}  |  Máximo: {stats[1]}  |  Promedio: {stats[2]}\n")

df_p8 = spark.createDataFrame(
    [(stats[0], stats[1], stats[2])],
    ["min_productos_por_factura", "max_productos_por_factura", "promedio_productos_por_factura"]
)
guardar_csv(df_p8, "p8_estadisticas_productos_factura")


# ─────────────────────────────────────────────────────────────────────────────
# P9 — ¿Qué mes del año tuvo más ventas?
# ─────────────────────────────────────────────────────────────────────────────
# InvoiceMonth ya calculado (número 1-12)
# groupBy mes → sum ventas → orderBy desc
print("─" * 50)
print("P9: ¿Qué mes tuvo más ventas?")

df_p9 = df \
    .groupBy("InvoiceMonth") \
    .agg(
        F.round(F.sum("TotalLine"), 2).alias("ventas_totales_GBP"),
        F.countDistinct("InvoiceNo").alias("numero_de_facturas")
    ) \
    .orderBy(F.desc("ventas_totales_GBP"))

df_p9.show(12, truncate=False)
guardar_csv(df_p9, "p9_ventas_por_mes")


# ─────────────────────────────────────────────────────────────────────────────
# P10 — ¿Cuál es el porcentaje de facturas con devoluciones?
# ─────────────────────────────────────────────────────────────────────────────
# Usamos df_RAW (sin limpiar) porque las devoluciones fueron filtradas en df
# En UCI las devoluciones tienen InvoiceNo que empieza con "C"
# when(condicion, 1).otherwise(0) → marca cada factura como devolución o no
print("─" * 50)
print("P10: ¿Qué % de facturas son devoluciones?")

df_fact_brutas = df_raw.select("InvoiceNo").distinct()
df_fact_brutas = df_fact_brutas.withColumn(
    "es_devolucion",
    F.when(F.col("InvoiceNo").startswith("C"), F.lit(1)).otherwise(F.lit(0))
)

total_fact_brutas  = df_fact_brutas.count()
fact_con_devolucion = df_fact_brutas.filter(F.col("es_devolucion") == 1).count()
pct_devoluciones   = round((fact_con_devolucion / total_fact_brutas) * 100, 2)

print(f"    ➜ {fact_con_devolucion} devoluciones de {total_fact_brutas} "
      f"facturas = {pct_devoluciones}%\n")

df_p10 = spark.createDataFrame(
    [(total_fact_brutas, fact_con_devolucion, pct_devoluciones)],
    ["total_facturas_brutas", "facturas_con_devolucion", "porcentaje_devoluciones"]
)
guardar_csv(df_p10, "p10_porcentaje_devoluciones")


# =============================================================================
# OPERACIONES EXTRAS DEL TALLER (orderBy, join, window)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# EXTRA — orderBy(): top 10 transacciones por valor
# ─────────────────────────────────────────────────────────────────────────────
df_extra_orden = df \
    .select("InvoiceNo", "Description", "Quantity", "UnitPrice", "TotalLine") \
    .orderBy(F.desc("TotalLine")) \
    .limit(10)

guardar_csv(df_extra_orden, "extra_orderby_top10_transacciones")


# ─────────────────────────────────────────────────────────────────────────────
# EXTRA — join(): facturas unidas con perfil del cliente
# ─────────────────────────────────────────────────────────────────────────────
# DF A: resumen por factura
df_resumen = df \
    .groupBy("InvoiceNo", "CustomerID", "Country") \
    .agg(F.round(F.sum("TotalLine"), 2).alias("total_factura"))

# DF B: perfil del cliente
df_perfil = df \
    .groupBy("CustomerID") \
    .agg(
        F.round(F.sum("TotalLine"), 2).alias("gasto_historico_GBP"),
        F.countDistinct("InvoiceNo").alias("total_facturas_cliente")
    )

# inner join por CustomerID
df_extra_join = df_resumen.join(df_perfil, on="CustomerID", how="inner") \
    .orderBy(F.desc("gasto_historico_GBP"))

guardar_csv(df_extra_join, "extra_join_facturas_clientes")


# ─────────────────────────────────────────────────────────────────────────────
# EXTRA — window functions: top 3 clientes por país
# ─────────────────────────────────────────────────────────────────────────────
# partitionBy("Country") → divide por país
# rank().over(ventana)   → asigna posición dentro de cada país
df_cli_pais = df \
    .groupBy("CustomerID", "Country") \
    .agg(F.round(F.sum("TotalLine"), 2).alias("total_GBP"))

ventana = Window.partitionBy("Country").orderBy(F.desc("total_GBP"))

df_extra_window = df_cli_pais \
    .withColumn("ranking_en_pais", F.rank().over(ventana)) \
    .withColumn("posicion_en_pais", F.row_number().over(ventana)) \
    .filter(F.col("ranking_en_pais") <= 3) \
    .orderBy("Country", "ranking_en_pais")

guardar_csv(df_extra_window, "extra_window_top3_por_pais")


# ─────────────────────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ✅ TALLER COMPLETADO — RESUMEN")
print("=" * 60)
print(f"  P1.  Facturas únicas:             {total_facturas:>10,}")
print(f"  P2.  Clientes únicos:             {total_clientes:>10,}")
print(f"  P3.  Ingreso total:               £{ingreso_total:>10,.2f}")
print(f"  P7.  Ticket promedio/factura:     £{ticket_promedio:>10,.2f}")
print(f"  P8.  Min={stats[0]}  Max={stats[1]}  Avg={stats[2]} prods/factura")
print(f"  P10. Devoluciones:                {pct_devoluciones:>9}%")
print(f"\n  📁 CSVs en: {OUTPUT_DIR}")
print("=" * 60)

spark.stop()
print("\n✅ ¡ETL finalizado correctamente!\n")