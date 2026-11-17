"""Limpieza, auditoria e integracion reproducibles para los lotes de NovaMarket."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_DIRECTORIO = Path(__file__).resolve().parent
_REFERENCIA = _DIRECTORIO / "metricas_semestre.json"
_CRM = _DIRECTORIO.parent / "S12" / "crm_clientes_novamarket.csv"
_BODEGAS = _DIRECTORIO.parent / "S12" / "bodega_categorias_novamarket.csv"
_COLUMNAS_CATEGORICAS = (
    "canal_compra",
    "metodo_pago",
    "ciudad_tienda",
    "categoria_producto",
    "nivel_satisfaccion",
    "nivel_lealtad",
)


def _cargar_referencia() -> dict[str, Any]:
    if not _REFERENCIA.exists():
        raise FileNotFoundError(
            f"No existe la referencia del semestre: {_REFERENCIA}. "
            "Genere metricas_semestre.json antes de limpiar lotes."
        )
    with _REFERENCIA.open(encoding="utf-8") as archivo:
        return json.load(archivo)


def _reportar_nulos_nuevos(
    columna: str, antes: pd.Series, despues: pd.Series
) -> None:
    nuevos = int((antes.notna() & despues.isna()).sum())
    print(f"Conversion {columna}: {nuevos:,} valores no vacios pasaron a vacios.")


def _normalizar_llave(valores: pd.Series) -> pd.Series:
    def normalizar(valor: Any) -> Any:
        if pd.isna(valor) or not str(valor).strip():
            return pd.NA
        texto = unicodedata.normalize("NFKD", str(valor).strip().casefold())
        return "".join(caracter for caracter in texto if not unicodedata.combining(caracter))

    return valores.map(normalizar).astype("string")


def _normalizar_nombres_columnas(df: pd.DataFrame) -> pd.DataFrame:
    originales = list(df.columns)
    normalizados = [str(columna).strip().casefold() for columna in originales]
    if len(set(normalizados)) != len(normalizados):
        raise ValueError(
            "Hay nombres de columnas que colisionan al normalizar mayusculas y espacios."
        )
    lote = df.copy()
    lote.columns = normalizados
    alias_canal = "canal" in lote.columns
    if alias_canal:
        if "canal_compra" in lote.columns:
            raise ValueError("El lote contiene tanto canal como canal_compra.")
        lote = lote.rename(columns={"canal": "canal_compra"})
    if originales != normalizados or alias_canal:
        print(
            f"Esquema de columnas normalizado: {len(originales):,} nombres; "
            f"alias canal -> canal_compra aplicado={alias_canal}."
        )
    return lote


def _normalizar_texto(valor: Any) -> Any:
    if pd.isna(valor):
        return pd.NA
    texto = str(valor).strip()
    texto = texto.replace("\ufffd", "")
    return texto


def _normalizar_categoria(valor: Any) -> Any:
    if pd.isna(valor):
        return pd.NA
    texto = _normalizar_llave(pd.Series([valor])).iloc[0]
    texto = str(texto).replace("electrnica", "electronica").replace("juguetera", "jugueteria")
    categorias = {
        "belleza": "Belleza",
        "deportes": "Deportes",
        "electronica": "Electronica",
        "hogar": "Hogar",
        "jugueteria": "Jugueteria",
        "moda": "Moda",
    }
    return categorias.get(texto, texto.title())


def _normalizar_ciudad(valor: Any) -> Any:
    if pd.isna(valor):
        return pd.NA
    texto = str(_normalizar_llave(pd.Series([valor])).iloc[0]).replace(",", " ")
    for sufijo in (" d.c.", " dc", " ant.", " caldas"):
        texto = texto.replace(sufijo, "")
    reemplazos = {
        "b/quilla": "barranquilla",
        "b/manga": "bucaramanga",
        "santiago de cali": "cali",
        "cartagena de indias": "cartagena",
        "santamarta": "santa marta",
        "sta. marta": "santa marta",
        "sta marta": "santa marta",
        "bog": "bogota",
        "mde": "medellin",
        "ctg": "cartagena",
        "baq": "barranquilla",
        "bga": "bucaramanga",
        "mzl": "manizales",
        "smr": "santa marta",
    }
    texto = " ".join(texto.split())
    return reemplazos.get(texto, texto).title()


def _normalizar_canal(valor: Any) -> Any:
    if pd.isna(valor):
        return pd.NA
    texto = " ".join(str(valor).strip().split())
    clave = str(_normalizar_llave(pd.Series([texto])).iloc[0])
    conocidos = {
        "app": "App",
        "marketplace": "Marketplace",
        "tienda": "Tienda",
        "web": "Web",
        "tiktok shop": "TikTok Shop",
    }
    return conocidos.get(clave, texto.title())


def _normalizar_pago(valor: Any) -> Any:
    if pd.isna(valor):
        return pd.NA
    clave = str(_normalizar_llave(pd.Series([valor])).iloc[0])
    conocidos = {
        "efectivo": "Efectivo",
        "paypal": "PayPal",
        "tarjeta": "Tarjeta",
        "transferencia": "Transferencia",
        "contraentrega": "Contraentrega",
    }
    return conocidos.get(clave, str(valor).strip().title())


def _parsear_moneda(valor: Any) -> Any:
    if pd.isna(valor):
        return np.nan
    texto = re.sub(r"(?i)\bCOP\b", "", str(valor)).replace("$", "").replace(" ", "").strip()
    if not texto:
        return np.nan
    if re.fullmatch(r"[+-]?\d{1,3}(?:\.\d{3})+(?:,\d+)?", texto):
        texto = texto.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", texto):
        texto = texto.replace(",", "")
    elif "," in texto and "." not in texto:
        texto = texto.replace(",", ".")
    try:
        return float(texto)
    except ValueError:
        return np.nan


def _parsear_decimal(valor: Any) -> Any:
    if pd.isna(valor):
        return np.nan
    texto = str(valor).strip()
    if not texto:
        return np.nan
    # Con coma y punto a la vez, el punto es separador de miles: "1.234,5" -> 1234.5.
    # Sin coma no se toca nada, para no convertir "12.345" en doce mil trescientos.
    if "," in texto:
        if "." in texto:
            texto = texto.replace(".", "")
        texto = texto.replace(",", ".")
    try:
        return float(texto)
    except ValueError:
        return np.nan


def _convertir_fecha(valores: pd.Series) -> pd.Series:
    texto = valores.astype("string").str.strip().str.lower()
    meses = {
        "enero": "01",
        "febrero": "02",
        "marzo": "03",
        "abril": "04",
        "mayo": "05",
        "junio": "06",
        "julio": "07",
        "agosto": "08",
        "septiembre": "09",
        "setiembre": "09",
        "octubre": "10",
        "noviembre": "11",
        "diciembre": "12",
    }
    texto = texto.str.replace(r"\s+de\s+", "/", regex=True)
    for nombre, numero in meses.items():
        texto = texto.str.replace(nombre, numero, regex=False)
    iso = pd.to_datetime(texto, format="%Y-%m-%d", errors="coerce")
    iso_puntos = pd.to_datetime(texto, format="%Y.%m.%d", errors="coerce")
    dmy = pd.to_datetime(texto, format="%d/%m/%Y", errors="coerce")
    dmy_corto = pd.to_datetime(texto, format="%d/%m/%y", errors="coerce")
    return iso.fillna(iso_puntos).fillna(dmy).fillna(dmy_corto)


def auditar_lote_vs_semestre(
    lote: pd.DataFrame, referencia: dict[str, Any] | None = None, titulo: str = "lote"
) -> dict[str, Any]:
    """Imprime diferencias de esquema, tipos y categorías contra el semestre."""
    ref = referencia or _cargar_referencia()
    esperado = ref["semestre"]
    columnas_lote = set(lote.columns)
    columnas_ref = set(esperado["columnas"])
    solo_lote = sorted(columnas_lote - columnas_ref)
    solo_semestre = sorted(columnas_ref - columnas_lote)
    print(f"\nAuditoria {titulo} frente al semestre ({len(lote):,} filas):")
    print(f"Columnas solo en el lote: {solo_lote or 'ninguna'}")
    print(f"Columnas solo en el semestre: {solo_semestre or 'ninguna'}")

    diferencias_tipos: dict[str, tuple[str, str]] = {}
    tipos_ref = esperado["tipos"]
    for columna in sorted(columnas_lote & set(tipos_ref)):
        tipo_lote = str(lote[columna].dtype)
        tipo_semestre = tipos_ref[columna]
        if tipo_lote != tipo_semestre:
            diferencias_tipos[columna] = (tipo_lote, tipo_semestre)
    print(f"Diferencias de tipos (lote, semestre): {diferencias_tipos or 'ninguna'}")

    normalizadores = {
        "canal_compra": _normalizar_canal,
        "metodo_pago": _normalizar_pago,
        "ciudad_tienda": _normalizar_ciudad,
        "categoria_producto": _normalizar_categoria,
        "nivel_satisfaccion": lambda x: str(x).strip().title(),
        "nivel_lealtad": lambda x: str(x).strip().title(),
    }
    diferencias_categorias: dict[str, dict[str, list[str]]] = {}
    for columna in _COLUMNAS_CATEGORICAS:
        if columna not in lote.columns or columna not in esperado["categorias"]:
            continue
        normalizador = normalizadores[columna]
        categorias_lote = {
            str(normalizador(valor)) for valor in lote[columna].dropna()
        }
        categorias_semestre = set(esperado["categorias"][columna])
        nuevas = sorted(categorias_lote - categorias_semestre)
        ausentes = sorted(categorias_semestre - categorias_lote)
        if nuevas or ausentes:
            diferencias_categorias[columna] = {
                "nuevas_en_lote": nuevas,
                "no_observadas_en_lote": ausentes,
            }
    print(f"Diferencias de categorias: {diferencias_categorias or 'ninguna'}")
    diferencias_medianas: dict[str, dict[str, float | None]] = {}
    for columna, mediana_semestre in esperado["medianas"].items():
        if columna not in lote.columns:
            continue
        if columna == "monto_compra":
            valores = pd.to_numeric(
                lote[columna].map(_parsear_moneda), errors="coerce"
            )
        else:
            valores = pd.to_numeric(lote[columna], errors="coerce")
        mediana_lote = valores.median()
        diferencia = (
            float(mediana_lote - mediana_semestre)
            if pd.notna(mediana_lote) and mediana_semestre is not None
            else None
        )
        diferencias_medianas[columna] = {
            "mediana_lote": float(mediana_lote) if pd.notna(mediana_lote) else None,
            "mediana_semestre": mediana_semestre,
            "diferencia": diferencia,
        }
    print(f"Diferencias de medianas (lote vs. semestre): {diferencias_medianas}")

    diferencias_vacios: dict[str, dict[str, float]] = {}
    for columna, porcentaje_semestre in esperado["porcentaje_vacios"].items():
        if columna not in lote.columns:
            continue
        porcentaje_lote = float(lote[columna].isna().mean() * 100)
        diferencias_vacios[columna] = {
            "porcentaje_lote": porcentaje_lote,
            "porcentaje_semestre": porcentaje_semestre,
            "diferencia_pp": porcentaje_lote - porcentaje_semestre,
        }
    print(f"Diferencias de porcentaje de vacios (puntos porcentuales): {diferencias_vacios}")
    return {
        "columnas_solo_lote": solo_lote,
        "columnas_solo_semestre": solo_semestre,
        "diferencias_tipos": diferencias_tipos,
        "diferencias_categorias": diferencias_categorias,
        "diferencias_medianas": diferencias_medianas,
        "diferencias_vacios": diferencias_vacios,
    }


def integrar_lote(
    lote: pd.DataFrame,
    tabla: pd.DataFrame,
    llave_lote: str,
    llave_tabla: str | None = None,
    nombre: str = "integracion",
) -> pd.DataFrame:
    """Integra una dimension m:1 y verifica cobertura y conservacion de filas."""
    llave_tabla = llave_tabla or llave_lote
    if llave_lote not in lote.columns:
        raise KeyError(f"No existe la llave {llave_lote!r} en el lote.")
    if llave_tabla not in tabla.columns:
        raise KeyError(f"No existe la llave {llave_tabla!r} en la tabla.")

    filas_antes = len(lote)
    clave = "__llave_normalizada_integracion__"
    indicador = "__resultado_integracion__"
    if clave in lote.columns or indicador in lote.columns:
        raise ValueError("El lote usa un nombre reservado para la integracion.")

    izquierda = lote.copy()
    derecha = tabla.copy()
    izquierda[clave] = _normalizar_llave(izquierda[llave_lote])
    derecha[clave] = _normalizar_llave(derecha[llave_tabla])
    derecha = derecha.loc[derecha[clave].notna()].copy()

    repetidas = int(derecha[clave].duplicated(keep=False).sum())
    if repetidas:
        print(
            f"Integracion {nombre}: {repetidas:,} filas de la tabla tienen llaves "
            "normalizadas repetidas."
        )
        raise ValueError(f"La tabla no es unica por {llave_tabla!r}; no se puede validar m:1.")

    columnas_comunes = (
        set(izquierda.columns) & set(derecha.columns)
    ) - {llave_lote, llave_tabla, clave}
    if columnas_comunes:
        raise ValueError(
            f"Columnas en conflicto durante {nombre}: {sorted(columnas_comunes)}."
        )

    integrada = izquierda.merge(
        derecha,
        on=clave,
        how="left",
        validate="m:1",
        indicator=indicador,
        suffixes=("", "_dimension"),
    )
    no_encontradas = int(integrada[indicador].eq("left_only").sum())
    integrada[f"{nombre}_encontrada"] = integrada[indicador].eq("both")
    integrada = integrada.drop(columns=[clave, indicador])
    filas_despues = len(integrada)
    print(
        f"Integracion {nombre}: filas antes={filas_antes:,}, despues={filas_despues:,}; "
        f"llaves sin pareja={no_encontradas:,}; llaves repetidas en tabla={repetidas:,}."
    )
    if filas_despues != filas_antes:
        raise AssertionError(
            f"La integracion {nombre} cambio las filas: {filas_antes:,} -> {filas_despues:,}."
        )
    return integrada


def _agregar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features temporales usando solo compras presentes en el lote."""
    df = df.copy()
    fecha_inscripcion = (
        df["fecha_inscripcion_club"]
        if "fecha_inscripcion_club" in df.columns
        else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    )
    df["antiguedad_cliente_dias"] = (df["fecha_pedido"] - fecha_inscripcion).dt.days
    df["antiguedad_cliente_dias"] = df["antiguedad_cliente_dias"].where(
        df["antiguedad_cliente_dias"].ge(0)
    )
    df["monto_por_unidad"] = df["monto_compra"] / df["unidades_vendidas"].where(
        df["unidades_vendidas"].gt(0)
    )

    columnas_historial = [
        "pedidos_previos_cliente",
        "gasto_previo_cliente",
        "ticket_promedio_previo_cliente",
        "dias_desde_compra_previa",
    ]
    for columna in columnas_historial:
        df[columna] = np.nan

    mascara = df["id_cliente"].notna() & df["fecha_pedido"].notna()
    validos = df.loc[
        mascara, ["id_cliente", "id_pedido", "fecha_pedido", "monto_compra"]
    ].copy()
    if not validos.empty:
        validos["_fila_origen"] = validos.index
        diario = (
            validos.groupby(["id_cliente", "fecha_pedido"], as_index=False, sort=True)
            .agg(
                pedidos_dia=("id_pedido", "nunique"),
                monto_dia=("monto_compra", "sum"),
            )
            .sort_values(["id_cliente", "fecha_pedido"], kind="stable")
        )
        por_cliente = diario.groupby("id_cliente", sort=False)
        diario["pedidos_previos_cliente"] = (
            por_cliente["pedidos_dia"].cumsum() - diario["pedidos_dia"]
        )
        diario["gasto_previo_cliente"] = (
            por_cliente["monto_dia"].cumsum() - diario["monto_dia"]
        )
        diario["fecha_compra_previa"] = por_cliente["fecha_pedido"].shift(1)
        diario["dias_desde_compra_previa"] = (
            diario["fecha_pedido"] - diario["fecha_compra_previa"]
        ).dt.days
        diario["ticket_promedio_previo_cliente"] = (
            diario["gasto_previo_cliente"]
            / diario["pedidos_previos_cliente"].where(
                diario["pedidos_previos_cliente"].gt(0)
            )
        )
        enriquecidas = validos.merge(
            diario[["id_cliente", "fecha_pedido", *columnas_historial]],
            on=["id_cliente", "fecha_pedido"],
            how="left",
            validate="m:1",
        ).set_index("_fila_origen")
        df.loc[enriquecidas.index, columnas_historial] = enriquecidas[
            columnas_historial
        ]

    df["pedidos_previos_cliente"] = df["pedidos_previos_cliente"].astype("Int64")
    dias = df["dias_desde_compra_previa"]
    frecuencia = df["pedidos_previos_cliente"]
    gasto = df["gasto_previo_cliente"]
    recencia = np.select(
        [
            dias.isna().to_numpy(dtype=bool)
            | dias.le(30).fillna(False).to_numpy(dtype=bool),
            dias.le(90).fillna(False).to_numpy(dtype=bool),
        ],
        [3, 2],
        default=1,
    )
    frecuencia_score = np.select(
        [
            frecuencia.ge(5).fillna(False).to_numpy(dtype=bool),
            frecuencia.ge(2).fillna(False).to_numpy(dtype=bool),
        ],
        [3, 2],
        default=1,
    )
    monetario = np.select(
        [
            gasto.ge(1_000_000).fillna(False).to_numpy(dtype=bool),
            gasto.ge(250_000).fillna(False).to_numpy(dtype=bool),
        ],
        [3, 2],
        default=1,
    )
    df["rfm_score"] = pd.Series(
        recencia + frecuencia_score + monetario, index=df.index
    ).where(frecuencia.notna()).astype("float64")
    return df


def limpiar_lote(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el CSV crudo y agrega referencias CRM/bodega y features S13."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("limpiar_lote espera un pandas.DataFrame.")
    referencia = _cargar_referencia()
    lote = _normalizar_nombres_columnas(df)
    filas_iniciales = len(lote)
    print(f"Lote crudo: {filas_iniciales:,} filas, {lote.shape[1]} columnas.")

    requeridas = {
        "id_pedido",
        "id_cliente",
        "fecha_pedido",
        "canal_compra",
        "metodo_pago",
        "ciudad_tienda",
        "codigo_postal",
        "categoria_producto",
        "producto",
        "precio_unitario",
        "unidades_vendidas",
        "monto_compra",
        "edad_cliente",
        "peso_pedido_kg",
        "correo_cliente",
        "nivel_satisfaccion",
        "nivel_lealtad",
        "comentario_cliente",
    }
    faltantes = sorted(requeridas - set(lote.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas del lote: {faltantes}.")

    # 1. Duplicados exactos: no se eliminan pedidos recurrentes por cliente/correo.
    duplicados = int(lote.duplicated().sum())
    lote = lote.drop_duplicates(keep="first").reset_index(drop=True)
    print(
        f"Duplicados exactos: {duplicados:,}; filas tras eliminarlos: {len(lote):,}."
    )
    ids_repetidos = int(lote["id_pedido"].duplicated(keep=False).sum())
    if ids_repetidos:
        raise ValueError(
            f"Quedan {ids_repetidos:,} filas con id_pedido repetido y contenido "
            "distinto; se requiere una decision antes de escoger una."
        )
    if lote["id_pedido"].isna().any():
        raise ValueError("Hay pedidos sin id_pedido; no se puede garantizar una fila por pedido.")

    # 2. Conversiones. Se informa toda conversion que pudo crear valores vacios.
    for columna in ("id_pedido", "id_cliente"):
        antes = lote[columna].copy()
        lote[columna] = lote[columna].astype("string").str.strip()
        _reportar_nulos_nuevos(columna, antes, lote[columna])

    antes = lote["codigo_postal"].copy()
    lote["codigo_postal"] = lote["codigo_postal"].astype("string").str.strip()
    _reportar_nulos_nuevos("codigo_postal", antes, lote["codigo_postal"])

    for columna in ("precio_unitario", "unidades_vendidas", "edad_cliente", "peso_pedido_kg"):
        antes = lote[columna].copy()
        # precio y peso pueden llegar con coma decimal desde otros sistemas.
        if columna in ("peso_pedido_kg", "precio_unitario"):
            lote[columna] = pd.to_numeric(
                lote[columna].map(_parsear_decimal), errors="coerce"
            ).astype("float64")
        else:
            lote[columna] = pd.to_numeric(lote[columna], errors="coerce")
        _reportar_nulos_nuevos(columna, antes, lote[columna])

    antes = lote["monto_compra"].copy()
    lote["monto_compra"] = pd.to_numeric(
        lote["monto_compra"].map(_parsear_moneda), errors="coerce"
    )
    _reportar_nulos_nuevos("monto_compra", antes, lote["monto_compra"])

    for columna in ("fecha_pedido", "fecha_actualizacion_stock", "fecha_llamada_retencion"):
        if columna in lote.columns:
            antes = lote[columna].copy()
            lote[columna] = _convertir_fecha(lote[columna])
            _reportar_nulos_nuevos(columna, antes, lote[columna])

    if "fuga_cliente" in lote.columns:
        antes = lote["fuga_cliente"].copy()
        lote["fuga_cliente"] = pd.to_numeric(lote["fuga_cliente"], errors="coerce").astype(
            "Int64"
        )
        _reportar_nulos_nuevos("fuga_cliente", antes, lote["fuga_cliente"])

    # 3. Texto: variantes conocidas convergen; las categorias nuevas se conservan.
    normalizadores = {
        "canal_compra": _normalizar_canal,
        "metodo_pago": _normalizar_pago,
        "ciudad_tienda": _normalizar_ciudad,
        "categoria_producto": _normalizar_categoria,
        "nivel_satisfaccion": lambda x: (
            pd.NA if pd.isna(x) else str(x).strip().title()
        ),
        "nivel_lealtad": lambda x: (
            pd.NA
            if pd.isna(x)
            else {
                "bronce": "Bronce",
                "plata": "Plata",
                "oro": "Oro",
                "platino": "Platino",
            }.get(str(_normalizar_llave(pd.Series([x])).iloc[0]), str(x).strip().title())
        ),
    }
    for columna, normalizador in normalizadores.items():
        antes = lote[columna].isna().sum()
        lote[columna] = lote[columna].map(normalizador).astype("string")
        nuevos = int(lote[columna].isna().sum() - antes)
        print(
            f"Normalizacion {columna}: {nuevos:,} vacios nuevos; "
            f"{lote[columna].nunique(dropna=True):,} categorias."
        )
    for columna in ("producto", "correo_cliente", "comentario_cliente"):
        lote[columna] = lote[columna].map(_normalizar_texto).astype("string")

    # 4. Valores imposibles: se eliminan registros incompatibles con una venta;
    # edad y peso imposibles quedan nulos con una bandera y conservan el pedido.
    precio = lote["precio_unitario"]
    unidades = lote["unidades_vendidas"]
    monto = lote["monto_compra"]
    precio_invalido = precio.lt(0) | precio.isin([50_000_000, 999_000_000])
    unidades_invalidas = unidades.le(0) | unidades.isin([5000, 9999])
    monto_negativo = monto.lt(0)
    eliminar = precio_invalido.fillna(False) | unidades_invalidas.fillna(False) | monto_negativo.fillna(False)
    print(
        "Valores imposibles de pedido: "
        f"precio negativo/centinela={int(precio_invalido.sum()):,}, "
        f"unidades no positivas/centinela={int(unidades_invalidas.sum()):,}, "
        f"monto negativo={int(monto_negativo.sum()):,}; "
        f"filas eliminadas (interseccion incluida)={int(eliminar.sum()):,}."
    )
    filas_eliminadas_imposibles = int(eliminar.sum())
    lote = lote.loc[~eliminar].copy()

    precio = lote["precio_unitario"]
    unidades = lote["unidades_vendidas"]
    monto = lote["monto_compra"]
    monto_esperado = precio * unidades
    comparables = precio.notna() & unidades.notna() & monto.notna()
    no_coincide = comparables & ~np.isclose(
        monto.fillna(0), monto_esperado.fillna(0), rtol=0, atol=0
    )
    lote["monto_compra_recalculado"] = no_coincide
    print(
        "Regla monto_compra = precio_unitario x unidades_vendidas: "
        f"{int(no_coincide.sum()):,} valores discrepantes; se recalculan."
    )
    lote.loc[no_coincide, "monto_compra"] = monto_esperado.loc[no_coincide]

    edad_imposible = lote["edad_cliente"].notna() & (
        lote["edad_cliente"].lt(0) | lote["edad_cliente"].gt(100)
    )
    peso_imposible = lote["peso_pedido_kg"].notna() & lote["peso_pedido_kg"].gt(100)
    lote["edad_cliente_imposible"] = edad_imposible.astype(bool)
    lote["peso_pedido_kg_imposible"] = peso_imposible.astype(bool)
    lote.loc[edad_imposible, "edad_cliente"] = np.nan
    lote.loc[peso_imposible, "peso_pedido_kg"] = np.nan
    print(
        f"Edad fuera de 0-100: {int(edad_imposible.sum()):,}; "
        f"peso mayor de 100 kg: {int(peso_imposible.sum()):,}; "
        "se conservan los pedidos y se dejan esos valores vacios."
    )

    # 5. Faltantes: se conservan las banderas del semestre y no se ocultan NaT.
    lote["precio_monto_imputado"] = (
        lote["precio_unitario"].isna() | lote["monto_compra"].isna()
    )
    lote["nivel_satisfaccion_imputado"] = lote["nivel_satisfaccion"].isna()
    lote["ciudad_tienda_imputado"] = lote["ciudad_tienda"].isna()
    lote["correo_cliente_imputado"] = lote["correo_cliente"].isna()
    lote["comentario_cliente_imputado"] = lote["comentario_cliente"].isna()

    mediana_precio = referencia["medianas"]["precio_unitario"]
    if mediana_precio is None or not np.isfinite(mediana_precio):
        raise ValueError("La referencia del semestre no contiene una mediana de precio valida.")
    unidades_validas = lote["unidades_vendidas"].gt(0)
    derivar_precio = lote["precio_unitario"].isna() & lote["monto_compra"].notna() & unidades_validas
    lote.loc[derivar_precio, "precio_unitario"] = (
        lote.loc[derivar_precio, "monto_compra"]
        / lote.loc[derivar_precio, "unidades_vendidas"]
    )
    derivar_monto = lote["monto_compra"].isna() & lote["precio_unitario"].notna() & unidades_validas
    lote.loc[derivar_monto, "monto_compra"] = (
        lote.loc[derivar_monto, "precio_unitario"]
        * lote.loc[derivar_monto, "unidades_vendidas"]
    )
    precio_pendiente = lote["precio_unitario"].isna()
    lote.loc[precio_pendiente, "precio_unitario"] = mediana_precio
    monto_pendiente = lote["monto_compra"].isna() & unidades_validas
    lote.loc[monto_pendiente, "monto_compra"] = (
        lote.loc[monto_pendiente, "precio_unitario"]
        * lote.loc[monto_pendiente, "unidades_vendidas"]
    )
    print(
        f"Precio imputado con mediana semestral ({mediana_precio:,.2f}): "
        f"{int(precio_pendiente.sum()):,}; monto recalculado/imputado: "
        f"{int(derivar_monto.sum() + monto_pendiente.sum()):,}."
    )

    modas = referencia["moda_satisfaccion_por_canal"]
    moda_global = referencia["moda_satisfaccion_global"]
    antes_satisfaccion = int(lote["nivel_satisfaccion"].isna().sum())
    for canal, indices in lote.loc[
        lote["nivel_satisfaccion"].isna()
    ].groupby("canal_compra", dropna=False).groups.items():
        canal_key = str(canal) if not pd.isna(canal) else ""
        lote.loc[indices, "nivel_satisfaccion"] = modas.get(canal_key, moda_global)
    lote["nivel_satisfaccion"] = lote["nivel_satisfaccion"].fillna(moda_global)
    print(
        f"Satisfaccion imputada por moda semestral del canal: "
        f"{antes_satisfaccion:,} (incluye canal nuevo con moda global si aplica)."
    )

    faltaban_correo = int(lote["correo_cliente"].isna().sum())
    lote["correo_cliente"] = lote["correo_cliente"].fillna(
        "sin_registro@desconocido.com"
    )
    faltaban_comentarios = int(lote["comentario_cliente"].isna().sum())
    lote["comentario_cliente"] = lote["comentario_cliente"].fillna("Sin comentario")
    print(
        f"Correo completado con marcador explicito: {faltaban_correo:,}; "
        f"comentarios completados: {faltaban_comentarios:,}; "
        f"ciudad sin imputar (con indicador): {int(lote['ciudad_tienda'].isna().sum()):,}."
    )

    # Se conserva la politica semestral de IQR: se marca venta corporativa y
    # solo se capa el grupo de 2-3 unidades sobre el umbral de referencia.
    limite = referencia.get("limite_superior_iqr_monto")
    lote["es_venta_corporativa"] = (
        lote["monto_compra"].gt(limite) & lote["unidades_vendidas"].isin([4, 5])
        if limite is not None
        else False
    )
    lote["monto_compra_capado"] = False
    if limite is not None:
        capar = lote["monto_compra"].gt(limite) & lote["unidades_vendidas"].isin([2, 3])
        lote.loc[capar, "monto_compra"] = limite
        lote.loc[capar, "monto_compra_capado"] = True
        print(
            f"Outliers segun IQR semestral: corporativas marcadas="
            f"{int(lote['es_venta_corporativa'].sum()):,}; "
            f"montos capados con limite {limite:,.2f}={int(capar.sum()):,}."
        )

    # 6. Integracion S12; validate='m:1' evita que una dimension duplique pedidos.
    if not _CRM.exists() or not _BODEGAS.exists():
        raise FileNotFoundError(
            f"Se requieren las dimensiones de S12: {_CRM} y {_BODEGAS}."
        )
    crm = pd.read_csv(_CRM, encoding="utf-8-sig")
    bodegas = pd.read_csv(_BODEGAS, encoding="utf-8-sig")
    lote = integrar_lote(
        lote, crm, "correo_cliente", "email_cliente", nombre="crm"
    )
    if "fecha_inscripcion_club" in lote.columns:
        antes = lote["fecha_inscripcion_club"].copy()
        lote["fecha_inscripcion_club"] = _convertir_fecha(
            lote["fecha_inscripcion_club"]
        )
        _reportar_nulos_nuevos("fecha_inscripcion_club", antes, lote["fecha_inscripcion_club"])
    lote = integrar_lote(
        lote,
        bodegas,
        "categoria_producto",
        "categoria",
        nombre="bodega",
    )

    lote = _agregar_features(lote)
    print(
        "Features de historial calculadas solo dentro del lote: "
        f"{int(lote['pedidos_previos_cliente'].notna().sum()):,} filas con fecha/cliente."
    )
    # La fecha de llamada ocurre despues del pedido; se excluye por riesgo de fuga.
    if "fecha_llamada_retencion" in lote.columns:
        lote = lote.drop(columns="fecha_llamada_retencion")
        print("Columna excluida por ser posterior al pedido: fecha_llamada_retencion.")
    # El correo se usa para la union CRM, pero no se entrega como dato personal.
    lote = lote.drop(columns=["correo_cliente", "email_cliente"], errors="ignore")
    print(
        "Columnas personales usadas solo para CRM y excluidas de la salida: "
        "correo_cliente, email_cliente."
    )

    lote["id_pedido"] = lote["id_pedido"].astype("string")
    lote["id_cliente"] = lote["id_cliente"].astype("string")
    lote["unidades_vendidas"] = lote["unidades_vendidas"].astype("Int64")
    lote["edad_cliente"] = lote["edad_cliente"].astype("Int64")
    if "fuga_cliente" in lote.columns:
        lote["fuga_cliente"] = lote["fuga_cliente"].astype("Int64")

    # Conteos reales de la corrida, para que el acta no los estime restando tamanos.
    lote.attrs["filas_eliminadas_imposibles"] = filas_eliminadas_imposibles
    lote.attrs["duplicados_exactos_eliminados"] = duplicados

    if lote["id_pedido"].duplicated().any():
        raise AssertionError("La limpieza no dejo una fila por id_pedido.")
    if len(lote) != len(lote.drop_duplicates()):
        raise AssertionError("Quedaron filas exactamente duplicadas tras la limpieza.")
    print(
        f"Limpieza finalizada: {len(lote):,} filas (inicio {filas_iniciales:,}), "
        f"{lote.shape[1]} columnas; pedidos unicos={lote['id_pedido'].nunique():,}."
    )
    return lote.reset_index(drop=True)
