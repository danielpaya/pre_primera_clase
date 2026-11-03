# Documento de versión — NovaMarket

- **Versión:** `novamarket_pipeline_v1.joblib`
- **Fecha:** 2026-11-03
- **Origen:** `S14/novamarket_features_seleccionadas.csv` (551.790 filas)
- **Objetivo:** `fuga_cliente`; `id_cliente` se excluye de los predictores.

Esta versión usa el dataset original solicitado de S14, no `S15/novamarket_train_balanceado.csv`; por tanto, mantiene la distribución de clases observada (20,3% de fuga) y no aplica SMOTE. El resultado reportado corresponde a esa línea base.

## Columnas y preprocesamiento

El `ColumnTransformer` aplica imputación por mediana y `StandardScaler` a estas columnas continuas:

- `monto_compra`
- `puntos_fidelidad_actuales`
- `monto_por_unidad`
- `unidades_vendidas`
- `ticket_promedio_previo_cliente`
- `edad_cliente`
- `antiguedad_cliente_dias`
- `dias_desde_compra_previa`

No hay columnas binarias entre las ocho features predictoras de este CSV, por lo que `columnas_binarias` está vacía. La etiqueta y el identificador no forman parte del preprocesamiento.

## Validación y evaluación

La regresión logística de validación se evaluó con validación cruzada de 5 folds sobre el conjunto de entrenamiento, usando `scoring="recall"`:

- **Recall promedio:** 0,0000
- **Desviación estándar:** 0,0000
- **Prevalencia de fuga observada:** aproximadamente 20,3% (no 9%)

El resultado no es aceptable para detectar clientes en fuga: el modelo simple no identificó positivos en ninguno de los folds. Debe tratarse como una línea base reproducible, no como un modelo listo para decisiones de retención. Aunque la exactitud (`score`) en prueba fue 79,69%, esa cifra es engañosa en este caso y no compensa el recall nulo.

## Desbalance y siguiente iteración hipotética con S15

Se mantiene S14 porque es el dataset solicitado para esta versión; su prevalencia positiva es aproximadamente 20,3%. Por eso la exactitud no basta para juzgar el modelo, y el recall nulo indica que la regresión logística base no detectó la clase minoritaria. El resultado se conserva como diagnóstico de la línea base, no como motivo para sustituir silenciosamente el dataset de esta entrega.

Como experimento posterior, se probaría SMOTE, la estrategia elegida en S15 para balancear el entrenamiento, comparándola con esta línea base. La decisión sería conservar SMOTE solo si mejora el recall y mantiene una precisión aceptable para el costo operativo de las alertas; también se revisarían F1 y PR-AUC. La validación y la prueba deben conservar la distribución original, y el conjunto de prueba se reservaría para la evaluación final. Para evitar fuga, SMOTE se aplicaría dentro de cada fold y solo sobre su partición de entrenamiento mediante un pipeline de `imblearn`; no se haría validación cruzada directamente sobre un CSV sobremuestreado antes de separar los folds.

Se separó el 20% de los registros para prueba con `random_state=42` y `stratify=y`. El ajuste final (`fit`) del pipeline se realizó únicamente con `X_entrenamiento` y `y_entrenamiento`; `X_prueba` y `y_prueba` solo se usaron para evaluación con `score`. El pipeline serializado se cargó de nuevo y produjo exactamente el mismo puntaje de prueba que el objeto original.

El artefacto se generó con `scikit-learn 1.8.0` y `joblib 1.5.3`. Para evitar incompatibilidades al cargarlo, se recomienda usar la misma versión de scikit-learn que la usada para serializarlo.
